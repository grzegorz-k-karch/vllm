# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AnyModel: dynamic-parent wrapper for NAS-optimized heterogeneous models.

At init time, ``__new__`` creates a wrapper subclass inheriting from both
AnyModel and the target base model (e.g. ``LlamaForCausalLM``).  The base
model builds the full structure with the global config; then
``_patch_anymodel_layers`` replaces layers whose ``per_layer_config``
entries carry overrides and injects no-ops for entries that list skipped
sub-module groups.  VL models work automatically (vision tower, forward,
load_weights all inherited).

The per-layer config follows the HuggingFace heterogeneity schema: a list
of flat dicts, each mapping top-level parent-config keys to per-layer
overrides.  Skipped sub-module groups are listed under the ``"skip"`` key
(e.g. ``{"skip": ["attention"]}`` or ``{"skip": ["attention", "mlp"]}``).

No-op layers are replaced with identity pass-throughs (``NoOpAttention``,
``NoOpMLP``) paired with ``NoOpNorm``.  The identity approach defers the
residual add to the next real norm, avoiding in-place aliasing issues from
vLLM's fused ``fused_add_rms_norm`` kernel.
"""

from __future__ import annotations

import copy
import functools
import importlib
import importlib.util
import inspect
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from typing import ClassVar

import regex as re
import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.logger import init_logger

from .interfaces import HasNoOps
from .utils import PPMissingLayer, maybe_prefix

logger = init_logger(__name__)

# Security: all config-supplied module paths must resolve within this package.
_VLLM_MODELS_PKG = "vllm.model_executor.models"
# Security: only simple identifier segments allowed in layers_path.
_SAFE_ATTR_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_layers_path(layers_path: str) -> None:
    """Reject dunder or non-identifier segments in a config-supplied layers_path."""
    for part in layers_path.split("."):
        if not _SAFE_ATTR_RE.match(part):
            raise ValueError(
                f"Security: invalid attribute segment in layers_path: {part!r}"
            )
        if part.startswith("__"):
            raise ValueError(
                f"Security: dunder attributes not allowed in layers_path: {part!r}"
            )


def _validate_config_arch_info(arch_info: ArchInfo) -> None:
    """Ensure config-supplied module paths resolve inside vllm.model_executor.models."""
    for field_name in ("decoder_layer_module", "base_model_module"):
        val = getattr(arch_info, field_name, None)
        if val is None:
            continue
        try:
            abs_name = importlib.util.resolve_name(val, _VLLM_MODELS_PKG)
        except (ImportError, ValueError) as exc:
            raise ValueError(
                f"Security: ArchInfo.{field_name} is not a valid relative "
                f"module path: {val!r}"
            ) from exc
        if not abs_name.startswith(_VLLM_MODELS_PKG + "."):
            raise ValueError(
                f"Security: ArchInfo.{field_name} must resolve within "
                f"'{_VLLM_MODELS_PKG}'. Got resolved name: {abs_name!r}"
            )
    _validate_layers_path(arch_info.layers_path)


# Module-group names used in the HF heterogeneity "skip" list.
_SKIP_GROUP_ATTENTION = "attention"
_SKIP_GROUP_MLP = "mlp"


def _layer_entry_items(entry) -> dict:
    """Return a dict view of a per_layer_config entry.

    HF may yield either a plain dict (from JSON load) or a ``LayerConfig``
    namespace (from ``HeterogeneousConfig``).
    """
    if isinstance(entry, dict):
        return entry
    return dict(vars(entry))


def _layer_skip_set(entry) -> set[str]:
    """Set of module-group names this layer skips (e.g. ``{"attention"}``)."""
    skip = _layer_entry_items(entry).get("skip") or ()
    return set(skip)


def _raw_per_layer_overrides(per_layer_config):
    """Return sparse layer overrides from HF's dict or sequence view."""
    if per_layer_config is None:
        return None
    if isinstance(per_layer_config, Mapping):
        return per_layer_config

    # Transformers v5 exposes config.per_layer_config as a Sequence of
    # resolved per-layer configs; the original sparse mapping is kept on the
    # parent config's private heterogeneity spec.
    parent_config = getattr(per_layer_config, "_config", None)
    heterogeneity_spec = getattr(parent_config, "_heterogeneity_spec", None)
    return getattr(heterogeneity_spec, "per_layer_overrides", None)


def _iter_layer_overrides(per_layer_config):
    """Yield ``(layer_idx, layer_overrides)`` from a sparse per_layer_config.

    JSON-loaded configs have string keys; Python-constructed ones may use
    ints.  Coerces to int and yields in ascending layer-index order.
    """
    raw_overrides = _raw_per_layer_overrides(per_layer_config)
    if raw_overrides is None:
        return
    for key, entry in sorted(raw_overrides.items(), key=lambda kv: int(kv[0])):
        yield int(key), entry


class NoOpAttention(nn.Module):
    """Identity pass-through replacing a skipped attention block."""

    def forward(self, *args, **kwargs) -> torch.Tensor:
        if "hidden_states" in kwargs:
            if "output" in kwargs:
                kwargs["output"].copy_(kwargs["hidden_states"])
                return kwargs["output"]
            return kwargs["hidden_states"]
        return args[0]


class NoOpMLP(nn.Module):
    """Identity pass-through replacing a skipped feed-forward block."""

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return hidden_states


class NoOpNorm(nn.Module):
    """Identity norm for layers adjacent to no-op attention/MLP.

    Two-arg path: returns both inputs unchanged (defers residual accumulation).
    Single-arg path (first layer, residual is None): returns zeros to break
    tensor aliasing that would cause fused_add_rms_norm to double the residual.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            return hidden_states, residual
        return torch.zeros_like(hidden_states)


@dataclass
class ArchInfo:
    """Per-architecture descriptor for building and patching layers."""

    # Decoder layer class
    decoder_layer_module: str  # dotted module path, e.g. ".llama"
    decoder_layer_class: str  # fallback class name

    # Hybrid / multi-type layer support
    decoder_layer_class_map: dict[str, str] | None = None
    hybrid_pattern_field: str | None = None

    # Sub-module attribute names on the decoder layer.  Used to decide which
    # sub-modules to replace with no-ops when a layer's ``"skip"`` list
    # contains the corresponding group name.
    attn_module: str = "self_attn"
    attn_norm_module: str = "input_layernorm"
    ffn_module: str = "mlp"
    ffn_norm_module: str = "post_attention_layernorm"

    # Dynamic-parent / multimodal support
    base_model_module: str | None = None  # None = use decoder_layer_module
    layers_path: str = "model.layers"
    init_prefix: str | None = None  # None = inherit from engine
    layer_hf_config: str | None = None  # None = use hf_config directly


def _resolve_layer_base_config(config, arch_info: "ArchInfo"):
    """Return the config that carries ``per_layer_config`` / ``layer_types``.

    ``arch_info.layer_hf_config`` names a sub-config (e.g. ``"text_config"``)
    on the top-level multimodal config. Depending on the call site, ``config``
    may be that top-level config (during ``__init__``) or already the resolved
    sub-config (``self.config`` during ``load_weights``, which the base text
    model sets to its own text config). Handle both: use the named sub-config
    when present, otherwise fall back to ``config`` itself.
    """
    key = arch_info.layer_hf_config
    if not key:
        return config
    sub = getattr(config, key, None)
    return sub if sub is not None else config


_ARCH_REGISTRY: dict[str, ArchInfo] = {
    # Dense: Llama family
    "LlamaForCausalLM": ArchInfo(
        decoder_layer_module=".llama",
        decoder_layer_class="LlamaDecoderLayer",
    ),
    "MistralForCausalLM": ArchInfo(
        decoder_layer_module=".mistral",
        decoder_layer_class="MistralDecoderLayer",
    ),
    # Dense: Qwen2/3 family
    "Qwen2ForCausalLM": ArchInfo(
        decoder_layer_module=".qwen2",
        decoder_layer_class="Qwen2DecoderLayer",
    ),
    "Qwen3ForCausalLM": ArchInfo(
        decoder_layer_module=".qwen3",
        decoder_layer_class="Qwen3DecoderLayer",
    ),
    # MoE: Qwen family
    "Qwen2MoeForCausalLM": ArchInfo(
        decoder_layer_module=".qwen2_moe",
        decoder_layer_class="Qwen2MoeDecoderLayer",
    ),
    # MoE: Mixtral family
    "MixtralForCausalLM": ArchInfo(
        decoder_layer_module=".mixtral",
        decoder_layer_class="MixtralDecoderLayer",
        ffn_module="block_sparse_moe",
    ),
    # Hybrid: NemotronH
    "NemotronHForCausalLM": ArchInfo(
        decoder_layer_module=".nemotron_h",
        decoder_layer_class="NemotronHAttentionDecoderLayer",
        decoder_layer_class_map={
            "*": "NemotronHAttentionDecoderLayer",
            "-": "NemotronHMLPDecoderLayer",
            "E": "NemotronHMoEDecoderLayer",
            "M": "NemotronHMambaDecoderLayer",
        },
        hybrid_pattern_field="hybrid_override_pattern",
        attn_module="mixer",
        attn_norm_module="norm",
        ffn_module="mixer",
        ffn_norm_module="norm",
    ),
    # MoE: GptOss
    "GptOssForCausalLM": ArchInfo(
        decoder_layer_module=".gpt_oss",
        decoder_layer_class="TransformerBlock",
        attn_module="attn",
    ),
    # Multimodal: Qwen3VL
    "Qwen3VLForConditionalGeneration": ArchInfo(
        decoder_layer_module=".qwen3",
        decoder_layer_class="Qwen3DecoderLayer",
        base_model_module=".qwen3_vl",
        layers_path="language_model.model.layers",
        init_prefix="model",
        layer_hf_config="text_config",
    ),
}

_ARCH_REGISTRY["NemotronHPuzzleForCausalLM"] = _ARCH_REGISTRY["NemotronHForCausalLM"]


def _resolve_layer_class(
    info: ArchInfo,
    global_config,
    layer_idx: int,
) -> type[nn.Module]:
    """Return the decoder layer class for this position (hybrid-aware)."""
    class_name = info.decoder_layer_class
    if info.decoder_layer_class_map and info.hybrid_pattern_field:
        pattern = getattr(global_config, info.hybrid_pattern_field, "") or ""
        if layer_idx < len(pattern):
            class_name = info.decoder_layer_class_map.get(
                pattern[layer_idx], class_name
            )
    try:
        mod = importlib.import_module(info.decoder_layer_module, package=__package__)
        return getattr(mod, class_name)
    except (ImportError, AttributeError) as exc:
        raise type(exc)(
            f"Failed to resolve layer class {class_name!r} from "
            f"module {info.decoder_layer_module!r} for layer {layer_idx}: {exc}"
        ) from exc


def _create_layer_config(global_config, layer_overrides, info: ArchInfo):
    """Deep-copy global_config with per-layer overrides applied.

    ``layer_overrides`` is a flat mapping of top-level parent-config keys to
    per-layer values (HF heterogeneity schema).  The ``"skip"`` key is
    handled separately by ``_apply_no_ops`` and ignored here.
    """
    config = copy.deepcopy(global_config)
    for key, val in _layer_entry_items(layer_overrides).items():
        if key == "skip":
            continue
        setattr(config, key, val)
    return config


def _attention_module_name(layer: nn.Module, info: ArchInfo) -> str:
    """Resolve the concrete attention attribute of a heterogeneous layer."""
    candidates = (info.attn_module, "linear_attn", "self_attn", "attention")
    for name in dict.fromkeys(candidates):
        if hasattr(layer, name):
            return name
    return info.attn_module


def _apply_no_ops(layer: nn.Module, layer_overrides, info: ArchInfo) -> None:
    """Replace sub-modules with no-ops per the layer's ``"skip"`` list."""
    skip = _layer_skip_set(layer_overrides)
    attn_noop = _SKIP_GROUP_ATTENTION in skip
    ffn_noop = _SKIP_GROUP_MLP in skip

    attn_module = _attention_module_name(layer, info)
    shared_module = attn_module == info.ffn_module
    shared_norm = info.attn_norm_module == info.ffn_norm_module

    if shared_module:
        if attn_noop and ffn_noop:
            setattr(layer, attn_module, NoOpAttention())
        if shared_norm and attn_noop and ffn_noop:
            setattr(layer, info.attn_norm_module, NoOpNorm())
    else:
        if attn_noop:
            setattr(layer, attn_module, NoOpAttention())
            setattr(layer, info.attn_norm_module, NoOpNorm())
        if ffn_noop:
            setattr(layer, info.ffn_module, NoOpMLP())
            if not shared_norm or attn_noop:
                setattr(layer, info.ffn_norm_module, NoOpNorm())


def _collect_noop_prefixes(
    per_layer_config: dict, info: ArchInfo, model: nn.Module
) -> frozenset[str]:
    """Build weight-name prefixes (ending with '.') for no-op sub-modules."""
    prefixes: set[str] = set()
    shared_norm = info.attn_norm_module == info.ffn_norm_module

    layers = model
    for part in info.layers_path.split("."):
        layers = getattr(layers, part)

    for idx, entry in _iter_layer_overrides(per_layer_config):
        lp = f"{info.layers_path}.{idx}"
        skip = _layer_skip_set(entry)
        attn_noop = _SKIP_GROUP_ATTENTION in skip
        ffn_noop = _SKIP_GROUP_MLP in skip
        layer = layers[idx] if 0 <= idx < len(layers) else None
        attn_module = (
            _attention_module_name(layer, info)
            if layer is not None
            else info.attn_module
        )
        shared_module = attn_module == info.ffn_module

        if shared_module:
            if attn_noop and ffn_noop:
                prefixes.add(f"{lp}.{attn_module}.")
                prefixes.add(f"{lp}.{info.attn_norm_module}.")
        else:
            if attn_noop:
                prefixes.add(f"{lp}.{attn_module}.")
                prefixes.add(f"{lp}.{info.attn_norm_module}.")
            if ffn_noop:
                prefixes.add(f"{lp}.{info.ffn_module}.")
                if not shared_norm or attn_noop:
                    prefixes.add(f"{lp}.{info.ffn_norm_module}.")
    return frozenset(prefixes)


@functools.cache
def _layer_init_params(layer_cls: type) -> frozenset[str]:
    return frozenset(inspect.signature(layer_cls.__init__).parameters)


def _instantiate_layer(
    layer_cls: type[nn.Module],
    vllm_config: VllmConfig,
    prefix: str,
    per_layer_config,
    layer_idx: int,
) -> nn.Module:
    """Instantiate a decoder layer, patching vllm_config with per-layer config."""
    from vllm.transformers_utils.config import get_hf_text_config

    mock_mc = copy.copy(vllm_config.model_config)
    mock_mc.hf_config = per_layer_config
    # ``hf_text_config`` is a cached field on ModelConfig (set once at init),
    # not recomputed from ``hf_config``. Decoder layers that read
    # ``vllm_config.model_config.hf_text_config`` (e.g. Qwen3.5 / Qwen3-Next)
    # would otherwise see the stale global config and ignore per-layer
    # overrides such as ``num_key_value_heads``. ``per_layer_config`` is the
    # resolved (text) config that carries the override, so point both at it.
    mock_mc.hf_text_config = get_hf_text_config(per_layer_config)
    mock_vc = copy.copy(vllm_config)
    mock_vc.model_config = mock_mc

    _pool = {
        "config": per_layer_config,
        "vllm_config": mock_vc,
        "cache_config": vllm_config.cache_config,
        "quant_config": vllm_config.quant_config,
        "parallel_config": vllm_config.parallel_config,
        "model_config": mock_mc,
        "prefix": prefix,
        "layer_idx": layer_idx,
    }
    params = _layer_init_params(layer_cls)
    if "layer_type" in params:
        layer_types = getattr(per_layer_config, "layer_types", None)
        if layer_types is None or layer_idx >= len(layer_types):
            raise ValueError(
                f"Layer class {layer_cls.__name__} requires layer_type, "
                "but config.layer_types is missing or too short."
            )
        _pool["layer_type"] = layer_types[layer_idx]
    return layer_cls(**{k: v for k, v in _pool.items() if k in params})


def _has_overrides(layer_overrides) -> bool:
    """True iff this layer carries any config override beyond ``"skip"``.

    HF canonicalizes the heterogeneity schema at construction time so
    per-layer entries only retain attributes that genuinely differ from the
    global config; any non-skip key therefore implies a layer rebuild.
    """
    return any(key != "skip" for key in _layer_entry_items(layer_overrides))


def _unregister_layer(layer_prefix: str, vllm_config: VllmConfig) -> None:
    """Remove a layer's entries from static_forward_context and
    static_all_moe_layers to avoid stale references after replacement."""
    cc = vllm_config.compilation_config
    prefix_dot = layer_prefix + "."

    stale = [k for k in cc.static_forward_context if k.startswith(prefix_dot)]
    for k in stale:
        del cc.static_forward_context[k]

    cc.static_all_moe_layers = [
        k for k in cc.static_all_moe_layers if not k.startswith(prefix_dot)
    ]


def _patch_anymodel_layers(
    model: nn.Module,
    vllm_config: VllmConfig,
    arch_info: ArchInfo,
    base_init_prefix: str,
) -> None:
    """Post-init: rebuild layers with overrides and apply no-ops."""
    config = vllm_config.model_config.hf_config
    layer_base_config = _resolve_layer_base_config(config, arch_info)
    per_layer_config = layer_base_config.per_layer_config

    obj = model
    for part in arch_info.layers_path.split("."):
        obj = getattr(obj, part)
    layers: nn.ModuleList = obj
    layers_prefix = maybe_prefix(base_init_prefix, arch_info.layers_path)

    for layer_idx, layer_overrides in _iter_layer_overrides(per_layer_config):
        if not 0 <= layer_idx < len(layers):
            logger.warning(
                "per_layer_config has entry for layer %d but model has %d "
                "layers; ignoring.",
                layer_idx,
                len(layers),
            )
            continue
        layer = layers[layer_idx]
        if isinstance(layer, PPMissingLayer):
            continue

        if _has_overrides(layer_overrides):
            layer_config = _create_layer_config(
                layer_base_config, layer_overrides, arch_info
            )
            layer_cls = _resolve_layer_class(arch_info, layer_base_config, layer_idx)
            layer_prefix = f"{layers_prefix}.{layer_idx}"
            _unregister_layer(layer_prefix, vllm_config)
            target_device = next(layer.parameters()).device
            with torch.device("cpu"):
                new_layer = _instantiate_layer(
                    layer_cls,
                    vllm_config,
                    layer_prefix,
                    layer_config,
                    layer_idx,
                )
            layers[layer_idx] = new_layer
            del layer
            new_layer.to(target_device)
            layer = new_layer

        _apply_no_ops(layer, layer_overrides, arch_info)

        layer_prefix = maybe_prefix(layers_prefix, str(layer_idx))
        skip = _layer_skip_set(layer_overrides)
        attn_noop = _SKIP_GROUP_ATTENTION in skip
        ffn_noop = _SKIP_GROUP_MLP in skip
        attn_module = _attention_module_name(layer, arch_info)
        shared_module = attn_module == arch_info.ffn_module

        if shared_module:
            if attn_noop and ffn_noop:
                _unregister_layer(
                    f"{layer_prefix}.{attn_module}", vllm_config
                )
        else:
            if attn_noop:
                _unregister_layer(
                    f"{layer_prefix}.{attn_module}", vllm_config
                )
            if ffn_noop:
                _unregister_layer(f"{layer_prefix}.{arch_info.ffn_module}", vllm_config)


def _arch_info_from_config(hf_config) -> ArchInfo | None:
    """Load ArchInfo from hf_config.anymodel_arch_info if present."""
    data = getattr(hf_config, "anymodel_arch_info", None)
    if not data:
        return None
    if not isinstance(data, dict):
        data = vars(data)
    known = {f.name for f in dataclass_fields(ArchInfo)}
    arch_info = ArchInfo(**{k: v for k, v in data.items() if k in known})
    _validate_config_arch_info(arch_info)
    return arch_info


# Class-level hooks a hybrid (mamba) base model must expose for vLLM to compute
# its KV-cache spec and (in ``mamba_cache_mode=align``) its state-copy buffers.
# A text-only base class (e.g. Qwen3_5ForCausalLM) may omit these even though it
# has linear_attention layers, while a sibling in the same module (the
# multimodal *ForConditionalGeneration) defines them.
_HYBRID_STATE_HOOKS = (
    "get_mamba_state_shape_from_config",
    "get_mamba_state_dtype_from_config",
    "get_mamba_state_copy_func",
)


def _borrow_hybrid_state_hooks(mod, base_cls) -> dict:
    """Return hooks the base class is missing, sourced from a module sibling.

    These hooks are ``@classmethod``s that read only from ``vllm_config``, so a
    version bound to a sibling class works unchanged on the wrapper.

    Only borrow from a *concrete* implementation: the ``IsHybrid`` Protocol
    (imported into the module namespace, so it precedes the concrete model
    classes in ``vars(mod)`` insertion order) declares ``...``-bodied stubs for
    some of these hooks. Borrowing such a stub yields a method that silently
    returns ``None`` -- e.g. ``get_mamba_state_shape_from_config`` would return
    ``None`` while ``get_mamba_state_dtype_from_config`` (absent from the
    Protocol) correctly resolved to the real class, leaving ``MambaSpec`` to
    crash on ``zip(None, dtypes)``. Skipping ``Protocol`` siblings makes both
    hooks resolve to the same concrete sibling (e.g.
    ``Qwen3_5ForConditionalGeneration``).
    """
    borrowed: dict = {}
    for hook in _HYBRID_STATE_HOOKS:
        if hasattr(base_cls, hook):
            continue
        for sibling in vars(mod).values():
            if (
                isinstance(sibling, type)
                and not getattr(sibling, "_is_protocol", False)
                and hook in sibling.__dict__
            ):
                borrowed[hook] = getattr(sibling, hook)
                break
    return borrowed


def _make_wrapper_cls(arch_name: str, arch_info: ArchInfo) -> type:
    """Create ``AnyModel{arch_name}(AnyModel, BaseModelCls)`` wrapper."""
    base_mod_path = arch_info.base_model_module or arch_info.decoder_layer_module
    mod = importlib.import_module(base_mod_path, package=__package__)
    base_cls = getattr(mod, arch_name)
    namespace = {"_anymodel_arch_info": arch_info, "has_noops": True}
    namespace.update(_borrow_hybrid_state_hooks(mod, base_cls))
    return type(
        f"AnyModel{arch_name}",
        (AnyModel, base_cls),
        namespace,
    )


def resolve_base_model_cls(hf_config) -> type | None:
    """Resolve the concrete base model class an AnyModel config wraps.

    Mirrors the resolution used to build the wrapper (``base_architecture`` +
    ``anymodel_arch_info.base_model_module``) but returns only the base class,
    e.g. ``Qwen3_5ForCausalLM``. Returns ``None`` if it cannot be resolved.

    This lets callers inspect the base model's capabilities (runner type,
    generation support, ...) directly when ``base_architecture`` is not present
    in vLLM's model registry -- as is the case for text-only LMs whose only
    registered architecture is the multimodal ``*ForConditionalGeneration``.
    """
    try:
        arch_name, arch_info, _ = AnyModel._resolve_arch(hf_config)
        base_mod_path = arch_info.base_model_module or arch_info.decoder_layer_module
        mod = importlib.import_module(base_mod_path, package=__package__)
        return getattr(mod, arch_name)
    except Exception:
        return None


def _expand_noop_prefixes_for_mapper(
    prefixes: frozenset[str], model_cls: type
) -> frozenset[str]:
    """Expand noop prefixes with HF-style equivalents via the weight mapper.

    ``AnyModel.load_weights`` filters weights *before* the base model's
    ``hf_to_vllm_mapper`` is applied, so noop prefixes (which use vLLM
    names) won't match HF-style checkpoint names.  This reverse-maps the
    prefixes so both naming conventions are covered.
    """
    mapper = getattr(model_cls, "hf_to_vllm_mapper", None)
    if mapper is None:
        return prefixes

    expanded: set[str] = set(prefixes)
    for orig, new in getattr(mapper, "orig_to_new_prefix", {}).items():
        if new is None:
            continue
        for p in prefixes:
            if p.startswith(new):
                expanded.add(orig + p[len(new) :])
    for orig, new in getattr(mapper, "orig_to_new_substr", {}).items():
        if new is None:
            continue
        for p in list(expanded):
            if new in p:
                expanded.add(p.replace(new, orig, 1))
    return frozenset(expanded)


class AnyModel(nn.Module, HasNoOps):
    """Factory entry point for NAS-optimized heterogeneous models.

    ``__new__`` creates a wrapper subclass ``(AnyModel, BaseModelCls)``.
    ``__init__`` delegates to the base model, then patches layers per
    ``per_layer_config``.  All base model methods are inherited automatically.
    """

    has_noops = True
    _anymodel_arch_info: ClassVar[ArchInfo | None] = None
    _wrapper_cache: ClassVar[dict[str, type[AnyModel]]] = {}

    @staticmethod
    def _resolve_arch(config) -> tuple[str, ArchInfo, bool]:
        """Return (arch_name, arch_info, from_config) for the given hf_config."""
        arch_name = getattr(config, "base_architecture", None)
        if not arch_name:
            raise ValueError(
                "AnyModel config must set 'base_architecture' to the name "
                "of the underlying model class (e.g. 'LlamaForCausalLM')."
            )

        arch_info = _arch_info_from_config(config)
        from_config = arch_info is not None
        if arch_info is None:
            arch_info = _ARCH_REGISTRY.get(arch_name)

        if arch_info is None:
            raise ValueError(
                f"No AnyModel support for architecture {arch_name!r}. "
                f"Supported: {sorted(_ARCH_REGISTRY)}"
            )

        return arch_name, arch_info, from_config

    @staticmethod
    def _get_or_create_wrapper(
        arch_name: str,
        arch_info: ArchInfo,
        *,
        from_config: bool = False,
    ) -> type[AnyModel]:
        cache_key = f"config:{repr(arch_info)}" if from_config else arch_name
        if cache_key not in AnyModel._wrapper_cache:
            AnyModel._wrapper_cache[cache_key] = _make_wrapper_cls(arch_name, arch_info)
        return AnyModel._wrapper_cache[cache_key]

    @classmethod
    def resolve_wrapper_cls(cls, model_config) -> type[AnyModel]:
        """Return the concrete wrapper class for model_config."""
        config = model_config.hf_config
        arch_name, arch_info, from_config = cls._resolve_arch(config)
        return cls._get_or_create_wrapper(arch_name, arch_info, from_config=from_config)

    def __new__(cls, *, vllm_config: VllmConfig, prefix: str = ""):
        if cls is not AnyModel:
            return super().__new__(cls)

        config = vllm_config.model_config.hf_config
        arch_name, arch_info, from_config = AnyModel._resolve_arch(config)
        wrapper_cls = AnyModel._get_or_create_wrapper(
            arch_name, arch_info, from_config=from_config
        )
        return object.__new__(wrapper_cls)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        arch_info = type(self)._anymodel_arch_info
        base_init_prefix = (
            arch_info.init_prefix if arch_info.init_prefix is not None else prefix
        )
        super().__init__(vllm_config=vllm_config, prefix=base_init_prefix)
        _patch_anymodel_layers(self, vllm_config, arch_info, base_init_prefix)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Filter out no-op module weights, then delegate to base class."""
        arch_info = type(self)._anymodel_arch_info
        if arch_info is not None:
            config = self.config
            layer_base_config = _resolve_layer_base_config(config, arch_info)
            per_layer_config = getattr(layer_base_config, "per_layer_config", None)
            if per_layer_config:
                noop_prefixes = _collect_noop_prefixes(
                    per_layer_config, arch_info, self
                )
                if noop_prefixes:
                    noop_prefixes = _expand_noop_prefixes_for_mapper(
                        noop_prefixes, type(self)
                    )
                    weights = (
                        (name, tensor)
                        for name, tensor in weights
                        if not any(name.startswith(p) for p in noop_prefixes)
                    )
        return super().load_weights(weights)

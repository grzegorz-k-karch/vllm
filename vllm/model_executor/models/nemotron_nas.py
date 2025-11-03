# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only deci model compatible with HuggingFace weights."""
from collections.abc import Iterable
from itertools import islice
from typing import Any, Optional, Union

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.attention import AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.models.llama import LlamaAttention, LlamaMLP
from vllm.sequence import IntermediateTensors
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2

from transformers import MambaConfig as TransformersMambaConfig

from .interfaces import HasNoOps, SupportsLoRA, SupportsPP
from .utils import (AutoWeightsLoader, PPMissingLayer, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)


def _ffn_mult_to_intermediate_size(ffn_mult: float, n_embd: int) -> int:
    # DeciLM-specific code
    intermediate_size = int(2 * ffn_mult * n_embd / 3)
    return _find_multiple(intermediate_size, 256)


def _find_multiple(n: int, k: int) -> int:
    # DeciLM-specific code
    if n % k == 0:
        return n
    return n + k - (n % k)


class DeciLMAttention(LlamaAttention):

    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__(config, hidden_size, num_heads, num_kv_heads,
                         rope_theta, rope_scaling, max_position_embeddings,
                         quant_config, bias, bias_o_proj, cache_config, prefix,
                         attn_type)

    def _init_rotary_emb(self, config, rope_scaling: Optional[dict[str, Any]],
                         quant_config: Optional[QuantizationConfig]) -> None:
        # Enables YARN for Mistral and LLaMA4 derivatives.
        is_neox_style = True
        if hasattr(config, "position_embedding_type"):
            is_neox_style = config.position_embedding_type not in [
                "mistral_yarn", "rope_llama4"
            ]

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
            partial_rotary_factor=self.partial_rotary_factor)


class DeciLMSSMDecoderLayer(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        prefix: str = "",
        block_config: Optional[Any] = None,
    ):
        super().__init__()
        mamba_config = block_config.attention.mamba
        self.mamba = MambaMixer2(
            hidden_size=config.hidden_size,
            ssm_state_size=mamba_config.state_dim,
            conv_kernel_size=4,  # hardcoded from megatron_lm__mamba_mixer.py
            intermediate_size=config.hidden_size,
            use_conv_bias=True, #config.mamba_conv_bias,
            use_bias=True, #config.mamba_proj_bias,
            n_groups=mamba_config.num_groups,
            num_heads=mamba_config.num_heads,
            head_dim=mamba_config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            activation=block_config.ffn.hidden_act,
            prefix=prefix,
        )
        # n_groups is overridden later by `MambaMixer2`
        self.groups_time_state_size = self.mamba.n_groups * block_config.attention.mamba.state_dim
        # hardcoded since we don't know about the availability of the multipliers
        # in the config yet but will be overridden later
        self.zxbcdt_multipliers = getattr(config, "ssm_multipliers", [1.0, 1.0, 1.0, 1.0, 1.0])
        self._init_mup_vector()

    def _init_mup_vector(self):
        """
        Non learnable per-block scaling vector composed of element-wise
        multipliers applied to each separate contiguous block of the output
        of the linear projection (in_proj) before further processing
        (gating, convolution, SSM):

            - Z block:  [0 : d_ssm]                      → zxbcdt_multipliers[0]
            - X block:  [d_ssm : 2 * d_ssm]              → zxbcdt_multipliers[1]
            - B block:  [2 * d_ssm : 2 * d_ssm + G * S]  → zxbcdt_multipliers[2]
            - C block:  [2 * d_ssm + G * S : 2 * d_ssm + 2 * G * S]
                        → zxbcdt_multipliers[3]
            - dt block: [2 * d_ssm + 2 * G * S : end]    → zxbcdt_multipliers[4]

        where:
            - d_ssm:     Dimension of state-space model latent
            - G:         Number of groups (n_groups)
            - S:         SSM state size per group
            - All indices are divided by tp_size to support tensor parallelism
        """
        vector_shape = (2 * self.d_ssm + 2 * self.groups_time_state_size +
                        self.config.mamba_n_heads) // self.tp_size
        mup_vector = torch.ones(1, vector_shape)
        # Z vector 0 -> d_ssm
        mup_vector[:, :self.d_ssm //
                   self.tp_size] *= self.zxbcdt_multipliers[0]
        # X vector d_ssm -> 2 * d_ssm
        mup_vector[:,
                   (self.d_ssm //
                    self.tp_size):(2 * self.d_ssm //
                                   self.tp_size)] *= self.zxbcdt_multipliers[1]
        # B vector 2 * d_ssm -> 2 * d_ssm + (n_group * d_state)
        mup_vector[
            :,
            (2 * self.d_ssm) //
            self.tp_size:(2 * self.d_ssm + self.groups_time_state_size) //
            self.tp_size,
        ] *= self.zxbcdt_multipliers[2]
        # C vector 2 * d_ssm + (n_group * d_state)
        # -> 2 * d_ssm + 2 * (n_group * d_state)
        mup_vector[
            :,
            (2 * self.d_ssm + self.groups_time_state_size) //
            self.tp_size:(2 * self.d_ssm + 2 * self.groups_time_state_size) //
            self.tp_size,
        ] *= self.zxbcdt_multipliers[3]
        # dt vector 2 * d_ssm + 2 * (n_group * d_state)
        # -> 2 * d_ssm + 2 * (n_group * d_state) + n_heads
        mup_vector[
            :,
            (2 * self.d_ssm + 2 * self.groups_time_state_size) //
            self.tp_size:,
        ] *= self.zxbcdt_multipliers[4]

        self.register_buffer("mup_vector", mup_vector, persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ):
        output = torch.empty_like(hidden_states)
        self.mamba(
            hidden_states,
            output,
            mup_vector=self.mup_vector,
        )
        return output, residual


class DeciLMDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        block_config: Optional[Any] = None,
    ) -> None:
        super().__init__()
        if block_config is None:
            block_config = config.block_configs[layer_idx]
        self._is_no_op_attention = block_config.attention.no_op
        self._is_no_op_ffn = True # TODO: uncomment this when we have a way to handle the FFN in the SSM branch

        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        bias_o_proj = attention_bias
        # support internlm/internlm3-8b with qkv_bias
        if hasattr(config, "qkv_bias"):
            attention_bias = config.qkv_bias

        hidden_activation = getattr(config, "hidden_act", block_config.ffn.hidden_act)

        if not self._is_no_op_attention:
            num_kv_heads = (config.num_attention_heads //
                            block_config.attention.n_heads_in_group)
            self.self_attn = DeciLMAttention(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=num_kv_heads,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                bias=attention_bias,
                bias_o_proj=bias_o_proj,
                cache_config=cache_config,
                prefix=f"{prefix}.self_attn",
            )
            self.input_layernorm = RMSNorm(config.hidden_size,
                                           eps=config.rms_norm_eps)

        if not self._is_no_op_ffn:
            if hasattr(block_config.ffn, "ffn_mult"):
                ffn_mult = block_config.ffn.ffn_mult
                intermediate_size = _ffn_mult_to_intermediate_size(
                    ffn_mult, config.hidden_size
                )
            else:
                intermediate_size = block_config.ffn.intermediate_size

            self.mlp = LlamaMLP(
                hidden_size=self.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=hidden_activation,
                quant_config=quant_config,
                bias=getattr(config, "mlp_bias", False),
                prefix=f"{prefix}.mlp",
            )
            self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                    eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention

        if self._is_no_op_attention:
            pass
        else:
            # TODO: uncomment this when we have a way to handle the input layernorm in the SSM branch
            # if (residual is None):
            #     residual = hidden_states
            #     hidden_states = self.input_layernorm(hidden_states)
            # else:
            #     hidden_states, residual = self.input_layernorm(
            #         hidden_states, residual)
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
            )

        # Fully Connected
        if not self._is_no_op_ffn:
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual)
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class DeciLMParallelHybrid(nn.Module):
    """
    A hybrid decoder layer for FalconH1 where the input is processed
    in parallel through both the self-attention branch and the SSM (Mamba)
    branch. Their outputs are then summed to produce the final hidden state.

    This layer uses:
      - DeciLMAttention for the multi-head self-attention branch.
      - DeciLMMambaMixer for the state-space (Mamba) branch.

    The layer is based on FalconH1ParallelHybrid in falcon_h1.py
    """

    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # block_config has the attentionm, ffn config and parallel_blocks 
        # (i.e, parallel attention and mamba) config
        block_config = config.block_configs[layer_idx]
        self._is_no_op_ffn = block_config.ffn.no_op
        self._has_parallel_blocks = block_config.parallel_blocks is not None
        mamba_block_config = block_config.parallel_blocks[0]
        attention_block_config = block_config.parallel_blocks[1]
        # input layernorm is shared between the attention and SSM branches
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)

        if self._has_parallel_blocks:
            # Temporarily set the FFN to no_op to avoid double processing
            # config.block_configs[layer_idx].ffn.no_op = True # TODO: uncomment 
            # this when we have a way to handle the FFN in the SSM branch

            # Instantiate the attention branch
            self.self_attn = DeciLMDecoderLayer(
                config=config,
                layer_idx=layer_idx,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.attention",
                block_config=attention_block_config,
            )

            # Restore the FFN configuration
            # config.block_configs[layer_idx].ffn.no_op = self._is_no_op_ffn 
            # TODO: uncomment this when we have a way to handle the FFN in the SSM branch

            # In V1 all attention/ssm layers must have
            # different index in prefix
            ssm_layer_idx = config.num_hidden_layers + layer_idx
            ssm_prefix = prefix.split(".")[0] + f".{ssm_layer_idx}"

            # Instantiate the SSM branch
            self.mamba = DeciLMSSMDecoderLayer(
                config=config,
                prefix=ssm_prefix,
                block_config=mamba_block_config,
            )
            # multipliers are hardcoded since we don't know about the availability
            # of the multipliers in the config yet but will be overridden later
            self.ssm_out_multiplier = getattr(config, "ssm_out_multiplier", 1.0)
            self.ssm_in_multiplier = getattr(config, "ssm_in_multiplier", 1.0)

            self.attention_in_multiplier = getattr(config, "attention_in_multiplier", 1.0)
            self.attn_out_multiplier = getattr(config, "attention_out_multiplier", 1.0)

        if not self._is_no_op_ffn:
            if hasattr(block_config.ffn, "ffn_mult"):
                ffn_mult = block_config.ffn.ffn_mult
                intermediate_size = _ffn_mult_to_intermediate_size(
                    ffn_mult, config.hidden_size
                )
            else:
                intermediate_size = block_config.ffn.intermediate_size

            hidden_activation = getattr(config, "hidden_act", block_config.ffn.hidden_act)

            self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                    eps=config.rms_norm_eps)
            self.mlp = LlamaMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=hidden_activation,
                quant_config=quant_config,
                bias=getattr(config, "mlp_bias", False),
                prefix=f"{prefix}.mlp",
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ):
        hidden_states = self.input_layernorm(hidden_states)
        if self._has_parallel_blocks:
            # Process input through the attention branch.
            # FalconH1AttentionDecoderLayer expects positions, hidden_states,
            # kv_cache, attn_metadata, and residual.
            attn_hidden, _ = self.self_attn(
                positions=positions,
                hidden_states=hidden_states * self.attention_in_multiplier,
                residual=residual,
            )

            # Process input through the SSM branch.
            # FalconH1SSMDecoderLayer expects hidden_states, attn_metadata,
            # residual, and sequence_idx.
            ssm_hidden, _ = self.mamba(
                hidden_states=hidden_states * self.ssm_in_multiplier,
                residual=residual,
            )
            # Sum the outputs from both branches.
            # We assume both branches produce outputs of the same
            # dimensionality (config.hidden_size).
            hidden_states = (attn_hidden * self.attn_out_multiplier) + (
                ssm_hidden * self.ssm_out_multiplier)
            hidden_states = hidden_states + residual

        # feed-forward
        if not self._is_no_op_ffn:
            residual = hidden_states            
            hidden_states, _ = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
        
        return hidden_states, residual


@support_torch_compile
class DeciModel(nn.Module):

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[DeciLMDecoderLayer] = DeciLMDecoderLayer,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.quant_config = quant_config
        self.padding_idx = config.pad_token_id
        lora_vocab = ((lora_config.lora_extra_vocab_size *
                       (lora_config.max_loras or 1)) if lora_config else 0)
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        def get_layer(prefix: str):
            layer_idx = int(prefix.rsplit(".", 1)[1])
            return layer_type(
                config,
                layer_idx,
                cache_config,
                quant_config=quant_config,
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        kv_cache_index = 0
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            if not layer._is_no_op_attention:
                hidden_states, residual = layer(positions, hidden_states,
                                                residual)
                kv_cache_index += 1
            else:
                hidden_states, residual = layer(positions, hidden_states,
                                                residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if self.quant_config is not None and (
                    scale_name := self.quant_config.get_cache_scale(name)):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = (loaded_weight if loaded_weight.dim() == 0 else
                                 loaded_weight[0])
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            if "scale" in name:
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class DeciLMForCausalLM(nn.Module, SupportsLoRA, SupportsPP, HasNoOps):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # LoRA specific attributes
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]

    # Mistral/Llama models can also be loaded with --load-format mistral
    # from consolidated.safetensors checkpoints
    mistral_mapping = {
        "layers": "model.layers",
        "attention": "self_attn",
        "wq": "q_proj",
        "wk": "k_proj",
        "wv": "v_proj",
        "wo": "o_proj",
        "attention_norm": "input_layernorm",
        "feed_forward": "mlp",
        "w1": "gate_proj",
        "w2": "down_proj",
        "w3": "up_proj",
        "ffn_norm": "post_attention_layernorm",
        "tok_embeddings": "model.embed_tokens",
        "output": "lm_head",
        "norm": "model.norm",
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.lora_config = lora_config

        self.model = self._init_model(vllm_config=vllm_config,
                                      prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = config.vocab_size
            if lora_config:
                self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=(
                    DEFAULT_VOCAB_PADDING_SIZE
                    # We need bigger padding if using lora for kernel
                    # compatibility
                    if not lora_config else
                    lora_config.lora_vocab_padding_size),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(
                    self.model.embed_tokens)

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                    config.vocab_size,
                                                    logit_scale)
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def _init_model(self, vllm_config: VllmConfig, prefix: str = ""):
        falcon_h1 = getattr(self.config, "falcon_h1", False)
        layer_type = DeciLMParallelHybrid if falcon_h1 else DeciLMDecoderLayer
        return DeciModel(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        model_output = self.model(input_ids, positions, intermediate_tensors,
                                  inputs_embeds)
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)

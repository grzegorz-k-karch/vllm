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
from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.models.llama import LlamaAttention, LlamaMLP
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator, MambaStateShapeCalculator)
from vllm.sequence import IntermediateTensors
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2

from transformers import MambaConfig as TransformersMambaConfig

from .interfaces import HasInnerState, HasNoOps, SupportsLoRA, SupportsPP, IsHybrid
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


class DeciLMMambaMixer(nn.Module):
    def __init__(self, 
                 config: LlamaConfig, 
                 mamba_config: Any, 
                 quant_config: Optional[QuantizationConfig], 
                 d_ssm: int, 
                 prefix: str):
        super().__init__()

        class ModelConfigTmp:
            def __init__(self, config: LlamaConfig):
                self.dtype = "bfloat16" #config.block_configs[0].ffn.weights_precision
                self.mamba_cache_dtype = "bfloat16"
                self.mamba_ssm_cache_dtype = "bfloat16"

        model_config = ModelConfigTmp(config)

        self.mamba_mixer = MambaMixer2(
            hidden_size=config.hidden_size,
            ssm_state_size=mamba_config.state_dim,
            conv_kernel_size=4,  # hardcoded from megatron_lm__mamba_mixer.py
            intermediate_size=d_ssm,
            use_conv_bias=False, #config.mamba_conv_bias,
            use_bias=False, #config.mamba_proj_bias,
            n_groups=mamba_config.num_groups,
            num_heads=mamba_config.num_heads,
            head_dim=mamba_config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            activation="silu", # mamba_config.ffn.hidden_act, # TODO: remove hardcoded activation
            model_config=model_config,
            cache_config=model_config,
            quant_config=quant_config,
            use_rms_norm=True,
            prefix=f"{prefix}.self_attn.mamba_mixer",
        )

    def forward(self, hidden_states: torch.Tensor, output: torch.Tensor, mup_vector: torch.Tensor):
        self.mamba_mixer(hidden_states, output) #, mup_vector) # TODO: uncomment


class DeciLMSSMDecoderLayer(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        prefix: str = "",
        block_config: Optional[Any] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        mamba_config = block_config.attention.mamba

        self.d_ssm = 4096 #config.hidden_size # TODO: remove hardcoded d_ssm
        self.n_heads = mamba_config.num_heads
        self.tp_size = get_tensor_model_parallel_world_size()
        self.self_attn = DeciLMMambaMixer(config, mamba_config, quant_config, 
                                          self.d_ssm, prefix)

        # n_groups is overridden later by `MambaMixer2`
        self.groups_time_state_size = mamba_config.num_groups * self.d_ssm
        # hardcoded since we don't know about the availability of the multipliers
        # in the config yet but will be overridden later
        self.zxbcdt_multipliers = getattr(config, "ssm_multipliers", [1.0, 1.0, 1.0, 1.0, 1.0])
        self._init_mup_vector()
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)


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
                        self.n_heads) // self.tp_size
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
        hidden_states = self.input_layernorm(hidden_states)
        output = torch.empty_like(hidden_states)
        self.self_attn(
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
        # TODO: comment this when we have a way to handle the FFN in the SSM branch
        self._is_no_op_ffn = True
        # TODO: uncomment this when we have a way to handle the FFN in the SSM branch
        # self._is_no_op_ffn = block_config.ffn.no_op

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
            print(f"|||| nemotron_nas.py: {num_kv_heads=}, {config.num_attention_heads=}, {block_config.attention.n_heads_in_group=}")
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
            # self.input_layernorm = RMSNorm(config.hidden_size,
            #                                eps=config.rms_norm_eps)

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
        # # input layernorm is shared between the attention and SSM branches
        # self.input_layernorm = RMSNorm(config.hidden_size,
        #                                eps=config.rms_norm_eps)

        self._has_parallel_blocks = block_config.parallel_blocks is not None

        if self._has_parallel_blocks:
            mamba_block_config = block_config.parallel_blocks[0]
            attention_block_config = block_config.parallel_blocks[1]
            # Temporarily set the FFN to no_op to avoid double processing
            # config.block_configs[layer_idx].ffn.no_op = True # TODO: uncomment 
            # this when we have a way to handle the FFN in the SSM branch

            # Instantiate the attention branch
            self.parallel_blocks_1 = DeciLMDecoderLayer(
                config=config,
                layer_idx=layer_idx,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.parallel_blocks_1",
                block_config=attention_block_config,
            )

            # Restore the FFN configuration
            # config.block_configs[layer_idx].ffn.no_op = self._is_no_op_ffn 
            # TODO: uncomment this when we have a way to handle the FFN in the SSM branch

            # Instantiate the SSM branch
            self.parallel_blocks_0 = DeciLMSSMDecoderLayer(
                config=config,
                prefix=f"{prefix}.parallel_blocks_0",
                block_config=mamba_block_config,
                quant_config=quant_config,
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
    ):
        residual = hidden_states
        # return hidden_states, hidden_states
        # hidden_states = self.input_layernorm(hidden_states)
        if self._has_parallel_blocks:
            # Process input through the attention branch.
            # FalconH1AttentionDecoderLayer expects positions, hidden_states,
            # kv_cache, attn_metadata, and residual.
            attn_hidden, _ = self.parallel_blocks_1(
                positions=positions,
                hidden_states=hidden_states * self.attention_in_multiplier,
                residual=residual,
            )

            # Process input through the SSM branch.
            # FalconH1SSMDecoderLayer expects hidden_states, attn_metadata,
            # residual, and sequence_idx.
            ssm_hidden, _ = self.parallel_blocks_0(
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
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
        
        return hidden_states


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
            # residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            # residual = intermediate_tensors["residual"]

        # kv_cache_index = 0
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states = layer(positions, hidden_states)
            
            # if getattr(layer, "parallel_blocks_1", None) is not None and not layer.parallel_blocks_1._is_no_op_attention:
            #  hidden_states, residual = layer(positions, hidden_states, residual)
            #     kv_cache_index += 1
            # else:
            #     hidden_states, residual = layer(positions, hidden_states,
            #                                     residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
            })

        hidden_states = self.norm(hidden_states)
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
        name_mapping = {
            "mamba_mixer.A_log": "mamba_mixer.A",
        }
        def maybe_remap_name(name: str) -> str:
            for k, v in name_mapping.items():
                if k in name:
                    return name.replace(k, v)
            return name

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

                name = maybe_remap_name(name)
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class DeciLMForCausalLM(nn.Module, HasInnerState, SupportsLoRA, SupportsPP, HasNoOps, IsHybrid):
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
        "A": "A_log",
    }

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        # Similar to falcon_h1.py lines 494-504
        return MambaStateDtypeCalculator.mamba2_state_dtype(
            "bfloat16", #vllm_config.model_config.dtype, 
            "bfloat16", #vllm_config.cache_config.mamba_cache_dtype, 
            "bfloat16", #vllm_config.cache_config.mamba_ssm_cache_dtype
            )
    
    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        # Similar to falcon_h1.py lines 506-537
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config

        class HFConfigTmp:
            def __init__(self, config: LlamaConfig):
                pass
        mamba_config = hf_config.block_configs[2].parallel_blocks[0].attention.mamba
        intermediate_size = hf_config.block_configs[3].ffn.intermediate_size

        return MambaStateShapeCalculator.mamba2_state_shape(
            intermediate_size=intermediate_size,
            tp_world_size=parallel_config.tensor_parallel_size,
            n_groups=mamba_config.num_groups,
            num_heads=mamba_config.num_heads,
            head_dim=mamba_config.head_dim,
            state_size=mamba_config.state_dim,
            conv_kernel=4, #hf_config.mamba_d_conv,
        )
        
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        print("|||| cache config: ", vllm_config.cache_config)

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

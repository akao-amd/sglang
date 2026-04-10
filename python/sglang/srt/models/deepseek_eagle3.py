"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Eagle3 speculative decoding model for DeepseekV2/V3 with MLP (no MoE)."""

import copy
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.layers.communicator import AttentionInputs
from sglang.srt.layers.dp_attention import get_attn_tp_context
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_v2 import (
    DeepseekV2AttentionMLA,
    DeepseekV2ForCausalLM,
    DeepseekV2MLP,
)
from sglang.srt.utils import BumpAllocator, add_prefix


class DeepseekV2Eagle3DecoderLayer(nn.Module):
    """
    Eagle3 decoder layer for Deepseek that:
    1. Always uses MLP (not MoE)
    2. First layer (layer_idx=0) accepts concatenated embeds + hidden_states (2x hidden_size input)
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        if hasattr(config, "rope_parameters"):
            rope_theta = config.rope_parameters["rope_theta"]
            rope_scaling = config.rope_parameters
            rope_type = rope_scaling.get("rope_type")
            if rope_type == "default":
                rope_scaling = None
        else:
            rope_theta = config.rope_theta
            rope_scaling = config.rope_scaling

        max_position_embeddings = config.max_position_embeddings
        q_lora_rank = config.q_lora_rank if hasattr(config, "q_lora_rank") else None

        self.self_attn = DeepseekV2AttentionMLA(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            layer_id=layer_idx,
            reduce_results=True,
            prefix=add_prefix("self_attn", prefix),
        )

        # For layer 0, override projection inputs to accept 2x hidden_size
        # (concatenated embeds + hidden_states)
        if layer_idx == 0:
            proj_input_size = 2 * self.hidden_size
            if q_lora_rank is not None:
                self.self_attn.fused_qkv_a_proj_with_mqa = ReplicatedLinear(
                    proj_input_size,
                    q_lora_rank + config.kv_lora_rank + config.qk_rope_head_dim,
                    bias=False,
                    quant_config=quant_config,
                    prefix=add_prefix("self_attn.fused_qkv_a_proj_with_mqa", prefix),
                )
            else:
                self.self_attn.q_proj = ColumnParallelLinear(
                    proj_input_size,
                    config.num_attention_heads
                    * (config.qk_nope_head_dim + config.qk_rope_head_dim),
                    bias=False,
                    quant_config=quant_config,
                    prefix=add_prefix("self_attn.q_proj", prefix),
                )
                self.self_attn.kv_a_proj_with_mqa = ReplicatedLinear(
                    proj_input_size,
                    config.kv_lora_rank + config.qk_rope_head_dim,
                    bias=False,
                    quant_config=quant_config,
                    prefix=add_prefix("self_attn.kv_a_proj_with_mqa", prefix),
                )

        # Always use MLP (not MoE) for Eagle3
        self.mlp = DeepseekV2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.layer_idx == 0:
            # First layer: normalize then concatenate embeds with hidden_states
            embeds = self.input_layernorm(embeds)
            hidden_states = self.hidden_norm(hidden_states)
            residual = hidden_states
            hidden_states = torch.cat([embeds, hidden_states], dim=-1)
        else:
            # Subsequent layers: standard pre-norm with residual
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # When using fused qkv projection (q_lora_rank is not None), the attention
        # uses get_attn_tp_context().fetch_qkv_latent() internally. We must set the
        # attn_inputs here since we bypass the normal LayerCommunicator.prepare_attn().
        if hasattr(self.self_attn, "fused_qkv_a_proj_with_mqa"):
            attn_inputs = AttentionInputs(
                hidden_states, forward_batch, self.self_attn.prepare_qkv_latent
            )
            get_attn_tp_context().set_attn_inputs(attn_inputs)

        # Self Attention
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # Fully Connected (MLP, not MoE)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class DeepseekV2Eagle3Model(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )

        self.layers = nn.ModuleList(
            [
                DeepseekV2Eagle3DecoderLayer(
                    config=config,
                    layer_idx=layer_idx,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_idx}", prefix),
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # fc layer to combine 3 auxiliary hidden states (3x hidden_size -> hidden_size)
        if hasattr(config, "target_hidden_size"):
            fc_input_size = config.target_hidden_size * 3
        else:
            fc_input_size = config.hidden_size * 3

        self.fc = torch.nn.Linear(
            fc_input_size,
            config.hidden_size,
            bias=False,
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if input_embeds is None:
            input_embeds = self.embed_tokens(input_ids)

        hidden_states = forward_batch.spec_info.hidden_states
        if hidden_states.shape[-1] != input_embeds.shape[-1]:
            hidden_states = self.fc(hidden_states)

        # idle batch
        if hidden_states.shape[0] == 0:
            return hidden_states, [hidden_states]

        device = input_embeds.device
        zero_allocator = BumpAllocator(
            buffer_size=2 * len(self.layers),
            dtype=torch.float32,
            device=device,
        )

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                embeds=input_embeds,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                residual=residual,
                zero_allocator=zero_allocator,
            )

        hidden_states_to_logits, hidden_states_to_aux = self.norm(
            hidden_states, residual
        )

        # For draft decode, we capture the hidden state before norm
        return hidden_states_to_logits, [hidden_states_to_aux]


class DeepseekV2ForCausalLMEagle3(DeepseekV2ForCausalLM):
    """Eagle3 speculative decoding causal LM for DeepseekV2/V3."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config

        if config.num_hidden_layers != 1:
            raise ValueError("Eagle3 currently only supports 1 layer")

        self.model = DeepseekV2Eagle3Model(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        if getattr(config, "draft_vocab_size", None) is None:
            self.load_lm_head_from_target = True
            config.draft_vocab_size = config.vocab_size
        else:
            self.load_lm_head_from_target = False

        self.lm_head = ParallelLMHead(
            config.draft_vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )

        config_ = copy.deepcopy(config)
        config_.vocab_size = config_.draft_vocab_size
        self.logits_processor = LogitsProcessor(config_)

        self.capture_aux_hidden_states = True
        self.hot_token_id = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ):
        hidden_states, aux_hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds
        )
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters())

        # Stacked parameter mappings
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        for name, loaded_weight in weights:
            # Remap midlayer -> layers.0
            if "midlayer." in name:
                name = name.replace("midlayer.", "layers.0.")

            # Handle d2t (draft-to-target token ID mapping)
            if "d2t" in name:
                self.hot_token_id = loaded_weight + torch.arange(loaded_weight.shape[0])
                continue

            if "t2d" in name:
                continue

            # Prefix with "model." if not already there
            param_name = name if name in params_dict else f"model.{name}"

            for p_name, w_name, shard_id in stacked_params_mapping:
                if w_name not in param_name:
                    continue
                mapped_name = param_name.replace(w_name, p_name)
                if mapped_name in params_dict:
                    param = params_dict[mapped_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def set_embed(self, embed):
        # NOTE: If draft hidden size != target hidden size, the embed weight cannot be shared
        if (
            hasattr(self.config, "target_hidden_size")
            and self.config.target_hidden_size != self.config.hidden_size
        ):
            return
        del self.model.embed_tokens.weight
        self.model.embed_tokens.weight = embed
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def get_hot_token_id(self):
        return self.hot_token_id


# Alias for DeepseekV3
DeepseekV3ForCausalLMEagle3 = DeepseekV2ForCausalLMEagle3

EntryClass = [DeepseekV2ForCausalLMEagle3, DeepseekV3ForCausalLMEagle3]

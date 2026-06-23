# Copyright 2026 starVLA community. All rights reserved.
"""Scale-wise coarse-to-fine Stage 1 action-code policy."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenVARParallel import QwenVARParallel
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("QwenVARScaleParallel")
class QwenVARScaleParallel(QwenVARParallel):
    """Predict action codes coarse-to-fine, with parallel slots per scale.

    This keeps Stage 2 off the text autoregressive path while avoiding the
    hardest part of full one-shot prediction. The model predicts each Stage 1
    scale in order. Slots within the same scale are predicted in parallel, and
    later scales cross-attend to teacher-forced earlier scale code embeddings
    during training or predicted code embeddings during inference.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        groups = int(self.num_factor_slots)
        head_cfg = self.config.framework.get("parallel_head", {})
        code_condition_dropout = float(head_cfg.get("code_condition_dropout", head_cfg.get("dropout", 0.0)))

        if self.stage1_tokenizer.quantization_mode == "product_vq":
            self.code_condition_projectors = nn.ModuleList(
                [nn.Linear(codebook.embedding_dim, hidden_size) for codebook in self.stage1_tokenizer.product_codebooks]
            )
        else:
            self.code_condition_projectors = nn.ModuleList(
                [nn.Linear(self.stage1_tokenizer.shared_codebook.embedding_dim, hidden_size)]
            )
        if len(self.code_condition_projectors) != groups:
            raise ValueError(
                f"Expected {groups} code-condition projectors, got {len(self.code_condition_projectors)}."
            )
        self.code_condition_norm = nn.LayerNorm(hidden_size)
        self.code_condition_dropout = nn.Dropout(code_condition_dropout)

    def _slot_mask(self, scale_idx: int, device: torch.device) -> torch.Tensor:
        return self.slot_scale_indices.to(device) == int(scale_idx)

    def _code_condition_embeddings(self, tokens: torch.Tensor, slot_indices: torch.Tensor) -> torch.Tensor:
        """Embed flat Stage 1 code slots as conditioning memory tokens."""
        factor_indices = self.slot_factor_indices.to(tokens.device).index_select(0, slot_indices)
        pieces = tokens.new_zeros(
            tokens.shape[0],
            slot_indices.numel(),
            int(self.qwen_vl_interface.model.config.hidden_size),
            dtype=torch.float32,
        )

        if self.stage1_tokenizer.quantization_mode == "product_vq":
            codebooks = list(self.stage1_tokenizer.product_codebooks)
        else:
            codebooks = [self.stage1_tokenizer.shared_codebook]

        for factor_idx, (codebook, projector) in enumerate(zip(codebooks, self.code_condition_projectors, strict=True)):
            mask = factor_indices == factor_idx
            if mask.any():
                code_vectors = codebook(tokens[:, mask].long()).float()
                pieces[:, mask, :] = projector(code_vectors).float()

        query_pos = self.action_token_queries.index_select(0, slot_indices).float()
        pieces = pieces + query_pos.unsqueeze(0)
        return self.code_condition_dropout(self.code_condition_norm(pieces))

    def _predict_scale_logits(
        self,
        context_states: torch.Tensor,
        pooled_context: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        code_memory: torch.Tensor | None,
        slot_indices: torch.Tensor,
    ) -> torch.Tensor:
        query_states = self.action_token_queries.index_select(0, slot_indices)[None, :, :]
        query_states = query_states.expand(context_states.shape[0], -1, -1).float()
        query_states = query_states + pooled_context[:, None, :]

        if code_memory is None:
            attention_context = context_states
            attention_mask = key_padding_mask
        else:
            attention_context = torch.cat([context_states, code_memory.to(context_states.dtype)], dim=1)
            if key_padding_mask is None:
                attention_mask = None
            else:
                extra_mask = torch.zeros(
                    code_memory.shape[:2],
                    device=key_padding_mask.device,
                    dtype=key_padding_mask.dtype,
                )
                attention_mask = torch.cat([key_padding_mask, extra_mask], dim=1)

        for block in self.action_query_cross_attn:
            query_states = block(query_states, attention_context, key_padding_mask=attention_mask)
        return self._classify_queries(query_states, slot_indices=slot_indices)

    def _encode_context(self, examples: List[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        qwen_inputs = self._build_qwenvl_inputs(images=batch_images, instructions=instructions)

        use_cuda_autocast = self.qwen_vl_interface.model.device.type == "cuda"
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda_autocast):
            outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        context_states = outputs.hidden_states[-1].float()
        pooled_context = self._pool_condition(context_states, qwen_inputs.get("attention_mask", None))
        key_padding_mask = self._key_padding_mask(qwen_inputs.get("attention_mask", None))
        return context_states, pooled_context, key_padding_mask

    def _predict_token_logits_teacher_forced(self, examples: List[dict], target_tokens: torch.Tensor) -> torch.Tensor:
        context_states, pooled_context, key_padding_mask = self._encode_context(examples)
        logits = context_states.new_empty(
            target_tokens.shape[0],
            self.token_dim,
            int(self.stage1_tokenizer.codebook_size),
        )
        code_memory = None
        for scale_idx, _scale in enumerate(self.stage1_tokenizer.scales):
            mask = self._slot_mask(scale_idx, target_tokens.device)
            slot_indices = torch.nonzero(mask, as_tuple=False).flatten()
            scale_logits = self._predict_scale_logits(
                context_states,
                pooled_context,
                key_padding_mask,
                code_memory,
                slot_indices,
            )
            logits[:, mask, :] = scale_logits.to(dtype=logits.dtype)
            scale_memory = self._code_condition_embeddings(target_tokens[:, mask], slot_indices)
            code_memory = scale_memory if code_memory is None else torch.cat([code_memory, scale_memory], dim=1)
        return logits

    def forward(self, examples: List[dict] = None, **kwargs) -> dict[str, torch.Tensor]:
        target_tokens = torch.stack([example["action_tokens"].long() for example in examples], dim=0)
        target_tokens = target_tokens.to(self.qwen_vl_interface.model.device)
        logits = self._predict_token_logits_teacher_forced(examples, target_tokens)
        label_smoothing = float(self.config.framework.get("parallel_head", {}).get("label_smoothing", 0.0))
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_tokens.reshape(-1),
            label_smoothing=label_smoothing,
        )
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            correct = predictions.eq(target_tokens)
            token_accuracy = correct.float().mean()
            token_losses = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                target_tokens.reshape(-1),
                reduction="none",
            ).view_as(target_tokens)

        metrics: dict[str, torch.Tensor] = {"action_loss": loss, "token_accuracy": token_accuracy}
        scale_indices = self.slot_scale_indices.to(logits.device)
        factor_indices = self.slot_factor_indices.to(logits.device)
        with torch.no_grad():
            for scale_idx, scale in enumerate(self.stage1_tokenizer.scales):
                mask = scale_indices == scale_idx
                if mask.any():
                    metrics[f"loss/scale_{int(scale)}"] = token_losses[:, mask].mean()
                    metrics[f"acc/scale_{int(scale)}"] = correct[:, mask].float().mean()
            for factor_idx in range(self.num_factor_slots):
                mask = factor_indices == factor_idx
                if mask.any():
                    metrics[f"loss/codebook_group_{factor_idx}"] = token_losses[:, mask].mean()
                    metrics[f"acc/codebook_group_{factor_idx}"] = correct[:, mask].float().mean()
        return metrics

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs) -> dict[str, np.ndarray]:
        if not isinstance(examples, list):
            examples = [examples]
        converted = []
        for example in examples:
            item = dict(example)
            item["image"] = to_pil_preserve(example["image"])
            converted.append(item)

        context_states, pooled_context, key_padding_mask = self._encode_context(converted)
        token_ids = torch.empty(context_states.shape[0], self.token_dim, device=context_states.device, dtype=torch.long)
        code_memory = None
        for scale_idx, _scale in enumerate(self.stage1_tokenizer.scales):
            mask = self._slot_mask(scale_idx, context_states.device)
            slot_indices = torch.nonzero(mask, as_tuple=False).flatten()
            scale_logits = self._predict_scale_logits(
                context_states,
                pooled_context,
                key_padding_mask,
                code_memory,
                slot_indices,
            )
            scale_tokens = scale_logits.argmax(dim=-1).long()
            token_ids[:, mask] = scale_tokens
            scale_memory = self._code_condition_embeddings(scale_tokens, slot_indices)
            code_memory = scale_memory if code_memory is None else torch.cat([code_memory, scale_memory], dim=1)

        token_ids = token_ids.to(next(self.stage1_tokenizer.parameters()).device)
        normalized_actions = self.stage1_tokenizer.decode(token_ids)
        normalized_actions = self._postprocess_decoded_actions(normalized_actions)
        normalized_actions = normalized_actions.detach().cpu().float().numpy()
        return {
            "normalized_actions": normalized_actions,
            "action_tokens": token_ids.detach().cpu().numpy(),
            "generation_diagnostics": [
                {
                    "valid_token_count": self.token_dim,
                    "padded": False,
                    "parallel": True,
                    "scale_wise": True,
                    "classifier": self.parallel_classifier_type,
                }
                for _ in converted
            ],
        }

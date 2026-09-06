from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from starVLA.model.framework.VLM4A.QwenVARParallel import ActionCodeQueryBlock, QwenVARParallel


@pytest.mark.parametrize(
    ("module_dtype", "input_dtype"),
    [
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.bfloat16),
    ],
)
def test_action_code_query_block_accepts_mixed_parameter_and_activation_dtypes(
    module_dtype: torch.dtype,
    input_dtype: torch.dtype,
) -> None:
    block = ActionCodeQueryBlock(
        hidden_size=16,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
    ).to(dtype=module_dtype)
    block.eval()

    queries = torch.randn(2, 5, 16, dtype=input_dtype)
    context = torch.randn(2, 7, 16, dtype=input_dtype)
    output = block(queries, context)

    assert output.dtype == input_dtype
    assert torch.isfinite(output.float()).all()


@pytest.mark.parametrize(
    ("module_dtype", "input_dtype"),
    [
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.bfloat16),
    ],
)
def test_shared_query_classifier_uses_its_parameter_dtype(
    module_dtype: torch.dtype,
    input_dtype: torch.dtype,
) -> None:
    model = QwenVARParallel.__new__(QwenVARParallel)
    nn.Module.__init__(model)
    model.action_token_norm = nn.LayerNorm(16).to(dtype=module_dtype)
    model.action_token_dropout = nn.Identity()
    model.parallel_classifier_type = "shared"
    model.action_token_classifier = nn.Linear(16, 32).to(dtype=module_dtype)

    logits = model._classify_queries(torch.randn(2, 5, 16, dtype=input_dtype))

    assert logits.dtype == module_dtype
    assert logits.shape == (2, 5, 32)
    assert torch.isfinite(logits.float()).all()


def test_per_factor_query_classifier_uses_its_parameter_dtype() -> None:
    model = QwenVARParallel.__new__(QwenVARParallel)
    nn.Module.__init__(model)
    model.action_token_norm = nn.LayerNorm(16).to(dtype=torch.bfloat16)
    model.action_token_dropout = nn.Identity()
    model.parallel_classifier_type = "per_factor"
    model.stage1_tokenizer = SimpleNamespace(codebook_size=32)
    model.slot_factor_indices = torch.tensor([0, 1, 0, 1, 0])
    model.action_factor_classifiers = nn.ModuleList(
        [
            nn.Linear(16, 32),
            nn.Linear(16, 32),
        ]
    ).to(dtype=torch.bfloat16)

    logits = model._classify_queries(torch.randn(2, 5, 16, dtype=torch.float32))

    assert logits.dtype == torch.bfloat16
    assert logits.shape == (2, 5, 32)
    assert torch.isfinite(logits.float()).all()

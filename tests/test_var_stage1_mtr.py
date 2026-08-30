import pytest
import torch

from starVLA.model.modules.action_tokenizer.var_action_tokenizer import VARActionTokenizer
from starVLA.training.train_var_stage1 import build_temporal_scale_target, compute_mtr_loss


def _run_shape_and_grad_case(horizon: int, action_dim: int, scales: list[int]) -> None:
    dim_groups = {
        "position": list(range(min(3, action_dim))),
        "rotation": list(range(3, min(6, action_dim))),
        "gripper": [action_dim - 1],
    }
    model = VARActionTokenizer(
        action_dim=action_dim,
        seq_len=horizon,
        scales=scales,
        embed_dim=32,
        quantization_mode="product_vq",
        product_codebook_groups=16,
        dim_groups=dim_groups,
    )
    actions = torch.randn(2, horizon, action_dim, requires_grad=True)
    output = model(actions, return_scale_recons=True)

    assert output["flat_token_ids"].shape == (2, sum(scales) * 16)
    assert len(output["scale_recons"]) == len(scales)
    assert all(recon.shape == actions.shape for recon in output["scale_recons"])

    baseline = model(actions.detach())["recon"]
    assert torch.allclose(output["recon"].detach(), baseline, atol=1e-6, rtol=1e-5)

    mtr_loss, _ = compute_mtr_loss(
        actions,
        output["scale_recons"][:-1],
        scales[:-1],
        dim_groups=dim_groups,
        gripper_weight=0.0,
    )
    mtr_loss.backward()
    assert torch.isfinite(mtr_loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize(
    ("horizon", "action_dim", "scales"),
    [(8, 7, [1, 2, 4, 8]), (16, 29, [1, 2, 4, 8, 16])],
)
def test_mtr_libero_and_robocasa_shapes_and_gradients(horizon, action_dim, scales):
    _run_shape_and_grad_case(horizon, action_dim, scales)


def test_temporal_scale_target_shape_and_validation():
    actions = torch.randn(2, 8, 7)
    assert build_temporal_scale_target(actions, 1).shape == actions.shape
    with pytest.raises(ValueError):
        build_temporal_scale_target(actions, 0)
    with pytest.raises(ValueError):
        build_temporal_scale_target(actions, 9)


def test_mtr_scale_weights_and_gripper_mask():
    actions = torch.zeros(1, 8, 2)
    scale_recons = []
    for position_error in (1.0, 2.0, 3.0):
        recon = torch.zeros_like(actions)
        recon[..., 0] = position_error
        recon[..., 1] = 100.0
        scale_recons.append(recon)

    scale_loss, per_scale = compute_mtr_loss(
        actions,
        scale_recons,
        [1, 2, 4],
        dim_groups={"position": [0], "gripper": [1]},
        gripper_weight=0.0,
        scale_loss_weights={1: 0.1, 2: 0.3, 4: 0.6},
    )

    assert torch.allclose(per_scale[1], torch.tensor(1.0))
    assert torch.allclose(per_scale[2], torch.tensor(4.0))
    assert torch.allclose(per_scale[4], torch.tensor(9.0))
    assert torch.allclose(scale_loss, torch.tensor(6.7))

    with pytest.raises(ValueError, match="unknown intermediate scales"):
        compute_mtr_loss(
            actions,
            scale_recons,
            [1, 2, 4],
            dim_groups={"position": [0], "gripper": [1]},
            gripper_weight=0.0,
            scale_loss_weights={8: 1.0},
        )

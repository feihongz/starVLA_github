import pytest
import torch

from starVLA.model.modules.action_tokenizer.var_action_tokenizer import VARActionTokenizer
from starVLA.training.intermediate_supervision import (
    INTERMEDIATE_SUPERVISION_MODES,
    build_temporal_scale_target,
    compute_intermediate_loss,
    compute_mtr_loss,
    dct_ii,
    normalized_raw_dct_mse,
    weighted_dim_mse,
)


LIBERO_GROUPS = {
    "position": [0, 1, 2],
    "rotation": [3, 4, 5],
    "gripper": [6],
}

ROBOCASA_GROUPS = {
    "arm": list(range(14)),
    "hand": list(range(14, 26)),
    "waist": list(range(26, 29)),
}


def test_supported_modes_and_none_is_a_noop():
    assert INTERMEDIATE_SUPERVISION_MODES == {
        "none",
        "full_target_time",
        "mint_paper_dct",
        "mtr",
    }

    actions = torch.randn(2, 8, 7)
    loss, per_scale = compute_intermediate_loss(
        actions,
        [],
        [],
        mode="none",
        dim_groups=LIBERO_GROUPS,
    )

    assert loss.shape == ()
    assert loss.item() == 0.0
    assert per_scale == {}


@pytest.mark.parametrize(
    ("horizon", "action_dim", "scales", "dim_groups"),
    [
        (8, 7, [1, 2, 4], LIBERO_GROUPS),
        (16, 29, [1, 2, 4, 8], ROBOCASA_GROUPS),
    ],
)
@pytest.mark.parametrize("mode", ["full_target_time", "mint_paper_dct", "mtr"])
def test_all_supervised_modes_have_finite_forward_and_backward(
    horizon,
    action_dim,
    scales,
    dim_groups,
    mode,
):
    actions = torch.randn(2, horizon, action_dim)
    scale_recons = [
        torch.randn_like(actions, requires_grad=True) for _ in scales
    ]

    loss, per_scale = compute_intermediate_loss(
        actions,
        scale_recons,
        scales,
        mode=mode,
        dim_groups=dim_groups,
        gripper_weight=0.0 if action_dim == 7 else 1.0,
    )

    assert torch.isfinite(loss)
    assert set(per_scale) == set(scales)
    assert all(torch.isfinite(scale_loss) for scale_loss in per_scale.values())

    loss.backward()
    for scale_recon in scale_recons:
        assert scale_recon.grad is not None
        assert torch.isfinite(scale_recon.grad).all()


def test_dct_ii_h4_matches_reference_values():
    actions = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    expected = torch.tensor([[[10.0], [-3.15432203], [0.0], [-0.22417076]]])

    actual = dct_ii(actions, normalization="none")

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_orthonormal_dct_obeys_parseval_for_mse():
    torch.manual_seed(7)
    prediction = torch.randn(3, 8, 5)
    target = torch.randn_like(prediction)

    time_mse = torch.mean((prediction - target).square())
    frequency_mse = torch.mean(
        (
            dct_ii(prediction, normalization="ortho")
            - dct_ii(target, normalization="ortho")
        ).square()
    )

    assert torch.allclose(frequency_mse, time_mse, atol=2e-6, rtol=2e-6)


def test_normalized_raw_dct_mse_is_not_time_mse():
    target = torch.zeros(1, 4, 1)
    prediction = torch.ones_like(target)
    dim_groups = {"position": [0]}

    time_mse = weighted_dim_mse(
        prediction,
        target,
        dim_groups=dim_groups,
        gripper_weight=1.0,
    )
    raw_dct_mse = normalized_raw_dct_mse(
        prediction,
        target,
        dim_groups=dim_groups,
        gripper_weight=1.0,
    )

    assert torch.allclose(time_mse, torch.tensor(1.0))
    assert torch.allclose(raw_dct_mse, torch.tensor(2.0), atol=1e-6)
    assert not torch.allclose(raw_dct_mse, time_mse)


def test_mtr_uses_the_scale_matched_down_up_target():
    actions = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1)
    expected = torch.tensor(
        [[[1.5], [1.5], [2.0], [3.0], [4.0], [5.0], [5.5], [5.5]]]
    )
    scale_target = build_temporal_scale_target(actions, scale=2)
    assert torch.equal(scale_target, expected)

    mtr_loss, per_scale = compute_intermediate_loss(
        actions,
        [scale_target],
        [2],
        mode="mtr",
        dim_groups={"position": [0]},
    )
    full_target_loss, _ = compute_intermediate_loss(
        actions,
        [scale_target],
        [2],
        mode="full_target_time",
        dim_groups={"position": [0]},
    )

    assert torch.allclose(mtr_loss, torch.tensor(0.0))
    assert torch.allclose(per_scale[2], torch.tensor(0.0))
    assert full_target_loss > 0.0


def test_compute_mtr_loss_remains_a_compatibility_wrapper():
    actions = torch.randn(2, 8, 7)
    scale_recons = [torch.randn_like(actions) for _ in range(3)]
    scales = [1, 2, 4]

    expected, expected_per_scale = compute_intermediate_loss(
        actions,
        scale_recons,
        scales,
        mode="mtr",
        dim_groups=LIBERO_GROUPS,
        gripper_weight=0.0,
        scale_weights={1: 0.1, 2: 0.3, 4: 0.6},
    )
    actual, actual_per_scale = compute_mtr_loss(
        actions,
        scale_recons,
        scales,
        dim_groups=LIBERO_GROUPS,
        gripper_weight=0.0,
        scale_loss_weights={1: 0.1, 2: 0.3, 4: 0.6},
    )

    assert torch.allclose(actual, expected)
    assert actual_per_scale.keys() == expected_per_scale.keys()
    for scale in scales:
        assert torch.allclose(actual_per_scale[scale], expected_per_scale[scale])


@pytest.mark.parametrize("mode", ["full_target_time", "mint_paper_dct", "mtr"])
def test_intermediate_loss_rejects_the_final_scale(mode):
    actions = torch.randn(1, 8, 7)

    with pytest.raises(ValueError, match="final|horizon|intermediate"):
        compute_intermediate_loss(
            actions,
            [torch.randn_like(actions)],
            [8],
            mode=mode,
            dim_groups=LIBERO_GROUPS,
        )


def test_dct_loss_rejects_time_weights():
    actions = torch.randn(2, 8, 7)

    with pytest.raises(ValueError, match="time_weights|frequency"):
        compute_intermediate_loss(
            actions,
            [torch.randn_like(actions)],
            [1],
            mode="mint_paper_dct",
            dim_groups=LIBERO_GROUPS,
            time_weights=torch.ones(2, 8),
        )


def test_zero_gripper_weight_masks_only_intermediate_gripper_error():
    actions = torch.zeros(1, 8, 7)
    scale_recon = actions.clone()
    scale_recon[..., 6] = 100.0

    intermediate_loss, _ = compute_intermediate_loss(
        actions,
        [scale_recon],
        [1],
        mode="full_target_time",
        dim_groups=LIBERO_GROUPS,
        gripper_weight=0.0,
    )
    final_recon_loss = weighted_dim_mse(
        scale_recon,
        actions,
        dim_groups=LIBERO_GROUPS,
        gripper_weight=1.0,
    )

    assert torch.allclose(intermediate_loss, torch.tensor(0.0))
    assert final_recon_loss > 0.0


def test_base_forward_has_one_decoder_call_and_scale_final_matches_recon(
    monkeypatch,
):
    model = VARActionTokenizer(
        action_dim=7,
        seq_len=8,
        scales=[1, 2, 4, 8],
        embed_dim=32,
        quantization_mode="product_vq",
        product_codebook_groups=16,
        dim_groups=LIBERO_GROUPS,
    )
    actions = torch.randn(2, 8, 7)
    original_decode = model.decode_features
    decode_calls = []

    def counted_decode(features):
        decode_calls.append(features)
        return original_decode(features)

    monkeypatch.setattr(model, "decode_features", counted_decode)
    default_output = model(actions)

    assert len(decode_calls) == 1
    assert "scale_recons" not in default_output
    assert "scale_recon_scales" not in default_output

    decode_calls.clear()
    scale_output = model(actions, return_scale_recons=True)

    assert len(decode_calls) == 4
    assert scale_output["scale_recon_scales"] == [1, 2, 4, 8]
    assert torch.equal(scale_output["scale_recons"][-1], scale_output["recon"])


def test_full_target_time_uses_actions_at_every_intermediate_scale():
    actions = torch.randn(2, 8, 7)
    loss, per_scale = compute_intermediate_loss(
        actions,
        [actions.clone() for _ in range(3)],
        [1, 2, 4],
        mode="full_target_time",
        dim_groups=LIBERO_GROUPS,
        gripper_weight=0.0,
    )

    assert torch.equal(loss, torch.zeros_like(loss))
    assert all(
        torch.equal(scale_loss, torch.zeros_like(scale_loss))
        for scale_loss in per_scale.values()
    )


@pytest.mark.parametrize(
    ("selected_group", "expected"),
    [("arm", 1.0), ("hand", 4.0), ("waist", 9.0)],
)
def test_robocasa_arm_hand_waist_group_masks(selected_group, expected):
    actions = torch.zeros(1, 16, 29)
    scale_recon = torch.zeros_like(actions)
    scale_recon[..., :14] = 1.0
    scale_recon[..., 14:26] = 2.0
    scale_recon[..., 26:] = 3.0
    group_weights = {group: 0.0 for group in ROBOCASA_GROUPS}
    group_weights[selected_group] = 1.0

    loss, _ = compute_intermediate_loss(
        actions,
        [scale_recon],
        [1],
        mode="full_target_time",
        dim_groups=ROBOCASA_GROUPS,
        group_weights=group_weights,
    )

    assert torch.allclose(loss, torch.tensor(expected))


def test_unknown_mode_and_group_are_rejected():
    actions = torch.zeros(1, 8, 7)

    with pytest.raises(ValueError, match="Unknown intermediate supervision mode"):
        compute_intermediate_loss(
            actions,
            [],
            [],
            mode="not_a_mode",
            dim_groups=LIBERO_GROUPS,
        )

    with pytest.raises(ValueError, match="Unknown action dim group"):
        weighted_dim_mse(
            actions,
            actions,
            dim_groups=LIBERO_GROUPS,
            group_weights={"missing": 1.0},
        )


@pytest.mark.parametrize(
    "dim_groups",
    [
        {"position": [0, 0]},
        {"position": [7]},
        {"position": [0], "rotation": [0]},
    ],
)
def test_invalid_dim_groups_are_rejected(dim_groups):
    actions = torch.zeros(1, 8, 7)

    with pytest.raises(ValueError, match="duplicate|outside|appears in both"):
        weighted_dim_mse(
            actions,
            actions,
            dim_groups=dim_groups,
        )


@pytest.mark.parametrize(
    ("scale_weights", "message"),
    [
        ({1: -1.0}, "finite and non-negative"),
        ({1: 0.0}, "must be positive"),
        ({2: 1.0}, "unknown intermediate scales"),
    ],
)
def test_invalid_scale_weights_are_rejected(scale_weights, message):
    actions = torch.zeros(1, 8, 7)

    with pytest.raises(ValueError, match=message):
        compute_intermediate_loss(
            actions,
            [actions.clone()],
            [1],
            mode="full_target_time",
            dim_groups=LIBERO_GROUPS,
            scale_weights=scale_weights,
        )


def test_none_rejects_accidental_intermediate_decoder_outputs():
    actions = torch.zeros(1, 8, 7)

    with pytest.raises(ValueError, match="must not request"):
        compute_intermediate_loss(
            actions,
            [actions.clone()],
            [1],
            mode="none",
            dim_groups=LIBERO_GROUPS,
        )


def test_paper_dct_forces_fp32_and_backpropagates_from_bfloat16():
    prediction = torch.randn(
        2,
        8,
        7,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    target = torch.zeros_like(prediction)

    loss = normalized_raw_dct_mse(
        prediction,
        target,
        dim_groups=LIBERO_GROUPS,
        gripper_weight=0.0,
    )
    loss.backward()

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()

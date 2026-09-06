from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from deployment.model_server import policy_wrapper


def test_raw_and_model_state_dimensions_have_distinct_contracts(
    tmp_path, monkeypatch
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"x")

    framework = mock.Mock()
    framework.use_proprio_state = True
    framework.proprio_state_dim = 58
    framework.action_horizon = 2
    framework.to.return_value = framework
    framework.eval.return_value = framework
    framework.predict_action.return_value = {
        "normalized_actions": np.zeros((1, 2, 3), dtype=np.float32)
    }
    processor = SimpleNamespace(
        unnorm_key="gr1",
        available_unnorm_keys=["gr1"],
        state_dim=29,
        action_dim=3,
        state_keys=["state.raw"],
        action_keys=["action.raw"],
        state_key_dims={"state.raw": 29},
        action_key_dims={"action.raw": 3},
        unapply_actions=lambda actions: actions,
    )
    config = {
        "framework": {
            "proprio_state": {"enabled": True, "state_dim": 58},
            "action_model": {"action_horizon": 2},
        },
        "datasets": {"vla_data": {"obs": []}},
    }

    monkeypatch.setattr(
        policy_wrapper.baseframework,
        "from_pretrained",
        mock.Mock(return_value=framework),
    )
    monkeypatch.setattr(
        policy_wrapper,
        "read_mode_config",
        mock.Mock(return_value=(config, {"gr1": {}})),
    )
    monkeypatch.setattr(
        policy_wrapper,
        "PolicyNormProcessor",
        mock.Mock(return_value=processor),
    )

    wrapper = policy_wrapper.PolicyServerWrapper(
        str(checkpoint), device="cpu", unnorm_key="gr1"
    )

    assert wrapper.metadata["state_dim"] == 29
    assert wrapper.metadata["expected_state_dim"] == 58
    result = wrapper.predict_action(
        [{"state": np.zeros(58, dtype=np.float32)}]
    )
    assert result["actions"].shape == (1, 2, 3)
    assert framework.predict_action.call_count == 1
    forwarded = framework.predict_action.call_args.kwargs["examples"]
    assert forwarded[0]["state"].shape == (58,)

    with pytest.raises(
        ValueError, match=r"State dim mismatch.*expected \(58,\)"
    ):
        wrapper.predict_action(
            [{"state": np.zeros(29, dtype=np.float32)}]
        )
    assert framework.predict_action.call_count == 1

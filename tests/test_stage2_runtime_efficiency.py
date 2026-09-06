from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import torch

import starVLA.dataloader.var_stage2_token_dataset as token_dataset_module
import starVLA.training.train_starvla as trainer_module


def test_build_accelerator_applies_explicit_deepspeed_settings(monkeypatch):
    captured = {}

    class FakeDeepSpeedPlugin:
        def __init__(self, **kwargs):
            captured["plugin"] = kwargs

    class FakeInitProcessGroupKwargs:
        def __init__(self, **kwargs):
            captured["process_group"] = kwargs

    class FakeAccelerator:
        state = "fake-state"

        def __init__(self, **kwargs):
            captured["accelerator"] = kwargs

        def print(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(trainer_module, "DeepSpeedPlugin", FakeDeepSpeedPlugin)
    monkeypatch.setattr(trainer_module, "InitProcessGroupKwargs", FakeInitProcessGroupKwargs)
    monkeypatch.setattr(trainer_module, "Accelerator", FakeAccelerator)
    monkeypatch.setenv("DEEPSPEED_REDUCE_BUCKET_SIZE", "123456789")
    monkeypatch.setenv("DEEPSPEED_ALLGATHER_BUCKET_SIZE", "98765432")
    monkeypatch.setenv("TORCH_DISTRIBUTED_TIMEOUT_SECONDS", "3456")

    cfg = SimpleNamespace(
        trainer=SimpleNamespace(
            gradient_accumulation_steps=2,
            gradient_clipping=1.0,
        )
    )
    trainer_module.build_accelerator(cfg)

    plugin_kwargs = captured["plugin"]
    ds_config = plugin_kwargs["hf_ds_config"]
    zero_config = ds_config["zero_optimization"]
    assert ds_config["gradient_accumulation_steps"] == 2
    assert ds_config["gradient_clipping"] == 1.0
    assert ds_config["bf16"] == {"enabled": True}
    assert ds_config["fp16"] == {"enabled": False}
    assert zero_config["stage"] == 2
    assert zero_config["reduce_bucket_size"] == 123456789
    assert zero_config["allgather_bucket_size"] == 98765432
    assert zero_config["offload_optimizer"]["device"] == "none"
    assert zero_config["offload_param"]["device"] == "none"
    assert plugin_kwargs["gradient_accumulation_steps"] == 2
    assert plugin_kwargs["gradient_clipping"] == 1.0
    assert captured["process_group"]["timeout"] == timedelta(seconds=3456)
    assert len(captured["accelerator"]["kwargs_handlers"]) == 1


def test_token_cache_load_prefers_mmap_and_drops_unused_metadata(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.pt"
    expected_tokens = torch.arange(12).reshape(3, 4)
    torch.save(
        {
            "tokens": expected_tokens,
            "metadata": {"cached_len": 3},
            "sample_metadata": [{"unused": index} for index in range(3)],
        },
        cache_path,
    )

    original_torch_load = torch.load
    load_kwargs = []

    def recording_torch_load(*args, **kwargs):
        load_kwargs.append(dict(kwargs))
        return original_torch_load(*args, **kwargs)

    monkeypatch.setattr(token_dataset_module.torch, "load", recording_torch_load)
    dataset = object.__new__(token_dataset_module.VARStage2TokenDataset)
    cache = dataset._load_token_cache(cache_path)

    assert load_kwargs[0]["mmap"] is True
    assert torch.equal(cache["tokens"], expected_tokens)
    assert cache["metadata"] == {"cached_len": 3}
    assert "sample_metadata" not in cache
    assert cache["path"] == str(cache_path)


def test_token_cache_load_falls_back_for_legacy_serialization(tmp_path):
    cache_path = tmp_path / "legacy_cache.pt"
    expected_tokens = torch.arange(8).reshape(2, 4)
    torch.save(
        {"tokens": expected_tokens, "metadata": {"cached_len": 2}},
        cache_path,
        _use_new_zipfile_serialization=False,
    )

    dataset = object.__new__(token_dataset_module.VARStage2TokenDataset)
    cache = dataset._load_token_cache(cache_path)

    assert torch.equal(cache["tokens"], expected_tokens)
    assert cache["path"] == str(cache_path)

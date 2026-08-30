from __future__ import annotations

import ast
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/stage1/run_clean_ablation_4xh100_8parallel.sh"
METHODS = (
    "multiscale_base",
    "full_target_time",
    "mint_paper_dct",
    "mtr",
)
EXPECTED_GPUS = {
    ("libero", "multiscale_base"): "0",
    ("libero", "full_target_time"): "1",
    ("libero", "mint_paper_dct"): "2",
    ("libero", "mtr"): "3",
    ("robocasa", "multiscale_base"): "3",
    ("robocasa", "full_target_time"): "2",
    ("robocasa", "mint_paper_dct"): "1",
    ("robocasa", "mtr"): "0",
}
ROBOCASA_REGISTRY = (
    REPO_ROOT
    / "examples/Robocasa_tabletop/train_files/data_registry/data_config.py"
)


def _robocasa_datasets() -> list[str]:
    module = ast.parse(ROBOCASA_REGISTRY.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id == "LOCAL_GR1_UNIFIED_1000_DATASETS"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            assert isinstance(value, list)
            return value
    raise AssertionError("RoboCasa dataset registry was not found")


def _tree_payload(root: Path) -> dict[str, bytes | None]:
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for path in sorted(root.rglob("*"))
    }


def _option(command: list[str], name: str) -> str:
    index = command.index(name)
    return command[index + 1]


def test_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True, cwd=REPO_ROOT)


def test_dry_run_is_read_only_and_materializes_the_locked_eight_commands(
    tmp_path: Path,
) -> None:
    libero_data = tmp_path / "libero_data"
    robocasa_data = tmp_path / "robocasa_data"
    libero_data.mkdir()
    robocasa_data.mkdir()
    for dataset in _robocasa_datasets():
        (robocasa_data / dataset).mkdir()
    libero_init = tmp_path / "libero_init.ckpt"
    robocasa_init = tmp_path / "robocasa_init.ckpt"
    libero_init.write_bytes(b"libero-fixed-init")
    robocasa_init.write_bytes(b"robocasa-fixed-init")
    checkpoint_root = tmp_path / "outputs"
    log_root = tmp_path / "logs"
    before = _tree_payload(tmp_path)

    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "RESUME_QUEUE": "0",
            "SKIP_SMOKE": "1",
            "SKIP_GPU_PREFLIGHT": "1",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "LAUNCH_STAGGER_SECONDS": "0",
            "PYTHON_BIN": sys.executable,
            "STAGE1_ABLATION_CHECKPOINT_ROOT": str(checkpoint_root),
            "STAGE1_LAUNCHER_LOG_ROOT": str(log_root),
            "LIBERO_DATA_ROOT": str(libero_data),
            "ROBOCASA_DATA_ROOT": str(robocasa_data),
            "LIBERO_STAGE1_INIT_CHECKPOINT": str(libero_init),
            "ROBOCASA_STAGE1_INIT_CHECKPOINT": str(robocasa_init),
            "LIBERO_INTERMEDIATE_WEIGHT": "0.02",
            "ROBOCASA_INTERMEDIATE_WEIGHT": "0.1",
        }
    )
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert _tree_payload(tmp_path) == before
    assert not checkpoint_root.exists()
    assert not log_root.exists()
    assert "DRY_RUN complete: no training/output/log directories were created." in completed.stdout

    pattern = re.compile(
        r"command\[(libero|robocasa)_train_("
        + "|".join(METHODS)
        + r")_gpu([0-3])\]: ([^\n]+)"
    )
    matches = pattern.findall(completed.stdout)
    assert len(matches) == 8
    assert {(benchmark, method) for benchmark, method, _, _ in matches} == set(
        EXPECTED_GPUS
    )

    processes_per_gpu: dict[str, list[str]] = {str(index): [] for index in range(4)}
    for benchmark, method, gpu, command_text in matches:
        assert gpu == EXPECTED_GPUS[(benchmark, method)]
        processes_per_gpu[gpu].append(benchmark)
        command = shlex.split(command_text)
        wrapper = (
            REPO_ROOT
            / (
                "examples/LIBERO/train_files/run_stage1_clean_supervision_ablation.sh"
                if benchmark == "libero"
                else "examples/Robocasa_tabletop/train_files/run_stage1_clean_supervision_ablation.sh"
            )
        )
        assert command[:2] == ["bash", str(wrapper)]
        assert _option(command, "--methods") == method
        assert _option(command, "--seeds") == "42"
        assert _option(command, "--gpus") == gpu
        assert _option(command, "--mode") == "train"
        assert _option(command, "--epochs") == "50"
        assert _option(command, "--checkpoint-root") == str(checkpoint_root)
        assert _option(command, "--data-root") == str(
            libero_data if benchmark == "libero" else robocasa_data
        )
        assert _option(command, "--init-checkpoint") == str(
            libero_init if benchmark == "libero" else robocasa_init
        )
        expected_python = Path(sys.executable).parent.resolve() / Path(sys.executable).name
        assert _option(command, "--python-bin") == str(expected_python)
        assert _option(command, "--intermediate-weight") == (
            "0.02" if benchmark == "libero" else "0.1"
        )
        assert "--dry-run" in command
        assert "--resume" not in command

    assert all(sorted(benchmarks) == ["libero", "robocasa"] for benchmarks in processes_per_gpu.values())
    assert "seed=42, epochs=50, batch=256/process, workers=8/process" in completed.stdout

    libero_cfg = OmegaConf.load(
        REPO_ROOT
        / "examples/LIBERO/train_files/train_var_stage1_libero_clean_supervision_ablation.yaml"
    )
    robocasa_cfg = OmegaConf.load(
        REPO_ROOT
        / "examples/Robocasa_tabletop/train_files/train_var_stage1_robocasa_clean_supervision_ablation.yaml"
    )
    assert int(libero_cfg.train.batch_size) == 256
    assert int(robocasa_cfg.train.batch_size) == 256

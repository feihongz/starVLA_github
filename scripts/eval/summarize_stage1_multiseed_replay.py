from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/home/zhangfeihong/starVLA/playground/Checkpoints")
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
EXPERIMENTS = {
    "g16_s8_structemb": ROOT / "var_stage1_pi05_libero_q99_e32_aeinit_productvq_g16_s8_structemb",
    "g16_s4_8": ROOT / "var_stage1_pi05_libero_q99_e32_aeinit_productvq_g16_s4_8",
    "g8_s4_8": ROOT / "var_stage1_pi05_libero_q99_e32_aeinit_productvq_g8_s4_8",
    "g8_s8_cb1024": ROOT / "var_stage1_pi05_libero_q99_e32_aeinit_productvq_g8_s8_cb1024",
    "g8_baseline": ROOT / "var_stage1_pi05_libero_q99_e32_aeinit_productvq_g8",
}


def summarize_one(name: str, exp_dir: Path) -> list[tuple[int, int, int, float, list[str]]]:
    out_dir = exp_dir / "replay_multiseed"
    seeds = sorted(
        {
            int(path.stem.split("_seed", 1)[1].split("_", 1)[0])
            for path in out_dir.glob("oracle_replay_*_seed*_recon.json")
            if "_seed" in path.stem
        }
    )
    rows = []
    for seed in seeds:
        successes = 0
        episodes = 0
        parts = []
        missing = []
        for suite in SUITES:
            path = out_dir / f"oracle_replay_{suite}_seed{seed}_recon.json"
            if not path.exists():
                missing.append(suite)
                continue
            data = json.loads(path.read_text())
            item = data["summary"]["recon"]
            s = int(item["successes"])
            e = int(item["episodes"])
            successes += s
            episodes += e
            parts.append(f"{suite}:{s}/{e}")
        rows.append((seed, successes, episodes, successes / max(episodes, 1), parts + [f"missing:{','.join(missing)}"] if missing else parts))
    return rows


def main() -> None:
    for name, exp_dir in EXPERIMENTS.items():
        rows = summarize_one(name, exp_dir)
        print(f"\n{name}")
        if not rows:
            print("  no completed multiseed reports")
            continue
        for seed, successes, episodes, rate, parts in sorted(rows, key=lambda row: row[3], reverse=True):
            print(f"  seed={seed}: {successes}/{episodes} = {rate:.3f} | " + " ".join(parts))


if __name__ == "__main__":
    main()

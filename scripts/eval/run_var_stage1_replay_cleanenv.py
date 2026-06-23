from __future__ import annotations

import runpy
import sys
from pathlib import Path


LIBERO_SITE_PACKAGES = "/home/zhangfeihong/miniconda3/envs/libero/lib/python3.10/site-packages"
REPLAY_SCRIPT = Path("examples/LIBERO/eval_files/eval_var_stage1_oracle_replay.py")


def main() -> None:
    sys.path.append(LIBERO_SITE_PACKAGES)
    sys.argv = [str(REPLAY_SCRIPT), *sys.argv[1:]]
    runpy.run_path(str(REPLAY_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()

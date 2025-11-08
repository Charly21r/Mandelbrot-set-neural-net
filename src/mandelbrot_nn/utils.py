from __future__ import annotations

import json
import time
from pathlib import Path


def make_run_dir(base="runs", tag=""):
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    name = ts + (f"_{tag}" if tag else "")
    run_dir = Path(base) / name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "images").mkdir(exist_ok=True)
    (run_dir / "ckpt").mkdir(exist_ok=True)
    return run_dir


def save_config(cfg: dict, run_dir: Path):
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

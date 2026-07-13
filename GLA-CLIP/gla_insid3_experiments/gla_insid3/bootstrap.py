"""Load INSID3 without modifying its source tree."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any


def activate_insid3(insid3_root: str | Path) -> Path:
    root = Path(insid3_root).expanduser().resolve()
    required = (root / "models" / "insid3.py", root / "utils" / "clustering.py")
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Invalid INSID3 root; missing: {missing}")
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return root


def build_model(insid3_root: str | Path, **kwargs: Any):
    activate_insid3(insid3_root)
    module = importlib.import_module("models")
    return module.build_insid3(**kwargs)


def sha256_file(path: str | Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_manifest(
    insid3_root: str | Path,
    experiment_root: str | Path,
    args: dict[str, Any],
) -> dict[str, Any]:
    insid3_root = Path(insid3_root).resolve()
    experiment_root = Path(experiment_root).resolve()
    tracked = [
        insid3_root / "models" / "insid3.py",
        insid3_root / "utils" / "clustering.py",
        insid3_root / "utils" / "data.py",
        experiment_root / "run_experiment.py",
    ]
    return {
        "format_version": 1,
        "arguments": args,
        "files": {
            str(path): sha256_file(path) for path in tracked if path.is_file()
        },
        "notes": {
            "positional_basis": "INSID3 current zero-image basis",
            "forward_gate": True,
            "area_weight": True,
            "clustering_spatial_connectivity": False,
            "network_required": False,
        },
    }


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

"""Load INSID3 without modifying its source tree."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import platform
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
        insid3_root / "models" / "__init__.py",
        insid3_root / "models" / "insid3.py",
        insid3_root / "utils" / "clustering.py",
        insid3_root / "utils" / "data.py",
    ]
    tracked.extend(sorted(experiment_root.glob("*.py")))
    tracked.extend(sorted((experiment_root / "gla_insid3").glob("*.py")))

    def package_version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "format_version": 2,
        "arguments": args,
        "files": {
            str(path): sha256_file(path) for path in tracked if path.is_file()
        },
        "notes": {
            "positional_basis": "INSID3 current zero-image basis",
            "forward_gate": True,
            "forward_gate_mode": args.get("forward_gate_mode", "zero"),
            "attention_cutoff_before_softmax": True,
            "area_weight": True,
            "clustering_spatial_connectivity": False,
            "network_required": False,
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                name: package_version(name)
                for name in ("torch", "torchvision", "numpy", "Pillow", "scikit-learn")
            },
        },
    }


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

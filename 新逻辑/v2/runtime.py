"""Runtime setup for V2 without changing Blender's Python environment."""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path


V2_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = V2_ROOT.parent
REPO_ROOT = PROJECT_ROOT / "source" / "flowhmr-main"
ASCII_MODEL_ROOT = Path(r"C:\mocap_v2_models")
ASCII_OUTPUT_ROOT = Path(r"C:\mocap_v2_outputs")


def configure_imports() -> None:
    paths = [
        PROJECT_ROOT,
        REPO_ROOT,
        REPO_ROOT / "third_party" / "YOLOX",
        REPO_ROOT / "third_party" / "sam-3d-body",
        REPO_ROOT / "third_party" / "vggt-omega",
        PROJECT_ROOT / "runtime_packages",
    ]
    for path in reversed(paths):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    # TorchScript's Windows loader cannot reliably open non-ASCII paths.
    # The junction is created once by setup_ascii_model_paths.py and points to
    # the real model directory; no model data is duplicated.
    sam_dir = ASCII_MODEL_ROOT / "sam-3d-body-dinov3"
    if sam_dir.exists():
        os.environ.setdefault("MOCAP_V2_SAM_CKPT_DIR", str(sam_dir))


def collect_runtime_info() -> dict:
    info = {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "repo_root": str(REPO_ROOT),
        "project_root": str(PROJECT_ROOT),
        "environment": os.environ.get("CONDA_DEFAULT_ENV", ""),
    }
    try:
        import torch
        info.update({
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        })
    except Exception as exc:
        info["torch_error"] = repr(exc)
    return info


def write_runtime_info(path: str | Path) -> dict:
    info = collect_runtime_info()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(info, ensure_ascii=True, indent=2), encoding="utf-8")
    return info

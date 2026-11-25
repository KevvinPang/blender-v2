"""Create ASCII-only junctions for Windows native model loaders."""

from __future__ import annotations

import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASCII_ROOT = Path(r"C:\mocap_v2_models")
ASCII_OUTPUT_ROOT = Path(r"C:\mocap_v2_outputs")


def main() -> None:
    ASCII_ROOT.mkdir(parents=True, exist_ok=True)
    if not ASCII_OUTPUT_ROOT.exists():
        subprocess.run([
            "cmd", "/c", "mklink", "/J", str(ASCII_OUTPUT_ROOT),
            str((PROJECT_ROOT / "outputs").resolve()),
        ], check=True)
    links = {
        "sam-3d-body-dinov3": PROJECT_ROOT / "models" / "sam-3d-body-dinov3",
        "smplh": PROJECT_ROOT / "models" / "body_models" / "smplh",
    }
    for name, target in links.items():
        link = ASCII_ROOT / name
        if link.exists():
            print(f"exists: {link}")
            continue
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target.resolve())], check=True)
        print(f"created: {link} -> {target.resolve()}")


if __name__ == "__main__":
    main()

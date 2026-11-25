"""Read video metadata without changing or transcoding the source."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2


def probe_video(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {source}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        duration = frame_count / fps if fps > 0 else 0.0
    finally:
        cap.release()
    if width <= 0 or height <= 0 or frame_count <= 0:
        raise ValueError(f"invalid video metadata: {source}")
    return {
        "path": str(source),
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "fps": fps,
        "duration_seconds": duration,
    }

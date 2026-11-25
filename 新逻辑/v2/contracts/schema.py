"""Serializable contracts shared by V2 stages.

The pipeline keeps source observations separate from derived Blender data.  All
arrays are stored in NPZ/NPY files; this JSON contract only contains metadata,
paths, conventions and quality flags.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Space(str, Enum):
    IMAGE_PX = "image_px"
    CAMERA = "camera"
    SOURCE_WORLD = "source_world"
    CHARACTER_LOCAL = "character_local"
    BLENDER_WORLD = "blender_world"


class ObservationState(str, Enum):
    OBSERVED = "observed"
    OCCLUDED = "occluded"
    OUT_OF_FRAME = "out_of_frame"
    INFERRED = "inferred"
    INVALID = "invalid"


@dataclass
class CameraTrack:
    schema_version: str = "mocap-v2-camera-1"
    mode: str = "dynamic"
    fps: float = 30.0
    frame_start: int = 0
    frame_end: int = -1
    image_size_hw: tuple[int, int] = (0, 0)
    source_convention: str = "opencv_cam_from_world"
    target_convention: str = "blender_world_camera"
    raw_npz: str = ""
    normalized_npz: str = ""
    smoothing: dict[str, Any] = field(default_factory=dict)
    scale: dict[str, Any] = field(default_factory=lambda: {
        "mode": "relative",
        "meters_per_source_unit": None,
        "confidence": 0.0,
    })
    quality: dict[str, Any] = field(default_factory=dict)

    @property
    def frame_count(self) -> int:
        return max(0, self.frame_end - self.frame_start + 1)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=True, indent=2), encoding="utf-8")


@dataclass
class MotionTake:
    schema_version: str = "mocap-v2-take-1"
    source_video: str = ""
    source_sha256: str = ""
    fps: float = 30.0
    frame_start: int = 0
    frame_end: int = -1
    camera_mode: str = "dynamic"
    camera_track_json: str = ""
    motion_npz: str = ""
    character_profile: str = ""
    solver_config: str = ""
    backend: dict[str, Any] = field(default_factory=dict)
    coordinate_spaces: list[str] = field(default_factory=lambda: [s.value for s in Space])
    notes: list[str] = field(default_factory=list)

    @property
    def frame_count(self) -> int:
        return max(0, self.frame_end - self.frame_start + 1)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=True, indent=2), encoding="utf-8")


@dataclass
class QualityReport:
    schema_version: str = "mocap-v2-quality-1"
    passed: bool = False
    checks: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=True, indent=2), encoding="utf-8")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()

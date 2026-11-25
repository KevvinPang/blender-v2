"""Camera trajectory utilities with explicit coordinate and scale handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _as_homogeneous(rt: np.ndarray) -> np.ndarray:
    rt = np.asarray(rt, dtype=np.float32)
    if rt.ndim != 3 or rt.shape[1:] not in ((3, 4), (4, 4)):
        raise ValueError(f"expected (T,3,4) or (T,4,4), got {rt.shape}")
    if rt.shape[1:] == (3, 4):
        out = np.repeat(np.eye(4, dtype=np.float32)[None], rt.shape[0], axis=0)
        out[:, :3, :4] = rt
        return out
    return rt


def load_camera_npz(path: str | Path) -> dict[str, Any]:
    """Load raw VGGT output and preserve its source convention."""
    with np.load(path, allow_pickle=False) as data:
        if "RT_vggt" in data.files:
            world_to_camera = _as_homogeneous(data["RT_vggt"])
        elif "camera_RT" in data.files:
            world_to_camera = _as_homogeneous(data["camera_RT"])
        elif "RT" in data.files:
            world_to_camera = _as_homogeneous(data["RT"])
        else:
            raise KeyError(f"no RT/camera_RT/RT_vggt in {path}")
        if "K" not in data.files:
            raise KeyError(f"no K in {path}")
        K = np.asarray(data["K"], dtype=np.float32)
        image_size = tuple(int(x) for x in data.get("image_size_hw", [0, 0]))
        frame_range = tuple(int(x) for x in data.get("frame_range", [0, len(world_to_camera) - 1]))
        mode = str(data.get("camera_mode", "dynamic"))
    if K.shape != (len(world_to_camera), 3, 3):
        raise ValueError(f"K length {K.shape} does not match RT length {world_to_camera.shape}")
    return {
        "world_to_camera": world_to_camera,
        "camera_to_world": np.linalg.inv(world_to_camera),
        "K": K,
        "image_size_hw": image_size,
        "frame_range": frame_range,
        "mode": mode,
        "source_convention": "opencv_cam_from_world",
    }


def normalize_camera_track(raw: dict[str, Any], *, anchor_frame: int = 0,
                           meters_per_source_unit: float | None = None) -> dict[str, Any]:
    """Put the first camera at the origin and keep VGGT scale explicit.

    VGGT's translation is relative scale.  Normalization removes the arbitrary
    world origin but never pretends that the translation is measured in meters.
    """
    c2w = np.asarray(raw["camera_to_world"], dtype=np.float32).copy()
    if not 0 <= anchor_frame < len(c2w):
        raise ValueError(f"anchor_frame {anchor_frame} outside {len(c2w)} frames")
    anchor_inv = np.linalg.inv(c2w[anchor_frame])
    c2w = np.einsum("ij,tjk->tik", anchor_inv, c2w)
    if meters_per_source_unit is not None:
        c2w[:, :3, 3] *= float(meters_per_source_unit)
    w2c = np.linalg.inv(c2w)
    return {
        **raw,
        "camera_to_world": c2w,
        "world_to_camera": w2c,
        "scale": {
            "mode": "calibrated" if meters_per_source_unit is not None else "relative",
            "meters_per_source_unit": meters_per_source_unit,
            "confidence": 1.0 if meters_per_source_unit is not None else 0.0,
        },
        "anchor_frame": int(anchor_frame),
    }


def smooth_camera_track(track: dict[str, Any], *, window: int = 5,
                        rotation_alpha: float = 0.35,
                        translation_alpha: float = 0.25) -> dict[str, Any]:
    """Apply conservative EMA smoothing to camera translation and rotation.

    Rotation is smoothed through quaternion SLERP; no frame is dropped, so the
    result remains aligned with motion and video frames.
    """
    if window < 1:
        raise ValueError("window must be positive")
    from scipy.spatial.transform import Rotation as R

    c2w = np.asarray(track["camera_to_world"], dtype=np.float32).copy()
    if len(c2w) == 0:
        return {**track, "smoothing": {"window": window, "rotation_alpha": rotation_alpha,
                                        "translation_alpha": translation_alpha}}
    trans = c2w[:, :3, 3].copy()
    rots = R.from_matrix(c2w[:, :3, :3])
    q = rots.as_quat()
    q_out = q.copy()
    t_out = trans.copy()
    for i in range(1, len(c2w)):
        a = float(np.clip(rotation_alpha, 0.0, 1.0))
        b = float(np.clip(translation_alpha, 0.0, 1.0))
        q_prev = R.from_quat(q_out[i - 1])
        q_cur = R.from_quat(q[i])
        delta = (q_prev.inv() * q_cur).as_rotvec() * a
        q_out[i] = (q_prev * R.from_rotvec(delta)).as_quat()
        t_out[i] = (1.0 - b) * t_out[i - 1] + b * trans[i]
    smoothed = c2w.copy()
    smoothed[:, :3, :3] = R.from_quat(q_out).as_matrix().astype(np.float32)
    smoothed[:, :3, 3] = t_out
    return {
        **track,
        "camera_to_world": smoothed,
        "world_to_camera": np.linalg.inv(smoothed),
        "smoothing": {
            "window": int(window),
            "rotation_alpha": float(rotation_alpha),
            "translation_alpha": float(translation_alpha),
        },
    }


def save_normalized_camera_npz(track: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        camera_to_world=np.asarray(track["camera_to_world"], dtype=np.float32),
        world_to_camera=np.asarray(track["world_to_camera"], dtype=np.float32),
        K=np.asarray(track["K"], dtype=np.float32),
        image_size_hw=np.asarray(track["image_size_hw"], dtype=np.int64),
        frame_range=np.asarray(track["frame_range"], dtype=np.int64),
        camera_mode=str(track.get("mode", "dynamic")),
        camera_world="normalized_source_world",
        scale_mode=str(track.get("scale", {}).get("mode", "relative")),
        meters_per_source_unit=np.asarray(
            track.get("scale", {}).get("meters_per_source_unit")
            if track.get("scale", {}).get("meters_per_source_unit") is not None else np.nan,
            dtype=np.float32,
        ),
    )

"""Export a normalized camera track for Blender without changing its scale semantics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .trajectory import load_camera_npz, normalize_camera_track, smooth_camera_track


_OPENCV_TO_BLENDER_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def opencv_c2w_to_blender_c2w(camera_to_world: np.ndarray) -> np.ndarray:
    """Convert OpenCV camera-local (+Z forward, Y down) to Blender camera-local."""
    c2w = np.asarray(camera_to_world, dtype=np.float32)
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"expected (T,4,4), got {c2w.shape}")
    return np.einsum("tij,jk->tik", c2w, _OPENCV_TO_BLENDER_CAMERA)


def export_blender_camera_npz(raw_path: str | Path, output_path: str | Path, *,
                              smooth: bool = True, window: int = 5,
                              rotation_alpha: float = 0.35,
                              translation_alpha: float = 0.25) -> dict:
    raw = load_camera_npz(raw_path)
    track = normalize_camera_track(raw)
    if smooth:
        track = smooth_camera_track(
            track, window=window, rotation_alpha=rotation_alpha,
            translation_alpha=translation_alpha,
        )
    blender_c2w = opencv_c2w_to_blender_c2w(track["camera_to_world"])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        camera_to_world=blender_c2w,
        world_to_camera=np.linalg.inv(blender_c2w),
        K=track["K"],
        image_size_hw=np.asarray(track["image_size_hw"], dtype=np.int64),
        frame_range=np.asarray(track["frame_range"], dtype=np.int64),
        camera_mode="dynamic",
        camera_world="blender_world_camera",
        source_convention="opencv_cam_from_world",
        scale_mode=str(track["scale"]["mode"]),
        meters_per_source_unit=np.asarray(np.nan, dtype=np.float32),
        smoothing_window=np.asarray(window, dtype=np.int32),
        rotation_alpha=np.asarray(rotation_alpha, dtype=np.float32),
        translation_alpha=np.asarray(translation_alpha, dtype=np.float32),
    )
    return {
        "output": str(output_path.resolve()),
        "frame_count": int(len(blender_c2w)),
        "scale_mode": track["scale"]["mode"],
        "smoothing": track["smoothing"],
    }

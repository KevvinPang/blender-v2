"""Fit continuous wrist targets before the constrained arm solver."""
import numpy as np
from mathutils import Vector

MAX_SHIFT_METERS = .06


def fit_goals(times, goals, cameras, rows, seconds=.075):
    from sam_ellen_temporal import trajectory

    times = np.asarray(times, dtype=float)
    original = np.asarray(goals, dtype=float)
    if original.ndim != 2 or original.shape[1] != 3:
        raise ValueError('Wrist goals must have shape (frames, 3)')
    if len(times) != len(original) or len(cameras) != len(times) or len(rows) != len(times):
        raise ValueError('Wrist goals, cameras, and evidence must have matching lengths')
    if len(times) < 3:
        return original.copy(), {'model': 'confidence-weighted-wrist-goals-v1', 'samples': len(times)}

    # Missing-frame targets already come from the solver's conservative
    # torso-relative fallback. They are not measurements and must not be
    # pulled toward the nearest visible gesture by a whole-take fit.
    locked = np.array([r.get('locked', False) or not r.get('visible', False)
                       for r in rows], dtype=bool)
    reliability = np.clip([r.get('tracking_reliability', .5) for r in rows], 0., 1.)
    fast = np.clip([0. if r.get('inferred_hand', False) else r.get('fast_motion', 0.)
                    for r in rows], 0., 1.)
    weights = (.15 + 1.5 * reliability) * (1. + 60. * fast)
    weights *= 1. + 12. * np.array([r.get('semantic_weight', 0.) for r in rows])
    weights[locked] = 1e8

    camera_points = np.array([camera[0] @ Vector(goal)
                              for camera, goal in zip(cameras, original)])
    perspective = np.array([camera[3] == 'PERSP' for camera in cameras])
    depth = np.where(perspective, np.maximum(.05, -camera_points[:, 2]), 1.)
    rays = camera_points[:, :2] / depth[:, None]
    smooth_rays = trajectory(times, rays, seconds, weights, robust=.018)
    smooth_z = trajectory(times, camera_points[:, 2], seconds * 1.25, weights, robust=.025)

    depth = np.where(perspective, np.maximum(.05, -smooth_z), 1.)
    smooth_z = np.where(perspective, -depth, smooth_z)
    camera_fitted = np.c_[smooth_rays * depth[:, None], smooth_z]
    result = np.array([camera[0].inverted() @ Vector(point)
                       for camera, point in zip(cameras, camera_fitted)])

    correction = result - original
    correction_length = np.linalg.norm(correction, axis=1)
    scale = np.minimum(1., MAX_SHIFT_METERS / np.maximum(correction_length, 1e-12))
    result = original + correction * scale[:, None]

    if locked.any():
        distance = np.min(abs(times[:, None] - times[locked][None, :]), axis=1)
        alpha = np.clip(distance / .12, 0., 1.)
        alpha = alpha * alpha * (3. - 2. * alpha)
        result = original + (result - original) * alpha[:, None]
    result[locked] = original[locked]

    shift = np.linalg.norm(result - original, axis=1)
    return result, {
        'model': 'confidence-weighted-wrist-goals-v1',
        'ray_seconds': seconds,
        'depth_seconds': seconds * 1.25,
        'max_shift_m': float(shift.max()),
        'shift_limit_m': MAX_SHIFT_METERS,
        'mean_shift_m': float(shift.mean()),
        'protected_samples': int(locked.sum()),
        'fast_supported_samples': int((fast > .3).sum()),
    }

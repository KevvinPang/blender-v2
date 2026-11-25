"""Shared, zero-phase camera/actor temporal coordinates (NumPy only).

Background transforms map frame zero pixels into each frame. They are an
image-motion reference, not a metric reconstruction of the filming room.
"""
import numpy as np


def smooth(values, fps, seconds):
    raw = np.asarray(values, float)
    out = raw.copy()
    sigma = max(.6, fps * seconds)
    radius = int(np.ceil(3 * sigma))
    for t in range(len(raw)):
        ids = np.arange(max(0, t-radius), min(len(raw), t+radius+1))
        dt = ids-t
        w = np.exp(-.5*(dt/sigma)**2)
        s0, s1, s2 = w.sum(), (w*dt).sum(), (w*dt*dt).sum()
        y0, y1 = np.einsum('i,i...->...', w, raw[ids]), np.einsum('i,i...->...', w*dt, raw[ids])
        den = s0*s2-s1*s1
        out[t] = y0/s0 if den < 1e-12 else (s2*y0-s1*y1)/den
    return out


def transform_points(H, points):
    p = np.einsum('tij,tj->ti', H, np.c_[points, np.ones(len(points))])
    return p[:, :2] / p[:, 2:3]


def temporal_support(confidence,fps):
    conf=np.clip(np.asarray(confidence,float),0,1)
    radius=max(1,int(np.ceil(fps*.33)))
    safe=[]
    for t in range(len(conf)):
        chunk=conf[max(0,t-radius):min(len(conf),t+radius+1)]
        safe.append(0. if chunk.min()<=0 else np.clip((chunk.mean()-.25)/.4,0,1))
    return np.clip(smooth(np.array(safe),fps,.1),0,1)


def stabilize_relative(samples, fps, background=None, confidence=None):
    """Denoise actor position/scale in background-relative image coordinates.

Reapply measured global image motion exactly. A camera shake therefore does
not become an actor stabilization offset. Smooth only the residual. Confidence
is faded across a full filter neighbourhood so missing tracking never snaps.
"""
    raw = np.asarray(samples, float)
    if background is None:
        return smooth(raw, fps, .11), {'background_used': False}
    H = np.asarray(background, float)
    if H.shape != (len(raw), 3, 3) or not np.isfinite(H).all():
        raise ValueError('Background and actor frame counts/coordinates differ')
    conf = np.clip(np.asarray(confidence, float), 0, 1)
    # Confidence requires the entire neighbourhood to have good support.
    safe = temporal_support(conf,fps)
    scale = np.sqrt(np.maximum(np.linalg.det(H[:, :2, :2]), 1e-12))
    relative = raw.copy()
    relative[:, :2] = transform_points(np.linalg.inv(H), raw[:, :2])
    relative[:, 2] -= np.log(scale)
    fitted = smooth(relative, fps, .11)
    # Real sustained/fast displacement is retained by reducing smoothing when
    # neighbouring velocity agrees. Isolated alternating jitter gets no bypass.
    v = np.diff(relative[:, :2], axis=0, prepend=relative[:1, :2])
    left, right = smooth(v, fps, .045), np.roll(smooth(v, fps, .045), -1, axis=0)
    agreement = np.maximum(0, np.sum(left*right, axis=1)) / np.maximum(np.linalg.norm(left,axis=1)*np.linalg.norm(right,axis=1),1e-9)
    speed = np.minimum(np.linalg.norm(left,axis=1), np.linalg.norm(right,axis=1))*fps
    fast = np.clip((speed-100)/250, 0, .85)*agreement**4
    # Keep the turning point of a quick gesture too: instantaneous speed is
    # zero there, but lowering the gate would flatten/split the motion peak.
    reach=max(1,int(np.ceil(fps*.12)))
    fast=np.array([fast[max(0,t-reach):min(len(fast),t+reach+1)].max() for t in range(len(fast))])
    fitted[:, :2] = fitted[:, :2]*(1-fast[:,None])+relative[:, :2]*fast[:,None]
    fitted[:, :2] = transform_points(H, fitted[:, :2])
    fitted[:, 2] += np.log(scale)
    fallback = smooth(raw, fps, .11)
    return fallback*(1-safe[:,None])+fitted*safe[:,None], {
        'background_used': True, 'supported_fraction': float(np.mean(safe>.5)),
        'method': 'background-relative actor residual; zero-phase; measured image motion restored'}

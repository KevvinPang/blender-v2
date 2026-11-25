"""Offline, time-based trajectory fitting with bounded memory and no causal lag."""
import numpy as np
from mathutils import Quaternion, Vector


def pentadiagonal(diagonal, lower1, lower2, rhs):
    n=len(diagonal)
    d=np.empty(n);a=np.zeros(n);b=np.zeros(n)
    for i in range(n):
        if i>=2:b[i]=lower2[i]/d[i-2]
        if i>=1:a[i]=(lower1[i]-b[i]*a[i-1])/d[i-1]
        d[i]=np.sqrt(max(1e-12,diagonal[i]-a[i]**2-b[i]**2))
    result=np.array(rhs,dtype=float,copy=True)
    for i in range(n):
        if i>=1:result[i]-=a[i]*result[i-1]
        if i>=2:result[i]-=b[i]*result[i-2]
        result[i]/=d[i]
    for i in range(n-1,-1,-1):
        if i+1<n:result[i]-=a[i+1]*result[i+1]
        if i+2<n:result[i]-=b[i+2]*result[i+2]
        result[i]/=d[i]
    return result


def trajectory(times, values, seconds, weights=None, robust=None):
    """Minimize observed displacement plus physical acceleration over the whole take."""
    t=np.asarray(times,dtype=float)
    y=np.asarray(values,dtype=float)
    if y.ndim == 0 or len(y) != len(t) or not np.isfinite(seconds) or seconds < 0:
        raise ValueError('Invalid temporal trajectory shape or duration')
    if not np.isfinite(t).all() or not np.isfinite(y).all() or np.any(np.diff(t)<=0):
        raise ValueError('Invalid temporal trajectory')
    if len(t)<3 or seconds<=0:return y.copy()
    base=np.ones(len(t)) if weights is None else np.maximum(np.asarray(weights,dtype=float),0.)
    if base.shape != t.shape or not np.isfinite(base).all():
        raise ValueError('Invalid temporal weights')
    cell=np.r_[t[1]-t[0],(t[2:]-t[:-2])*.5,t[-1]-t[-2]]
    cell/=np.median(cell)
    weight=base.copy()
    result=y.copy()
    for iteration in range(3 if robust else 1):
        d=np.maximum(weight*cell,1e-7)
        rhs=y*d.reshape((-1,)+(1,)*(y.ndim-1))
        lo1=np.zeros(len(t));lo2=np.zeros(len(t))
        for i in range(1,len(t)-1):
            left,right=t[i]-t[i-1],t[i+1]-t[i]
            c=2/(left+right)*np.array([1/left,-1/left-1/right,1/right])
            penalty=seconds**4*cell[i]
            d[i-1:i+2]+=penalty*c*c
            lo1[i]+=penalty*c[0]*c[1]
            lo1[i+1]+=penalty*c[1]*c[2]
            lo2[i+1]+=penalty*c[0]*c[2]
        result=pentadiagonal(d,lo1,lo2,rhs)
        if robust:
            error=np.linalg.norm((result-y).reshape(len(t),-1),axis=1)
            weight=base*np.minimum(1.,robust/np.maximum(error,1e-10))
    return result


def quaternions(times, values, seconds, weights=None):
    rotations=[Quaternion(v).normalized() for v in values]
    if not rotations:return []
    for i in range(1,len(rotations)):
        if rotations[i].dot(rotations[i-1])<0:rotations[i].negate()
    # Unwrap a local chart; wide rotations use sign-continuous quaternion components.
    reference=rotations[0]
    vectors=[]
    for q in rotations:
        delta=reference.rotation_difference(q)
        if delta.w<0:delta.negate()
        axis,angle=delta.to_axis_angle()
        vectors.append(np.array(axis)*angle)
    if max(np.linalg.norm(v) for v in vectors)>2.8:
        filtered=trajectory(times,[list(q) for q in rotations],seconds,weights)
        return [Quaternion(tuple(v)).normalized() for v in filtered]
    filtered=trajectory(times,vectors,seconds,weights,robust=np.radians(10))
    result=[]
    for v in filtered:
        angle=float(np.linalg.norm(v))
        result.append(reference@Quaternion(Vector(v/angle),angle) if angle>1e-10 else reference.copy())
    return result


def bridge_missing(times, values, observed, max_gap=.8, hold=.2, fade=.45):
    """Carry a correction through brief losses; release long losses continuously.

    Measurement validity and correction lifetime are different. In particular,
    dropping a face measurement must not reset an already applied camera/body
    correction. Interpolate bounded gaps between actual observations; otherwise
    hold the last observation briefly and smoothly release it to zero.
    """
    t=np.asarray(times,dtype=float);result=np.asarray(values,dtype=float).copy()
    valid=np.flatnonzero(observed)
    if not len(valid):return np.zeros_like(result)
    def ease(v):
        v=float(np.clip(v,0,1));return v*v*(3-2*v)
    for i in np.flatnonzero(~np.asarray(observed,dtype=bool)):
        k=int(np.searchsorted(valid,i));left=valid[k-1] if k else None
        right=valid[k] if k<len(valid) else None
        if left is not None and right is not None and t[right]-t[left]<=max_gap:
            w=(t[i]-t[left])/(t[right]-t[left])
            result[i]=result[left]*(1-w)+result[right]*w
        else:
            result[i]=0
            if left is not None:
                result[i]+=result[left]*(1-ease((t[i]-t[left]-hold)/fade))
            if right is not None:
                # Reacquisition ramps in without a one-frame activation jump.
                result[i]+=result[right]*(1-ease((t[right]-t[i])/.18))
    return result

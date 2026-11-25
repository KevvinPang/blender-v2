"""Suggest gesture episodes from local joint motion; no semantic gesture labels."""
import numpy as np


def angle(a,b):
    return 2*np.degrees(np.arccos(np.clip(np.abs(np.sum(a*b,axis=-1)),0,1)))


def smooth(values,radius):
    if radius<1:return values.copy()
    return np.convolve(np.pad(values,(radius,radius),mode='edge'),np.ones(2*radius+1)/(2*radius+1),mode='valid')


def episodes(quaternions,fps=30,visibility=None):
    """Arm, elbow, wrist, then finger rotations; frames returned as zero-based inclusive.

    A hold at the top of a gesture is part of that gesture. Split at sustained
    return to the reference pose, not at every low-speed frame or reversal.
    The initial relaxed pose is an estimate; users can correct suggested bounds.
    """
    q=np.asarray(quaternions,dtype=float)
    if q.ndim!=3 or q.shape[-1]!=4 or len(q)<3:return []
    if not np.isfinite(q).all():raise ValueError('动作旋转数据包含无效值')
    q=q/np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-9)
    count=len(q);radius=max(1,round(.08*fps))
    # Use a short initial window so an early raised-hand hold cannot redefine rest.
    reference=q[:min(count,max(1,round(.12*fps)))].copy()
    reference=np.where(np.sum(reference*q[:1],axis=-1,keepdims=True)<0,-reference,reference).mean(axis=0)
    reference/=np.maximum(np.linalg.norm(reference,axis=-1,keepdims=True),1e-9)
    distance=angle(q,reference)
    thresholds=np.array([10.,13.,24.]+[32.]*(q.shape[1]-3))[:q.shape[1]]
    activity=smooth(np.max(distance/thresholds,axis=1),radius)
    # Small natural drift and capture jitter must not create dozens of clips.
    neutral=activity<.80
    if visibility is not None:
        visible=np.asarray(visibility,dtype=bool)
        if visible.shape!=(count,):raise ValueError('手部可见性与动作帧数不一致')
        # A hand leaving the picture with a relaxed arm is a useful boundary.
        # Occlusion while an arm stays raised (e.g. scratching) is not.
        neutral|=(~visible)&(distance[:,0]<14)&(distance[:,1]<20)
    quiet=max(3,round(.25*fps));padding=max(2,round(.10*fps))
    boundaries=[];i=0
    while i<count:
        if not neutral[i]:i+=1;continue
        j=i+1
        while j<count and neutral[j]:j+=1
        if j-i>=quiet:boundaries.append((i,j-1))
        i=j
    candidates=[];left=0
    for a,b in boundaries:
        if a>left:candidates.append((max(0,left-padding),min(count-1,a+padding)))
        left=b+1
    if left<count:candidates.append((max(0,left-padding),count-1))
    result=[]
    for a,b in candidates:
        if b-a<max(3,round(.2*fps)) or float(activity[a:b+1].max())<1.:continue
        if result and a<=result[-1][1]:result[-1]=(result[-1][0],b)
        else:result.append((a,b))
    # A completely unfamiliar initial pose can remain active for a long time.
    # Do not invent semantic subdivisions: return one editable suggestion.
    return result

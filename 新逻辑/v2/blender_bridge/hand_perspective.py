"""Separate palm magnification from translation across an image boundary."""
import numpy as np
from hand_visibility import inside, sample, PALM


def palm_step(uv, field, reverse, size):
    # Include finger bases when the wrist is cropped. Use pixel aspect for angles.
    ids=[0,1,2,5,6,9,10,13,14,17,18]
    if inside(uv[ids]).sum()<8:
        ids=list(range(21))
    points=uv[ids]
    flow=sample(field,points)
    visible=inside(points)&inside(points+flow)
    good=visible&(np.linalg.norm((flow+sample(reverse,points+flow))*size,axis=1)<.014*min(size))
    if good.sum()<5:return None
    x=points[good]*size
    y=(points+flow)[good]*size
    def fit(x,y):
        center=x.mean(0); target=y.mean(0)
        a=x-center;b=y-target
        power=np.sum(a*a)
        real=np.sum(a*b)/max(power,1e-9)
        imag=np.sum(a[:,0]*b[:,1]-a[:,1]*b[:,0])/max(power,1e-9)
        matrix=np.array([[real,imag],[-imag,real]])
        return matrix,target-center@matrix
    matrix,offset=fit(x,y)
    residual=np.linalg.norm(x@matrix+offset-y,axis=1)
    keep=residual<=max(.004*min(size),2*np.median(residual))
    if keep.sum()<5:return None
    x,y=x[keep],y[keep]
    if np.linalg.norm(np.ptp(x,axis=0))<.065*min(size):return None
    matrix,offset=fit(x,y)
    gain=np.linalg.norm(matrix[0])
    if not .88<gain<1.14 or np.median(np.linalg.norm(x@matrix+offset-y,axis=1))>.008*min(size):return None
    return (uv*size@matrix+offset)/size,np.log(gain)


def resolve_perspective(world,uv,valid,raw,face,forward,backward,size):
    count=len(uv)
    recovered=np.zeros(count,bool)
    scale=np.ones(count)
    if forward is None or backward is None or face is None or face.get('landmarks') is None:
        return recovered,scale
    eyes=face['landmarks'][:,[33,263],:2]*size
    span=np.linalg.norm(eyes[:,0]-eyes[:,1],axis=1)
    reliable=face['valid'] & np.isfinite(span) & (span>12)
    gain=np.full(count-1,np.nan)
    for i in np.flatnonzero(valid[:-1]&valid[1:]&reliable[:-1]&reliable[1:]):
        result=palm_step(uv[i],forward[i],backward[i],size)
        if result is None:continue
        predicted,log_scale=result
        if np.median(np.linalg.norm((predicted-uv[i+1])*size,axis=1))>.08*min(size):continue
        # A rapid palm turn alone can shorten otherwise stable hand bones.
        pair=world[i:i+2]
        normal=np.cross(pair[:,5]-pair[:,0],pair[:,17]-pair[:,0])
        normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-9)
        if normal[0]@normal[1]<.88:continue
        gain[i]=log_scale-np.log(span[i+1]/span[i])
    # Preserve the trusted observed endpoints. Only promote missing samples
    # where pixel transport also agrees with a separate weak hand detection.
    good=np.flatnonzero(np.isfinite(gain))
    for run in np.split(good,np.flatnonzero(np.diff(good)>1)+1):
        if len(run)<3:continue
        path=np.r_[0.,np.cumsum(gain[run])]
        if np.ptp(path)<.055:continue
        start,end=run[0],run[-1]+1
        curve={i:v for i,v in zip(range(start,end+1),path)}
        for anchor,step in ((start,-1),(end,1)):
            points=uv[anchor].copy();value=curve[anchor];pending=[];witness=None
            for dt in range(1,23):
                i=anchor+dt*step;k=min(i,i-step)
                if not 0<=i<count or valid[i] or not reliable[i] or not reliable[i-step]:break
                field,reverse=(forward[k],backward[k]) if step>0 else (backward[k],forward[k])
                result=palm_step(points,field,reverse,size)
                if result is None:break
                points,log_step=result
                if inside(points).sum()<8:break
                value+=log_step-np.log(span[i]/span[i-step])
                pending.append((i,points.copy(),value))
                if np.isfinite(raw[i]).all():
                    # A partial detector box can confirm occupancy even when
                    # its finger identities are inconsistent with optical flow.
                    boxes=np.clip([points,raw[i]],0,1)
                    lo=boxes.min(axis=1);hi=boxes.max(axis=1)
                    area=np.prod(hi-lo,axis=1)
                    overlap=np.prod(np.maximum(0,hi.min(0)-lo.max(0)))
                    if overlap/max(area.sum()-overlap,1e-8)<.38:break
                    witness=i
                if witness is not None and abs(i-witness)<=9:
                    for j,pts,v in pending:
                        uv[j]=pts;world[j]=world[anchor];valid[j]=recovered[j]=True;curve[j]=v
                    pending=[]
        ids=np.array(sorted(curve))
        log=np.array([curve[i] for i in ids])
        # Keep the size of the middle observed pose; transfer relative change.
        local=np.exp(np.clip(log-np.median(path),-.42,.42))
        scale[ids]=local
    return recovered,scale

"""Prepare one camera track consumed by Blender, with background diagnostics.

Run outside Blender (OpenCV/SciPy). Keep the raw OpenCV extrinsic convention;
never feed Blender-world matrices to a consumer expecting RT_vggt.
"""
import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_motion import smooth,temporal_support


def foreground_mask(shape, face, pose):
    h,w = shape
    mask = np.full((h,w),255,np.uint8)
    if face is None or not np.isfinite(face).all():
        return np.zeros((h,w),np.uint8)  # no invented background on missing face
    xy = face[:,:2]*[w,h]
    lo,hi = xy.min(0),xy.max(0)
    size = hi-lo
    # Include hair above face, neck below it, and a safety margin at the ears.
    cv2.rectangle(mask,tuple(np.floor(lo-size*[.35,.85]).astype(int)),
                  tuple(np.ceil(hi+size*[.35,.15]).astype(int)),0,-1)
    if pose is not None:
        # Conservative upper-body polygon; extended below frame for close crops.
        shoulders = pose[[5,6]]
        left,right = shoulders[np.argsort(shoulders[:,0])]
        margin = max(25.,np.linalg.norm(right-left)*.12)
        poly = np.array([[lo[0]-margin,hi[1]],[hi[0]+margin,hi[1]],
                         right+[margin,-margin], [right[0]+margin,h],
                         [left[0]-margin,h],left+[-margin,-margin]],np.int32)
        cv2.fillConvexPoly(mask,cv2.convexHull(poly),0)
        for ids in [(5,7,62),(6,8,41)]:
            for a,b in zip(ids[:-1],ids[1:]):
                cv2.line(mask,tuple(pose[a].astype(int)),tuple(pose[b].astype(int)),0,int(2*margin))
        for ids in (range(20,41),range(41,62)):
            pts=pose[list(ids)]
            if np.isfinite(pts).all():
                hull=cv2.convexHull(pts.astype(np.int32));cv2.fillConvexPoly(mask,hull,0)
                cv2.polylines(mask,[hull],True,0,int(margin))
    else:
        mask[int(max(0,hi[1])):]=0
    return mask


def track_background(video, K, face_path, observations_path, output):
    T=len(K);cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS)
    faces=np.load(face_path);obs=np.load(observations_path)
    ids=obs['frame_indices'];kp=obs['keypoints_2d']
    poses=np.stack([np.interp(np.arange(T),ids,kp[:,j,d]) for j in range(kp.shape[1]) for d in range(2)],axis=1).reshape(T,-1,2)
    transforms=np.repeat(np.eye(3)[None],T,axis=0);conf=np.zeros(T);rows=[]
    ok,im=cap.read()
    if not ok:raise ValueError('Cannot read source video')
    h,w=im.shape[:2];prev=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    for t in range(1,T):
        ok,im=cap.read()
        if not ok:raise ValueError(f'Video ended at {t}, expected {T}')
        cur=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        mask=foreground_mask((h,w),faces['landmarks'][t-1] if faces['valid'][t-1] else None,poses[t-1])
        mask2=foreground_mask((h,w),faces['landmarks'][t] if faces['valid'][t] else None,poses[t])
        p=cv2.goodFeaturesToTrack(prev,500,.006,8,mask=mask,blockSize=5)
        A=None;n=0;error=99.;spread=0.
        if p is not None and len(p)>=12:
            q,st,_=cv2.calcOpticalFlowPyrLK(prev,cur,p,None,winSize=(21,21),maxLevel=3)
            back,sb,_=cv2.calcOpticalFlowPyrLK(cur,prev,q,None,winSize=(21,21),maxLevel=3)
            good=st[:,0].astype(bool)&sb[:,0].astype(bool)&(np.linalg.norm(back[:,0]-p[:,0],axis=1)<.6)
            qi=np.rint(q[:,0]).astype(int);inside=(qi[:,0]>=0)&(qi[:,0]<w)&(qi[:,1]>=0)&(qi[:,1]<h)
            good &= inside
            good[inside] &= mask2[qi[inside,1],qi[inside,0]]>0
            a,b=p[:,0][good],q[:,0][good]
            if len(a)>=12:
                A,keep=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=1.2,maxIters=3000,confidence=.995)
                if A is not None:
                    a,b=a[keep[:,0]>0],b[keep[:,0]>0];n=len(a)
                    error=float(np.median(np.linalg.norm(np.c_[a,np.ones(n)]@A.T-b,axis=1)))
                    spread=float(np.ptp(a[:,0])/w)
                    scale=np.hypot(A[0,0],A[1,0])
                    if n<18 or spread<.45 or error>.8 or abs(np.log(scale))>.035 or np.linalg.norm(A[:,2])>w*.12:A=None
        step=np.eye(3)
        if A is not None:
            step[:2]=A;conf[t]=min(1.,n/40)*min(1.,spread/.65)*max(0.,1-error/1.5)
        transforms[t]=step@transforms[t-1]
        rows.append(dict(frame=t+1,inliers=n,residual_px=error,coverage=spread,confidence=float(conf[t])))
        if t in (81,327,441,620):
            overlay=im.copy();overlay[mask2==0]=(overlay[mask2==0]*.25).astype(np.uint8)
            cv2.imencode('.png',cv2.resize(overlay,(360,640)))[1].tofile(output.parent/f'background_mask_{t+1:04d}.png')
        if t%120==0:print(f'V2_PROGRESS background {t}/{T}',flush=True)
        prev=cur
    cap.release();conf[0]=conf[1] if T>1 else 0
    np.savez_compressed(output,image_from_reference=transforms,confidence=conf,fps=fps,image_size_hw=[h,w])
    return transforms,conf,fps,rows


def regularize_camera(RT,K,background,confidence,fps):
    c2w=np.linalg.inv(RT);out=c2w.copy();Kout=K.copy()
    # Zero-phase bounded regularization; unlike the old EMA, no systematic lag.
    ref=Rotation.from_matrix(c2w[0,:3,:3]);rv=(ref.inv()*Rotation.from_matrix(c2w[:,:3,:3])).as_rotvec()
    slow=smooth(rv,fps,.10)
    if background is not None:
        # Near-field translation and rotation cannot be uniquely separated.
        # Use only small, supported high-frequency background ray motion;
        # retain the model's low-frequency orientation and relative translation.
        center=np.array([K[0,0,2],K[0,1,2],1.])
        xy=np.einsum('tij,j->ti',background,center);xy=xy[:,:2]/xy[:,2:]
        delta=xy-center[:2]
        proxy=np.c_[delta[:,1]/K[:,1,1],-delta[:,0]/K[:,0,0],-np.unwrap(np.arctan2(background[:,1,0],background[:,0,0]))]
        high=proxy-smooth(proxy,fps,.10)
        good=temporal_support(confidence,fps)
        slow+=high*good[:,None]
    change=slow-rv;norm=np.linalg.norm(change,axis=1)
    change*=np.minimum(1,np.radians(.7)/np.maximum(norm,1e-12))[:,None]
    out[:,:3,:3]=(ref*Rotation.from_rotvec(rv+change)).as_matrix()
    # Do not add the same rapid translation again after using background rays.
    out[:,:3,3]=smooth(c2w[:,:3,3],fps,.10)
    focal=np.sqrt(K[:,0,0]*K[:,1,1]);logf=np.log(focal)
    fixed=logf+np.clip(smooth(logf,fps,.18)-logf,-.025,.025)
    Kout[:,0,0]=Kout[:,1,1]=np.exp(fixed)
    for i in (0,1):Kout[:,i,2]=smooth(K[:,i,2],fps,.18)
    return np.linalg.inv(out),Kout,{'max_rotation_correction_deg':float(np.degrees(np.linalg.norm(change,axis=1)).max()),
        'max_focal_correction_fraction':float(np.max(abs(np.exp(fixed-logf)-1))),
        'translation_scale':'relative; high-frequency background ray motion is an approximation'}


def prepare(raw_path,video,face,observations,output,background_output):
    raw=np.load(raw_path);RT=np.array(raw['RT_vggt'] if 'RT_vggt' in raw.files else raw['RT'],float);K=np.array(raw['K'],float)
    H,confidence,fps,rows=track_background(video,K,face,observations,background_output)
    rt,k,report=regularize_camera(RT,K,H,confidence,fps)
    # Write a minimal unambiguous schema, never stale alternative RT keys.
    np.savez_compressed(output,RT_vggt=rt,K=k,image_size_hw=raw['image_size_hw'],
                        camera_mode='dynamic',source_convention='opencv_cam_from_world',
                        joint_camera_version=1,background_motion=str(background_output.resolve()))
    report.update(background_supported_fraction=float(np.mean(confidence>.5)),frames=len(K),fps=fps,
                  source=str(raw_path),output=str(output),background_pairs=rows)
    output.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print('V2_PROGRESS joint camera ready',json.dumps({k:v for k,v in report.items() if k!='background_pairs'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw');p.add_argument('video');p.add_argument('face');p.add_argument('observations');p.add_argument('output');p.add_argument('background')
    a=p.parse_args();prepare(*map(Path,[a.raw,a.video,a.face,a.observations,a.output,a.background]))

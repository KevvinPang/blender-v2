"""Track individual landmarks through short border transitions in the source video."""
from pathlib import Path
import sys
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'source'))
from cotracker.predictor import CoTrackerPredictor

PALM = [0, 5, 9, 13, 17]
TIPS = [4, 8, 12, 16, 20]


def sequences(xy, scores):
    active = []
    done = []
    for t in range(len(xy)):
        hands = set(np.flatnonzero((scores[t] >= .55) & np.isfinite(xy[t]).all(axis=(1, 2))))
        matches = []
        for j, seq in enumerate(active):
            prev = xy[seq[-1]]
            for h in hands:
                dist = np.median(np.linalg.norm(xy[t,h,PALM]-prev[PALM], axis=1))
                if dist < .30:
                    matches.append((dist, j, h))
        used = set()
        for _, j, h in sorted(matches):
            if j not in used and h in hands:
                active[j].append((t,h))
                hands.remove(h)
                used.add(j)
        done.extend(seq for j,seq in enumerate(active) if j not in used)
        active = [seq for j,seq in enumerate(active) if j in used] + [[(t,h)] for h in sorted(hands)]
    return done+active


def tip_cap(lab, p, axis, radius, reference):
    # Search for an exposed end and a flanking edge at the scale of this finger.
    side = np.array([-axis[1], axis[0]])
    height,width = lab.shape[:2]
    span = int(3*radius)
    x,y = np.rint(p).astype(int)
    xs = np.arange(max(0,x-span), min(width,x+span+1),2)
    ys = np.arange(max(0,y-span), min(height,y+span+1),2)
    if not len(xs) or not len(ys): return None
    xx,yy = np.meshgrid(xs,ys)
    pts = np.column_stack((xx.ravel(),yy.ravel()))
    def pixels(a,b):
        pos = np.rint(pts+(a*axis+b*side)*radius).astype(int)
        good = ((pos>=0)&(pos<[width,height])).all(1)
        return lab[pos[:,1].clip(0,height-1),pos[:,0].clip(0,width-1)],good
    def difference(a,b):
        return np.linalg.norm((a-b)*[.5,1.5,1],axis=1)
    inner,ok = pixels(-.9,0)
    outer,visible = pixels(.65,0)
    flank1,on1 = pixels(-1.6,1.5)
    flank2,on2 = pixels(-1.6,-1.5)
    center,on3 = pixels(-1.6,0)
    cap = difference(inner,outer)
    flank = np.maximum(on1*difference(center,flank1),on2*difference(center,flank2))
    accept = ok & visible & on3 & (difference(inner,reference)<30) & (cap>19) & (flank>17)
    if not accept.any():return None
    score = np.minimum(cap,55)+.6*np.minimum(flank,55)-.45*np.linalg.norm(pts-p,axis=1)
    return pts[np.argmax(np.where(accept,score,-np.inf))]


def estimate(video_path, observations, output):
    with np.load(observations) as data:
        xy=data['normalized_landmarks'][...,:2]
        runs=sequences(xy,data['scores'])
    cap=cv2.VideoCapture(str(video_path))
    size=np.array([cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT)])
    model=None
    rows=[]
    try:
        for run in runs:
            if len(run)<5:continue
            for chain,step in ((run,-1),(run[::-1],1)):
                boundary=chain[0][0]
                if np.minimum(xy[chain[0]],1-xy[chain[0]]).min()>.105:continue
                # A completely or mostly visible pose identifies the digit in both directions.
                def quality(item):
                    p=xy[item]
                    return np.minimum(p,1-p).min(1)
                seed=next((j for j,item in enumerate(chain[:7]) if j>=2 and (quality(item)>.035).sum()>=19),None)
                if seed is None or seed+1>=len(chain):continue
                anchors=chain[seed:seed+2]
                end=int(np.clip(boundary+step*7,0,len(xy)-1))
                lo,hi=sorted((anchors[1][0],end))
                if hi-lo<2:continue
                cap.set(cv2.CAP_PROP_POS_FRAMES,lo)
                frames=[]
                for t in range(lo,hi+1):
                    ok,im=cap.read()
                    if not ok:raise ValueError(f'Missing border tracking frame {t+1}')
                    frames.append(im)
                if model is None:
                    model=CoTrackerPredictor(checkpoint=str(ROOT/'models/cotracker3_offline.pth')).cuda().eval()
                inputs=np.stack([cv2.resize(im,(360,round(360*size[1]/size[0])))[:,:,::-1] for im in frames])
                small=np.array(inputs.shape[1:3][::-1])
                video=torch.from_numpy(inputs).permute(0,3,1,2)[None].cuda().float()
                query=np.concatenate([np.column_stack((np.full(21,t-lo),xy[t,h]*small)) for t,h in anchors])
                with torch.inference_mode():
                    pred,vis=model(video,queries=torch.as_tensor(query,device='cuda',dtype=torch.float32)[None],backward_tracking=True)
                pred=pred[0].cpu().numpy().reshape(-1,2,21,2)/small
                vis=vis[0].cpu().numpy().reshape(-1,2,21)
                a,h=anchors[0]
                prior=xy[a,h]
                anchor_lab=cv2.cvtColor(frames[a-lo],cv2.COLOR_BGR2Lab).astype(float)
                alive=np.ones(21,bool)
                for t in range(a+step,end+step,step):
                    i=t-lo
                    p=pred[i].mean(0)
                    agree=np.linalg.norm((pred[i,0]-pred[i,1])*size,axis=1)<.065*min(size)
                    in_frame=((p>=0)&(p<=1)).all(1)
                    supported=alive & in_frame & agree & vis[i].all(0)
                    # A fingertip seen projecting out of the frame by the detector
                    # cannot be reacquired on the background by contour alone.
                    sample=next((slot for f,slot in chain[:8] if f==t),None)
                    if sample is not None:
                        raw=xy[t,sample]
                        alive &= np.minimum(raw,1-raw).min(1)>-.025
                    supported &= alive
                    if abs(t-a)>2:
                        moved=np.linalg.norm((p-prior)*size,axis=1)
                        initial=np.linalg.norm((prior-xy[anchors[1]])*size,axis=1)
                        supported &= (moved>np.minimum(initial,.04*min(size))*.45) | (initial<.012*min(size))
                    lab=None
                    for tip in TIPS:
                        if not alive[tip] or not in_frame[tip] or not agree[tip]:continue
                        v=(p[tip]-p[tip-2])*size
                        length=np.linalg.norm(v)
                        if length<.06*min(size):continue
                        if lab is None:lab=cv2.cvtColor(frames[i],cv2.COLOR_BGR2Lab).astype(float)
                        center=np.rint((prior[tip]*.8+prior[tip-1]*.2)*size).astype(int)
                        center=np.clip(center,0,size.astype(int)-1)
                        found=tip_cap(lab,p[tip]*size,v/length,np.clip(.13*length,4,.035*min(size)),anchor_lab[center[1],center[0]])
                        if found is not None and (supported[tip] or vis[i,:,tip-1].any() or supported.sum()>=3):
                            p[tip]=found/size
                            supported[tip]=True
                    alive &= in_frame
                    if supported.any():
                        rows.append((t,a,h,p.astype('f'),supported))
                print(f'V2_PROGRESS 手部边缘追踪 {lo+1}-{hi+1}',flush=True)
    finally:
        cap.release()
    np.savez_compressed(output,frames=np.array([r[0] for r in rows],int),
                        anchors=np.array([r[1] for r in rows],int),slots=np.array([r[2] for r in rows],int),
                        xy=np.array([r[3] for r in rows],np.float32).reshape(-1,21,2),
                        support=np.array([r[4] for r in rows],bool).reshape(-1,21))

if __name__=='__main__':
    estimate(*map(Path,sys.argv[1:]))

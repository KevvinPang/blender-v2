"""Video-backed entry observations shared by the worker and final arm solver."""
import hashlib,json
from pathlib import Path
import numpy as np

VERSION=2

def signature(data):
    video=Path(data['video']);stat=video.stat() if video.is_file() else None
    rows=[(r['source_frame'],r['time']) for r in data['frames']]
    observations=data['screen']['frames']
    payload=[VERSION,str(video.resolve()),[stat.st_size,stat.st_mtime_ns] if stat else None,rows,
             [r.get('hands',{}) for r in observations],data['screen'].get('wilor_hands',[])]
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()

def load_cache(data,path):
    cache=Path(path).with_name('hand_entry_v1.json')
    if not cache.is_file():return False
    record=json.loads(cache.read_text(encoding='utf8'))
    if record.get('input_signature')!=signature(data):return False
    data['screen']['hand_entry']=record
    return True

def usable(hand):
    if not hand:return False
    p=np.asarray(hand.get('points',[]),float);mask=np.asarray(hand.get('in_frame',[True]*21))
    return p.shape==(21,2) and np.isfinite(p).all() and mask.shape==(21,) and mask.sum()>=4 and np.linalg.norm(np.ptp(p,axis=0))>=.015

def hand_at(screen,i,side):
    primary=screen['frames'][i].get('hands',{}).get(side)
    if usable(primary):return primary
    fallback=screen.get('wilor_hands',[])
    return fallback[i].get(side) if len(fallback)==len(screen['frames']) else None

def long_entries(data,side):
    times=np.array([r['time'] for r in data['frames']]);active=np.array([usable(hand_at(data['screen'],i,side)) for i in range(len(times))])
    changes=np.diff(np.r_[False,active,False].astype(int))
    for a,b in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)):
        if b-a<3:active[a:b]=False
    # Verified silhouettes keep a visible episode alive through landmark loss.
    masks=data['screen'].get('punch_tracking',{}).get('frames',[])
    silhouette=np.array([bool(r.get('hands',{}).get(side,{}).get('silhouette')) for r in masks]) if len(masks)==len(times) else np.zeros(len(times),bool)
    valid=np.flatnonzero(active)
    starts=[];last=None
    for i in valid:
        if last is None or times[i]-times[last]>.30 and not (i-last>1 and silhouette[last+1:i].mean()>.6):starts.append(int(i))
        last=i
    return starts

def incoming_edge(data,side,start,tracked=None):
    """Intersect the backward VIDEO velocity with the image, not nearest palm."""
    aspect=data['screen']['aspect'];times=np.array([r['time'] for r in data['frames']]);samples=[]
    for i in range(start,min(len(times),start+5)):
        h=hand_at(data['screen'],i,side)
        if h:samples.append((times[i],np.asarray(h['points'])[[5,9,13,17]].mean(0)))
    if tracked:
        samples=[(times[int(k)],np.asarray(v['points'])[[5,9,13,17]].mean(0)) for k,v in sorted(tracked.items(),key=lambda kv:int(kv[0]))]+samples
        first=np.asarray(tracked[min(tracked,key=int)]['points']);overshoot=[max(0.,-first[:,0].min()),max(0.,first[:,0].max()-aspect),max(0.,-first[:,1].min()),max(0.,first[:,1].max()-1)]
        if max(overshoot)>.012:return int(np.argmax(overshoot)),'tracked-crop-boundary'
    if len(samples)>1:
        t=np.array([v[0] for v in samples]);p=np.array([v[1] for v in samples]);t=t-t.mean()
        speed=np.sum(t[:,None]*(p-p.mean(0)),axis=0)/max(1e-8,float(t@t));center=samples[0][1]
        candidates=[]
        for edge,(axis,boundary,inward) in enumerate(((0,0,1),(0,aspect,-1),(1,0,1),(1,1,-1))):
            if speed[axis]*inward<=.06:continue
            elapsed=(center[axis]-boundary)/speed[axis]
            if elapsed<-.04 or elapsed>.45:continue
            at=center-speed*elapsed;other=1-axis;extent=1 if other else aspect
            if -.12<=at[other]<=extent+.12:candidates.append((max(0.,elapsed),edge))
        if candidates:return min(candidates)[1],'video-trajectory'
    h=hand_at(data['screen'],start,side);p=np.asarray(h['palm'])/np.array([aspect,1.])
    return int(np.argmin([p[0],1-p[0],p[1],1-p[1]])),'edge-proximity-fallback'

def merge_wilor(data,take_path):
    """Reuse actual hand detections when the primary landmark detector misses."""
    import json
    from pathlib import Path
    screen=data.get('screen')
    if not screen:return 0
    cache=Path(take_path).with_name('wilor_predictions.json')
    if not cache.is_file():return 0
    record=json.loads(cache.read_text(encoding='utf8'))
    if Path(record.get('video','')).resolve()!=Path(data['video']).resolve():return 0
    lookup={int(r['frame']):r for r in record.get('frames',[])};rows=[];count=0
    aspect=screen['aspect']
    for sample in data['frames']:
        raw=lookup.get(int(sample['source_frame']),{});height=raw.get('height',0);pairs=[]
        points=sample.get('image_points',[]);hands=raw.get('hands',[])
        if height and len(points)>62:
            for j,hand in enumerate(hands):
                confidence=float(hand.get('confidence',0))
                if confidence<.30:continue
                p=np.asarray(hand.get('joints2d',[]),dtype=float)/height
                if p.shape!=(21,2) or not np.isfinite(p).all():continue
                for side,k in [('L',62),('R',41)]:
                    distance=float(np.linalg.norm(p[0]-np.asarray(points[k][:2])/height))
                    limit=.10 if confidence<.75 else .18
                    if distance<=limit:pairs.append((distance,j,side,p))
        found={};used=set()
        for distance,j,side,p in sorted(pairs,key=lambda v:v[0]):
            if side in found or j in used:continue
            mask=(p[:,0]>=0)&(p[:,0]<=aspect)&(p[:,1]>=0)&(p[:,1]<=1)
            hand={'points':p.tolist(),'in_frame':mask.tolist(),'wrist':p[0].tolist(),
                'palm':p[[0,5,9,13,17]].mean(0).tolist(),'span':float(np.linalg.norm(p[5]-p[17])),
                'source':'wilor-hand-detector','confidence':float(hands[j]['confidence']),
                'confidence_kind':'hand_detection / not joint localization','localization_confidence':None}
            hand['tracking_reliability']=float(np.clip((hands[j]['confidence']-.25)*.8,.08,.48))
            hand['inferred']=hands[j]['confidence']<.75
            hand['orientation_observed']=not hand['inferred']
            if usable(hand):found[side]=hand;used.add(j);count+=1
        rows.append(found)
    screen['wilor_hands']=rows
    return count


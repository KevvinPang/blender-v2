"""Shared image-motion support; rapid motion alone is never confidence."""
import numpy as np


HARD_ARM_RATE = 1200.


def edge_limits(first, last):
    visible = first.get('visible', False) or last.get('visible', False)
    base = 780. if visible else 480.
    # Additional speed needs a temporally corroborated image trajectory.
    fast = max(first.get('fast_motion', 0.), last.get('fast_motion', 0.)) if visible else 0.
    return base + 300.*fast, 660. + 180.*fast


def motion_evidence(data, side):
    from sam_ellen_arm_visibility import screen_hand, hand_landmark_mask
    times = np.array([r['time'] for r in data['frames']], dtype=float)
    n = len(times)
    centers = np.zeros((n, 2)); spans = np.zeros(n); support = np.zeros(n)
    for i in range(n):
        hand = screen_hand(data.get('screen', {}), i, side)
        mask = hand_landmark_mask(hand)
        if not mask.any():
            continue
        points = np.asarray(hand['points'], dtype=float)
        ids = np.array([0, 5, 9, 13, 17]); ids = ids[mask[ids]]
        if len(ids) < 3:
            continue
        centers[i] = np.median(points[ids], axis=0)
        spans[i] = np.median([np.linalg.norm(points[a]-points[b])
                             for j, a in enumerate(ids) for b in ids[j+1:]])
        support[i] = min(1., len(ids)/4.) * min(1., mask.sum()/10.)
    fast = np.zeros(n); confidence = np.zeros(n)
    faces=np.array([max(.01,(r.get('face') or {}).get('width',1.)) for r in data.get('screen',{}).get('frames',[])])
    if len(faces)!=n:faces=np.ones(n)
    sizes=spans/faces
    depth_fast=np.zeros(n);reversals=np.zeros(n,bool)
    for i in range(1, n-1):
        if min(support[i-1:i+2]) < .6:
            continue
        dt = np.diff(times[i-1:i+2])
        if np.any(dt <= 0) or np.max(dt) > .12:
            continue
        v0, v1 = np.diff(centers[i-1:i+2], axis=0)/dt[:, None]
        speed0, speed1 = np.linalg.norm(v0), np.linalg.norm(v1)
        # One-frame outliers reverse immediately. Genuine reversals keep the
        # ordinary safe budget until neighboring frames confirm their motion.
        coherence = float(v0@v1/max(1e-8, speed0*speed1))
        # Genuine reversals have a supported incoming AND outgoing path.
        # A lone outlier lacks these outer edges, so never gains fast support.
        if coherence < -.2 and i>=2 and i+2<n and min(support[i-2:i+3])>=.6:
            before=(centers[i-1]-centers[i-2])/(times[i-1]-times[i-2])
            after=(centers[i+2]-centers[i+1])/(times[i+2]-times[i+1])
            if (min(np.linalg.norm(before),np.linalg.norm(after),speed0,speed1)>.45
                and before@v0/max(1e-8,np.linalg.norm(before)*speed0)>.7
                and after@v1/max(1e-8,np.linalg.norm(after)*speed1)>.7):
                coherence=.85;reversals[i]=True
        ratio = max(spans[i-1:i+2])/max(.005, min(spans[i-1:i+2]))
        confidence[i] = min(support[i-1:i+2])*np.clip((coherence+.1)/.7, 0, 1)*np.clip((2.5-ratio)/1.0, 0, 1)
        fast[i] = confidence[i]*np.clip((min(speed0, speed1)-.45)/1.25, 0, 1)
        # Relative scale is a separate motion cue, including a centred reach.
        # It grants continuity budget, not a punch label or a depth target.
        growth=np.diff(np.log(np.maximum(.001,sizes[i-1:i+2])))/dt
        hands=[screen_hand(data['screen'],j,side) for j in (i-1,i,i+1)]
        shapes=[]
        for hand in hands:
            pts=np.asarray(hand['points'])[ [0,5,9,13,17] ];pts=pts-pts.mean(0)
            shapes.append(pts/max(.005,np.linalg.norm(pts)))
        stable_shape=max(np.linalg.norm(shapes[0]-shapes[1]),np.linalg.norm(shapes[1]-shapes[2]))<.22
        if growth[0]*growth[1]>0 and max(abs(growth))<18 and stable_shape:
            depth_fast[i]=min(support[i-1:i+2])*float(np.clip((min(abs(growth))-1.2)/4.,0,1))
            fast[i]=max(fast[i],depth_fast[i])
        hand=screen_hand(data['screen'],i,side)
        fast[i]*=float(np.clip(hand.get('tracking_reliability',1.)/.65,0,1))
    return [dict(motion_reliability=float(confidence[i]), fast_motion=float(fast[i]),
                 scale_motion=float(depth_fast[i]), supported_reversal=bool(reversals[i]),
                 image_speed=float(np.linalg.norm(centers[i]-centers[i-1])/max(1e-8, times[i]-times[i-1]))
                 if i and min(support[i-1:i+1]) > .6 else 0.) for i in range(n)]

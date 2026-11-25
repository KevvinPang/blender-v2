"""Recover exposed digits in bounded detector gaps from the source images."""
import numpy as np

from hand_visibility import inside
from hand_perspective import palm_step


def exposed_tips(points, size):
    pixels = points * size
    tips = np.array([4, 8, 12, 16, 20])
    spread = np.linalg.norm(pixels[tips, None] - pixels[tips], axis=2)
    np.fill_diagonal(spread, np.inf)
    length = np.linalg.norm(pixels[tips] - pixels[tips - 2], axis=1)
    return tips[inside(points[tips]) & (length > .055 * min(size))
                & (spread.min(axis=1) > .09 * min(size))]


def patch_motion(cv2, previous, current, center, guess, radius):
    size = np.array(previous.shape[::-1])
    c = np.rint(center).astype(int)
    if np.any(c-radius < 0) or np.any(c+radius >= size):
        return None
    patch = previous[c[1]-radius:c[1]+radius+1, c[0]-radius:c[0]+radius+1]
    if patch.std() < 9:
        return None
    g = np.rint(guess).astype(int)
    lo = np.maximum(g-radius*3, 0)
    hi = np.minimum(g+radius*3+1, size)
    if np.any(hi-lo < radius*2+1):
        return None
    score = cv2.matchTemplate(current[lo[1]:hi[1],lo[0]:hi[0]], patch, cv2.TM_CCOEFF_NORMED)
    _, confidence, _, loc = cv2.minMaxLoc(score)
    location = lo+loc+radius
    if confidence < .79:
        for scale in (.85,1.18):
            scaled = cv2.resize(patch,None,fx=scale,fy=scale)
            extent = np.array(scaled.shape[::-1])
            if np.any(hi-lo<extent):continue
            scores = cv2.matchTemplate(current[lo[1]:hi[1],lo[0]:hi[0]],scaled,cv2.TM_CCOEFF_NORMED)
            _,value,_,corner = cv2.minMaxLoc(scores)
            if value>confidence:
                confidence,location = value,lo+corner+(extent-1)/2
    return location if confidence>=.79 else None


def trace_digits(cv2, images, first, anchor, step, uv, forward, backward, size):
    points = uv[anchor].copy()
    tips = exposed_tips(points, size)
    small = np.array(images[0].shape[::-1])
    radius = max(5, round(min(small)*.042))
    result = {}
    for i in range(anchor+step, first-1 if step<0 else first+1, step):
        k = min(i, i-step)
        flow, reverse = (forward[k], backward[k]) if step>0 else (backward[k], forward[k])
        fit = palm_step(points, flow, reverse, size)
        guess = fit[0] if fit is not None else points.copy()
        previous, current = images[i-step-min(anchor,first)], images[i-min(anchor,first)]
        supported = []
        corrections = []
        for tip in tips:
            at = points[tip]*small
            found = patch_motion(cv2,previous,current,at,guess[tip]*small,radius)
            if found is None:
                continue
            back = patch_motion(cv2,current,previous,found,at,radius)
            if back is None or np.linalg.norm(back-at)>2.2:
                continue
            supported.append(tip)
            corrections.append(found/small-guess[tip])
        if not supported:
            break
        tips = supported
        correction = np.median(corrections,axis=0)
        guess += correction
        for tip, delta in zip(tips, corrections):
            guess[tip-2:tip+1] += delta-correction
        points = guess
        result[i] = points.copy(), tips.copy()
    return result


def recover_digits(video, world, uv, valid, observed, forward, backward, size):
    weights = np.zeros((len(valid),21))
    estimated = np.zeros(len(valid),bool)
    if video is None or forward is None or backward is None:
        return weights,estimated
    import cv2
    ids = np.flatnonzero(observed)
    capture = cv2.VideoCapture(str(video))
    try:
        for a,b in zip(ids,ids[1:]):
            if not 3 <= b-a <= 32:
                continue
            anchors = []
            for anchor,step in ((a,1),(b,-1)):
                # Require a cropped or near-border hand and a distinguishable digit.
                if np.min(np.minimum(uv[anchor,0],1-uv[anchor,0]))>.16:
                    continue
                if len(exposed_tips(uv[anchor],size)):
                    anchors.append((anchor,step))
            if not anchors:
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES,int(a))
            images = []
            for i in range(a,b+1):
                ok,frame = capture.read()
                if not ok:break
                small = cv2.resize(frame,None,fx=min(1.,360/min(size)),fy=min(1.,360/min(size)),interpolation=cv2.INTER_AREA)
                images.append(cv2.cvtColor(small,cv2.COLOR_BGR2GRAY))
            if len(images)!=b-a+1:continue
            candidates = {}
            for anchor,step in anchors:
                other = b if step==1 else a
                traced = trace_digits(cv2,images,other,anchor,step,uv,forward,backward,size)
                for i,(points,tips) in traced.items():
                    if a<i<b:
                        candidates.setdefault(i,[]).append((anchor,points,tips))
            for i,choices in candidates.items():
                mix = np.array([1./abs(i-anchor)**2 for anchor,_,_ in choices])
                mix /= mix.sum()
                pts = sum(w*p for w,(_,p,_) in zip(mix,choices))
                pose = sum(w*world[j] for w,(j,_,_) in zip(mix,choices))
                # Ease into the other measured pose at an inconsistent endpoint.
                for endpoint in (a,b):
                    if all(j!=endpoint for j,_,_ in choices):
                        blend = max(0.,1.-abs(i-endpoint)/5.)
                        pts = pts*(1-blend)+uv[endpoint]*blend
                        pose = pose*(1-blend)+world[endpoint]*blend
                uv[i],world[i],valid[i] = pts,pose,True
                for w,(_,_,tips) in zip(mix,choices):
                    weights[i,tips] += w*min(1.,(i-a)/5.,(b-i)/5.)
            if candidates:
                for anchor,near in ((a,min(candidates)),(b,max(candidates))):
                    gap = abs(near-anchor)
                    if not 1<gap<=6:continue
                    visible = weights[near]>0
                    if not inside(uv[anchor,visible]).any():continue
                    distance = np.linalg.norm((uv[anchor,visible]-uv[near,visible])*size,axis=1)
                    if np.median(distance)>.32*min(size):continue
                    for i in range(min(anchor,near)+1,max(anchor,near)):
                        u = abs(i-anchor)/gap
                        uv[i] = (1-u)*uv[anchor]+u*uv[near]
                        world[i] = (1-u)*world[anchor]+u*world[near]
                        valid[i] = estimated[i] = True
                        weights[i] = u*weights[near]
    finally:
        capture.release()
    return weights,estimated


def apply_parts(track, parts, slots, raw_uv):
    """Keep a pose prior for hidden joints and mark only measured image constraints."""
    n=len(slots)
    anchor_for=np.full(n,-1,int)
    track['part_anchor']=anchor_for
    proposals={}
    for row,(t,a,h) in enumerate(zip(parts['frames'],parts['anchors'],parts['slots'])):
        if not (0<=a<n and 0<=t<n) or slots[a]!=h or not track['observed'][a]:continue
        mask=parts['support'][row]
        # A lone track on a stationary background can survive several frames.
        # Position the arm only from a cluster of independently tracked joints.
        if mask.sum()<3:continue
        if np.linalg.norm(np.ptp(parts['xy'][row,mask],axis=0))<.025:continue
        # Preserve reliable complete-hand samples, and never bridge another assigned hand.
        if track['observed'][t] and np.isfinite(raw_uv[t]).all() and np.minimum(raw_uv[t],1-raw_uv[t]).min()>.055:continue
        if t in proposals:
            prev=proposals[t]
            if abs(a-t)>=abs(parts['anchors'][prev]-t):continue
        proposals[t]=row
    prior=track['world'].copy()
    for t,row in proposals.items():
        a=parts['anchors'][row]
        mask=parts['support'][row]
        pts=parts['xy'][row]
        shape=raw_uv[a].copy()
        shape+=np.median(pts[mask]-shape[mask],axis=0)
        shape[mask]=pts[mask]
        track['uv'][t]=shape
        track['world'][t]=prior[a]
        track['digit_weights'][t]=mask
        track['valid'][t]=True
        track['observed'][t]=False
        track['confidence'][t]=min(1.,.30+mask.sum()/16.)
        track['held'][t]=False
        track['edge'][t]=-1
        track['boundary'][t]=np.nan
        anchor_for[t]=a
    return len(proposals)

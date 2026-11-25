"""Reconstruct short hand/face occlusions using two observed endpoints."""
import numpy as np
from hand_visibility import sample, inside, PALM


def unsupported_face_track(uv, face, body_uv, image_size):
    """An isolated face-sized detection needs independent arm or track support."""
    reject = np.zeros(len(uv), bool)
    if face is None or face.get('landmarks') is None or body_uv is None:
        return reject
    visible = np.isfinite(uv).all(axis=(1, 2))
    wrists = body_uv[:, [41, 62]]
    for run in np.split(np.flatnonzero(visible), np.flatnonzero(np.diff(np.flatnonzero(visible)) > 1) + 1):
        if not len(run) or not face['valid'][run].all():
            continue
        points = uv[run] * image_size
        faces = face['landmarks'][run, :, :2] * image_size
        lo, hi = faces.min(axis=1), faces.max(axis=1)
        width = np.linalg.norm(hi-lo, axis=1)
        inside_face = ((points >= lo[:, None]) & (points <= hi[:, None])).all(axis=2)
        if not np.all(inside_face.mean(axis=1) >= .85):
            continue
        if not np.all(np.linalg.norm(np.ptp(points, axis=1), axis=1) < .55*width):
            continue
        # Trust either body wrist in case the anatomical assignment is uncertain.
        distance = np.linalg.norm(wrists[run] - points[:, :1], axis=2)
        if not np.all(distance > width[:, None]):
            continue
        # A real arriving or departing hand may briefly lose its body wrist.
        context = np.r_[max(0,run[0]-6):run[0], run[-1]+1:min(len(uv),run[-1]+7)]
        near = context[visible[context]]
        if len(near):
            centers = uv[near, 0] * image_size
            # Only independent observations outside the face support a contact.
            supported = ((centers < lo[0]) | (centers > hi[0])).any(axis=1)
            ends = points[[0, -1], 0]
            if np.any(supported & (np.linalg.norm(centers[:,None]-ends,axis=2).min(axis=1)<width.mean())):
                continue
        reject[run] = True
    return reject


def recover_contact_pose(world,uv,valid,face,body_wrist,size):
    """Resolve a collapsed palm from the same hand touching the same area."""
    filled=np.zeros(len(valid),bool)
    if face is None or face.get('landmarks') is None or body_wrist is None:
        return filled
    tips=[8,12,16,20]
    eyes=face['landmarks'][:,[33,263],:2]*size
    eye_width=np.linalg.norm(eyes[:,1]-eyes[:,0],axis=1)
    center=np.median(uv[:,tips],axis=1)
    near=np.linalg.norm(center*size-eyes.mean(axis=1),axis=1)<1.45*eye_width
    contact=np.flatnonzero(valid & face['valid'] & near)
    palm_length=np.linalg.norm((uv[:,0]-uv[:,[5,9,13,17]].mean(axis=1))*size,axis=1)
    wrist_error=np.linalg.norm(uv[:,0]*size-body_wrist,axis=1)
    raw=uv.copy();pose=world.copy()
    for run in np.split(contact,np.where(np.diff(contact)>3)[0]+1):
        if len(run)<5:continue
        for i in range(run[0],run[-1]+1):
            if not face['valid'][i] or (valid[i] and not near[i]):continue
            neighbors=run[np.abs(run-i)<=16]
            if len(neighbors)<4:continue
            length=np.percentile(palm_length[neighbors],85)
            anchors=neighbors[(palm_length[neighbors]>=.84*length)&(wrist_error[neighbors]<1.35*eye_width[neighbors])]
            if len(anchors)<2:continue
            error=np.median(wrist_error[anchors])
            if valid[i] and (palm_length[i]>.83*length or wrist_error[i]<error+.22*eye_width[i]):continue
            weights=np.exp(-np.abs(anchors-i)/4.)
            weights/=weights.sum()
            template=np.average(raw[anchors]-center[anchors,None],axis=0,weights=weights)
            anchor=center[i] if valid[i] else np.array([np.interp(i,run,center[run,d]) for d in (0,1)])
            if np.min(np.linalg.norm((center[anchors]-anchor)*size,axis=1))>.45*eye_width[i]:continue
            # Use one anatomical pose, since averaging opposing rotations distorts digits.
            ref=anchors[np.argmax(weights)]
            uv[i]=anchor+template
            world[i]=pose[ref]
            valid[i]=True;filled[i]=True
    return filled


def sweep(points, start, stop, step, fields, reverse):
    rows = {start: points.copy()}
    for t in range(start, stop, step):
        xy = rows[t]
        k = min(t, t+step)
        flow = sample(fields[k], xy)
        good = inside(xy) & (np.linalg.norm(flow + sample(reverse[k], xy+flow), axis=1)<.05)
        if good.sum() >= 4:
            center = np.mean(xy[good],axis=0)
            translation = np.mean(flow[good],axis=0)
            radial = xy[good]-center
            growth = np.clip(np.sum(radial*(flow[good]-translation))/max(np.sum(radial**2),1e-6),-.25,.35)
            rows[t+step] = xy+translation+growth*(xy-center)
        else:
            rows[t+step] = xy.copy()
    return rows


def recover(track, raw, face, forward, backward):
    obs=track['observed']
    scale=np.ones(len(obs))
    if face is None or face.get('landmarks') is None or forward is None or backward is None:
        return scale
    ids=np.flatnonzero(obs)
    extent=np.linalg.norm(np.ptp(raw,axis=1),axis=1)
    for a,b in zip(ids[:-1],ids[1:]):
        if not 2<=b-a-1<=12 or a<6 or b+3>=len(obs):continue
        if np.mean(face['valid'][a+1:b])>.25:continue
        pre=np.arange(max(0,a-10),a-3)
        pre=pre[obs[pre]&face['valid'][pre]]
        if len(pre)<3 or not obs[b:b+3].all():continue
        base=np.median(extent[pre])
        if min(extent[a],extent[b])<max(.22,1.4*base):continue
        f=face['landmarks'][pre[-1],[33,263,1],:2]
        if not np.isfinite(f).all():continue
        eye_span=np.linalg.norm(f[0]-f[1])
        if np.linalg.norm(raw[a,0]-f.mean(0))>1.6*eye_span:continue
        if not np.isfinite(raw[[a,b]]).all():continue
        left=sweep(raw[a],a,b,1,forward,backward)
        right=sweep(raw[b],b,a,-1,backward,forward)
        uv=track['uv']
        for t in range(a,b+1):
            u=(t-a)/(b-a)
            uv[t]=(1-u)*left[t]+u*right[t]
            if a<t<b:
                track['world'][t]=(1-u)*track['world'][a]+u*track['world'][b]
                track['held'][t]=True
            track['valid'][t]=True
            track['confidence'][t]=1.
            track['edge'][t]=-1
            track['boundary'][t]=np.nan
        lo=a
        while lo and obs[lo-1] and a-lo<12 and extent[lo-1]>base*1.05:lo-=1
        hi=b
        while hi+1<len(obs) and obs[hi+1] and hi-b<12 and extent[hi+1]>base*1.05:hi+=1
        for t in range(lo,hi+1):
            amount=np.linalg.norm(np.ptp(uv[t],axis=0))/max(base,1e-6)
            scale[t]=np.clip(amount,1.,6.)
    return scale

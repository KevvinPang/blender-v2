"""Image evidence for approaching fists. Runs in the local vision Python worker."""
import math
import numpy as np

ANCHORS = (5, 6, 9, 10, 13, 14, 17, 18)
VERSION = 2
MASK_VERSION = 2
DETECTION_VERSION = 6


def mask_signature(body):
    """Invalidate masks on video replacement, retiming or observation changes."""
    import hashlib
    import json
    from pathlib import Path
    video = Path(body['video']).resolve()
    stat = video.stat()
    # Derived WiLoR fallback observations do not change the original video-mask
    # inputs. Keep them separate so cache reuse remains stable and repeatable.
    screen = {k:v for k,v in body['screen'].items() if k not in ('punch_tracking','wilor_hands','hand_pose_depth','hand_entry')}
    value = [str(video),stat.st_size,stat.st_mtime_ns,screen,
             [(r['time'],r['source_frame']) for r in body['frames']],MASK_VERSION,DETECTION_VERSION]
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def fist_observation(hand):
    if not hand:
        return None
    # The legacy MediaPipe confidence is a handedness classification score.
    # Only an explicitly identified localization score may reject coordinates.
    localization = hand.get('localization_confidence')
    if localization is not None and (not math.isfinite(localization) or localization < .7):
        return None
    p = np.asarray(hand.get('points', []), dtype=float)
    mask = np.asarray(hand.get('in_frame', [True]*21))
    if p.shape != (21, 2) or not np.isfinite(p).all() or not mask[list(ANCHORS)].all():
        return None
    closed = np.mean([np.linalg.norm(p[j+3]-p[0]) < 1.35*np.linalg.norm(p[j]-p[0])
                      for j in (5, 9, 13, 17)])
    if closed < .75:
        return None
    a = p[list(ANCHORS)]
    center = a.mean(axis=0)
    radius = float(np.sqrt(np.mean(np.sum((a-center)**2, axis=1))))
    if radius < .008:
        return None
    return {'points': p.tolist(), 'center': center.tolist(), 'radius': radius,
            'confidence': float(localization) if localization is not None else 1.,
            'confidence_kind': 'localization' if localization is not None else 'geometric-fist-evidence',
            'source': 'landmarks', 'closed': float(closed)}


def flow_step(before, after, state, time):
    import cv2
    if time-state['seed_time'] > .85 or time-state['time'] > .12:
        return None
    if len(state['features']) < 7:
        return None
    p = state['features'].reshape(-1, 1, 2).astype(np.float32)
    q, good, _ = cv2.calcOpticalFlowPyrLK(before, after, p, None,
        winSize=(31, 31), maxLevel=4)
    if q is None:
        return None
    back, valid, _ = cv2.calcOpticalFlowPyrLK(after, before, q, None,
        winSize=(31, 31), maxLevel=4)
    if back is None:
        return None
    keep = (good.ravel() != 0) & (valid.ravel() != 0) & (np.linalg.norm(back-p, axis=2).ravel() < 1.5)
    if keep.sum() < 7:
        return None
    transform, inliers = cv2.estimateAffinePartial2D(p[keep], q[keep],
        method=cv2.RANSAC, ransacReprojThreshold=2., maxIters=500)
    if transform is None or inliers.mean() < .65:
        return None
    scale = float(np.linalg.norm(transform[:, 0]))
    if not .72 < scale < 1.40:
        return None
    source = np.asarray(state['observation']['points'])*state['height']
    points = np.c_[source, np.ones(21)] @ transform.T / state['height']
    anchors = points[list(ANCHORS)]; center = anchors.mean(0)
    radius = float(np.sqrt(np.mean(np.sum((anchors-center)**2, axis=1))))
    obs = dict(state['observation'], points=points.tolist(), center=center.tolist(), radius=radius,
               source='optical-flow', confidence=float(min(.95, inliers.mean())),
               seed_time=state['seed_time'], flow_inliers=int(inliers.sum()))
    features = q[keep][inliers.ravel() != 0].reshape(-1, 2)
    return dict(state, observation=obs, features=features, time=time)


def extract(body, screen):
    import cv2
    from video_input import open_video, SequentialFrames
    capture = open_video(body['video']); reader = SequentialFrames(capture)
    states, previous, result = {}, None, []
    try:
        for row, observed in zip(body['frames'], screen['frames']):
            ok, bgr = reader.read(row['source_frame'])
            if not ok:
                raise ValueError('Cannot decode punch evidence frame')
            h, w = bgr.shape[:2]
            scale = min(1., 720/max(h, w))
            small = cv2.resize(bgr, None, fx=scale, fy=scale)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            height = gray.shape[0]; hands = {}
            for side in ('L', 'R'):
                direct = fist_observation(observed.get('hands', {}).get(side))
                state = None
                if previous is not None and side in states:
                    state = flow_step(previous, gray, states[side], row['time'])
                if direct:
                    # Re-detection must agree with the moving patch; otherwise the
                    # hand detector may have switched to an occluding face/forearm.
                    agree = (not state or
                             np.linalg.norm(np.array(direct['center'])-state['observation']['center'])
                             < max(.08, direct['radius']*1.5) and
                             .65 < direct['radius']/max(1e-8,state['observation']['radius']) < 1.5)
                    if agree:
                        mask = np.zeros_like(gray)
                        p = (np.asarray(direct['points'])[list(ANCHORS)]*height).astype(np.int32)
                        cv2.fillConvexPoly(mask, cv2.convexHull(p), 255)
                        mask = cv2.dilate(mask, np.ones((9, 9), np.uint8))
                        features = cv2.goodFeaturesToTrack(gray, 100, .005, 3, mask=mask)
                        state = dict(observation=direct, features=features.reshape(-1, 2)
                                     if features is not None else np.empty((0, 2), np.float32),
                                     time=row['time'], seed_time=row['time'], height=height)
                        hands[side] = direct
                if state:
                    hands[side] = state['observation']; states[side] = state
                else:
                    states.pop(side, None)
            result.append({'time': row['time'], 'hands': hands})
            previous = gray
    finally:
        capture.release()
    return {'version': VERSION, 'video': body['video'], 'frames': result}


def _detect_landmarks(screen, side):
    """Require a sustained closed fist and scale growth, relative to head scale."""
    evidence = screen.get('punch_tracking', {})
    if (evidence.get('version') not in (1, VERSION) or evidence.get('video') != screen.get('video')
            or len(evidence.get('frames', [])) != len(screen.get('frames', []))
            or any(abs(a.get('time', -100)-b['time']) > .0001
                   for a,b in zip(evidence.get('frames', []), screen.get('frames', [])))):
        evidence = {}
    frames = evidence.get('frames', screen.get('frames', []))
    if len(frames) != len(screen.get('frames', [])):
        return [], [None]*len(screen.get('frames', []))
    observations = [r.get('hands', {}).get(side) if evidence else fist_observation(r.get('hands', {}).get(side))
                    for r in frames]
    times = np.array([r['time'] for r in frames], dtype=float)
    if len(times) < 4:
        return [], observations
    faces = [(i, r['face'].get('scale', r['face']['width'])) for i, r in enumerate(screen['frames']) if r.get('face')]
    # Missing faces behind the fist are a gap in face evidence, not a smaller head.
    face = np.interp(times, times[[i for i, _ in faces]], [v for _, v in faces]) if faces else np.ones(len(times))
    sizes = np.array([o['radius']/max(.01, face[i]) if o else np.nan for i, o in enumerate(observations)])
    segments = []
    for peak in range(2, len(times)-1):
        if not np.isfinite(sizes[peak]) or observations[peak].get('confidence', 0) < .65:
            continue
        prior = [i for i in range(peak) if .08 <= times[peak]-times[i] <= .7
                 and np.isfinite(sizes[i]) and observations[i]['source'] == 'landmarks']
        if len(prior) < 3:
            continue
        reference = min(prior, key=lambda i: sizes[i])
        ratio = sizes[peak]/sizes[reference]
        if ratio < 1.45 or sizes[peak]-sizes[reference] < .10:
            continue
        # Lowering a fist after a punch rotates its knuckle plane into view and
        # can triple its 2-D extent without approaching the lens.
        if observations[peak]['center'][1]-observations[reference]['center'][1] > .18:
            continue
        visible_face = screen['frames'][peak].get('face')
        if visible_face and 'center' in visible_face and observations[peak]['center'][1] > visible_face['center'][1]+max(.28, visible_face['width']*2.5):
            continue
        # No discontinuous jump or long unobserved bridge can establish an event.
        values = sizes[reference:peak+1]
        if not np.isfinite(values).all():
            continue
        steps = np.abs(np.diff(np.log(values)))
        if np.max(steps) > .35:
            # A fast forward punch can grow sharply just before landmark loss.
            # Accept only bounded jumps corroborated by two enlarged samples;
            # a single bad frame or an unobserved bridge cannot seed tracking.
            if np.max(steps) > .55 or len(values) < 4 or min(values[-2:]) < sizes[reference]*1.45:
                continue
        future = [i for i in range(peak+1, len(times)) if times[i]-times[peak] <= .17 and np.isfinite(sizes[i])]
        if future and max(sizes[future]) > sizes[peak]*1.03:
            continue
        start = reference
        while start > 0 and times[reference]-times[start-1] < .12 and observations[start-1]:
            start -= 1
        end = peak+1
        while end < len(times)-1 and times[end]-times[peak] < .7 and observations[end] and sizes[end] > sizes[reference]*1.15:
            end += 1
        segment = {'side': side, 'start': float(times[start]-times[0]),
                   'peak': float(times[peak]-times[0]), 'release': float(times[peak]-times[0]),
                   'end': float(min(times[-1], times[end]+.12)-times[0]), 'reference': float(times[reference]-times[0]),
                   'reference_index': reference, 'growth': float(ratio)}
        segment['incomplete'] = bool(end < len(times)-1 and observations[end] is None)
        if segments and segment['start'] <= segments[-1]['end']:
            if sizes[peak] > sizes[segments[-1]['peak_index']]:
                segment['start'] = segments[-1]['start']; segments[-1] = dict(segment, peak_index=peak)
        else:
            segments.append(dict(segment, peak_index=peak))
    return segments, observations


def detect(screen, side):
    """Establish a punch from joints, then follow its verified video silhouette.

    Segmentation cannot establish a fist or infer hidden finger articulation.
    It can maintain image position/extent while the joint detector is absent.
    """
    import copy
    tracking = screen.get('punch_tracking', {})
    valid = (tracking.get('version') == VERSION and tracking.get('mask_version') == MASK_VERSION
             and side in tracking.get('completed_sides', [])
             and tracking.get('video') == screen.get('video')
             and len(tracking.get('frames', [])) == len(screen.get('frames', []))
             and all(abs(a.get('time', -100)-b['time']) < .0001
                     for a,b in zip(tracking.get('frames', []), screen.get('frames', []))))
    if not valid:
        if 'mask_version' in tracking:
            raw=dict(screen);raw.pop('punch_tracking',None)
            return _detect_landmarks(raw,side)
        return _detect_landmarks(screen, side)
    base = copy.deepcopy(screen)
    for row in base['punch_tracking']['frames']:
        hand = row.get('hands', {}).get(side)
        if hand and hand.get('points'):
            p = np.asarray(hand['points'], dtype=float)[list(ANCHORS)]
            hand['center'] = p.mean(0).tolist()
            hand['radius'] = float(np.sqrt(np.mean(np.sum((p-p.mean(0))**2, axis=1))))
        elif hand:
            row['hands'].pop(side)
    segments, observations = _detect_landmarks(base, side)
    masks = []
    for row in tracking['frames']:
        obs = row.get('hands', {}).get(side)
        shape = obs.get('silhouette') if obs else None
        good = (shape and len(shape.get('bounds', [])) == 4 and len(shape.get('outline', [])) >= 3
                and np.isfinite(np.asarray(shape['bounds'])).all() and np.isfinite(np.asarray(shape['outline'])).all()
                and np.isfinite(shape.get('area',0)) and shape.get('area', 0) > 0
                and shape.get('mask_probability', 0) >= .7
                and (shape.get('bidirectional_iou') is None or shape['bidirectional_iou'] >= .3))
        masks.append(obs if good else None)
    times = np.array([r['time'] for r in screen['frames']])
    faces=[(i,r['face'].get('scale',r['face']['width']))
           for i,r in enumerate(screen['frames']) if r.get('face')]
    face_sizes=np.interp(times,times[[i for i,_ in faces]],[v for _,v in faces]) if faces else None
    # Landmark loss at maximum foreshortening is common. A weak approach seed
    # is promoted only after independent video masks show sustained growth.
    for seed in tracking.get('approach_candidates',{}).get(side,[]):
        ref=seed['reference_index']
        if ref not in tracking.get('verified_approach_references',{}).get(side,[]):continue
        if not masks[ref]:continue
        base_area=masks[ref]['silhouette']['area'];base_y=masks[ref]['center'][1]
        grown=[]
        for i in range(ref+1,len(times)):
            if times[i]-times[ref]>1.2:break
            m=masks[i]
            # Actual mask growth must exceed common person/camera zoom too.
            # Interpolate brief face occlusion; do not treat missing face as
            # a sudden shrink or discard the observed fist's own growth.
            head_growth=(face_sizes[i]/max(.01,face_sizes[ref]))**2 if face_sizes is not None else 1.
            if (m and m['silhouette']['area']>=base_area*2.1
                    and m['silhouette']['area']/base_area/max(1.,head_growth)>=1.5
                    and m['center'][1]<=base_y+.15):grown.append(i)
        if not any(b==a+1 and c==b+1 for a,b,c in zip(grown,grown[1:],grown[2:])):continue
        peak=max(grown,key=lambda i:masks[i]['silhouette']['area'])
        if any(s['start']<=times[peak]-times[0]<=s['end'] for s in segments):continue
        segments.append(dict(seed,candidate_only=False,peak_index=peak,peak=float(times[peak]-times[0]),
            release=float(times[peak]-times[0]),end=float(min(times[-1],times[peak]+.3)-times[0]),
            growth=float(np.sqrt(masks[peak]['silhouette']['area']/base_area)),
            confirmation='sustained-video-mask-growth'))
    for segment in segments:
        reference, peak = segment['reference_index'], segment['peak_index']
        if not masks[reference]:
            continue
        base_y = masks[reference]['center'][1]
        last, gap, returned = reference, 0, 0
        initial_area=masks[reference]['silhouette']['area'];maximum_area=initial_area
        for i in range(reference+1, len(times)):
            if times[i]-times[reference] > 3.0:
                break
            obs = masks[i]
            gap = 0 if obs else gap+1
            if gap > 2:
                break
            if not obs:
                continue
            # The tracked hand becomes a forearm patch on its way off-screen.
            # End the punch on the return, before that unrelated scale increase.
            if i > peak and obs['center'][1] > base_y+.20:
                break
            last = i
            area=obs['silhouette']['area'];maximum_area=max(maximum_area,area)
            returned=returned+1 if maximum_area>initial_area*2.25 and area<initial_area*1.35 else 0
            if returned>=3:break
        if last <= peak:
            continue
        visible = [i for i in range(reference, last+1) if masks[i]]
        peak = max(visible, key=lambda i: masks[i]['silhouette']['area'])
        segment.update(peak=float(times[peak]-times[0]), release=float(times[peak]-times[0]),
                       peak_index=peak, end=float(min(times[-1], times[last]+.22)-times[0]),
                       incomplete=last < len(times)-1 and masks[last+1] is None,
                       silhouette=True, last_mask_index=last)
        for i in range(reference,len(times)):
            if times[i]-times[0] >= segment['end']:break
            # End tracking when the verified hand has returned. Later masks
            # can be a forearm patch and must not own the release trajectory.
            if i<=last and masks[i]:observations[i] = masks[i]
    # A bounded gap inside a verified track is not a gesture release. Both
    # neighbours are real masks of the same established punch; do not use
    # these interpolated contours to establish a new event.
    observations=bridge_silhouettes(observations,times,segments)
    return segments, observations


def bridge_silhouettes(observations,times,segments=None,max_gap=.105):
    """Interpolate at most two missing contours between measured neighbours."""
    import copy
    result=list(observations)
    valid=[i for i,o in enumerate(result) if o and o.get('silhouette')]
    def sample(outline):
        p=np.asarray(outline);edges=np.roll(p,-1,axis=0)-p
        length=np.linalg.norm(edges,axis=1);distance=np.r_[0,np.cumsum(length)]
        at=np.linspace(0,distance[-1],64,endpoint=False)
        ids=np.minimum(np.searchsorted(distance,at,side='right')-1,len(p)-1)
        return p[ids]+edges[ids]*((at-distance[ids])/np.maximum(1e-8,length[ids]))[:,None]
    for lo,hi in zip(valid,valid[1:]):
        if hi-lo<=1 or hi-lo>3 or times[hi]-times[lo]>max_gap:continue
        if segments is not None and not any(s['start']<=times[lo]-times[0] and
            times[hi]-times[0]<s['end'] for s in segments):continue
        a=sample(result[lo]['silhouette']['outline']);b=sample(result[hi]['silhouette']['outline'])
        b=min((np.roll(q,k,axis=0) for q in (b,b[::-1]) for k in range(64)),key=lambda q:np.square(q-a).sum())
        for i in range(lo+1,hi):
            t=float((times[i]-times[lo])/(times[hi]-times[lo]));p=a*(1-t)+b*t
            shape=copy.deepcopy(result[lo]['silhouette'])
            shape.update(outline=p.tolist(),bounds=[*p.min(0),*p.max(0)],
                area=float(result[lo]['silhouette']['area']*(1-t)+result[hi]['silhouette']['area']*t),
                interpolated=True,neighbour_indices=[lo,hi])
            result[i]={'source':'bounded-silhouette-gap','confidence':min(result[lo].get('confidence',.8),result[hi].get('confidence',.8)),
                'silhouette':shape,'closed':1.,'center':p.mean(0).tolist(),'radius':float(np.sqrt(shape['area']))}
    return result


def approach_candidates(screen,side):
    """Start video verification before fast approaches lose their landmarks.

    This function does not establish a punch, change a pose or synthesize lost
    landmarks. Candidates must subsequently pass the video-mask growth check.
    """
    rows=screen['frames'];times=np.array([r['time'] for r in rows]);result=[]
    observations=[fist_observation(r.get('hands',{}).get(side)) for r in rows]
    # A sustained fist only requests segmentation; it cannot animate an arm.
    # Do not wait for size growth here: the largest frames often have no joints.
    present=np.array([o is not None for o in observations])
    edges=np.diff(np.r_[False,present,False].astype(int))
    for first,end in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        ready=int(np.searchsorted(times,times[first]+.10-1e-6))
        if ready>=end or ready-first<3:continue
        centers=np.array([observations[i]['center'] for i in range(first,ready+1)])
        radii=np.array([observations[i]['radius'] for i in range(first,ready+1)])
        dt=np.diff(times[first:ready+1])
        if (dt>.10).any() or (np.linalg.norm(np.diff(centers,axis=0),axis=1)>
                             np.maximum(.045,2*radii[:-1])).any():continue
        # Choose the smallest reliable observed fist, not a near-lens pose.
        ref=min(range(first,ready+1),key=lambda j:observations[j]['radius'])
        result.append({'side':side,'reference_index':int(ref),'peak_index':ready,
            'reference':float(times[ref]-times[0]),'start':float(times[ref]-times[0]),
            'peak':float(times[ready]-times[0]),'release':float(times[ready]-times[0]),
            'end':float(min(times[-1],times[ref]+1.2)-times[0]),
            'candidate_only':True,'trigger':'sustained-observed-fist'})
    faces=[(i,r['face'].get('scale',r['face']['width'])) for i,r in enumerate(rows) if r.get('face')]
    if not faces:return result
    face=np.interp(times,times[[i for i,_ in faces]],[v for _,v in faces])
    sizes=np.array([o['radius']/max(.01,face[i]) if o else np.nan for i,o in enumerate(observations)])
    for i in range(5,len(rows)):
        ids=[j for j in range(max(0,i-18),i+1) if observations[j] is not None]
        if len(ids)<6 or ids[-1]!=i:continue
        ref=min(ids[:-2],key=lambda j:sizes[j])
        values=sizes[ref:i+1]
        if len(values)<4 or not np.isfinite(values).all():continue
        if sizes[i]<sizes[ref]*1.2 or sizes[i]-sizes[ref]<.07:continue
        if min(values[-2:])<sizes[ref]*1.15 or np.max(abs(np.diff(np.log(values))))>.30:continue
        if observations[i]['center'][1]>observations[ref]['center'][1]+.10:continue
        face_row=rows[ref].get('face')
        if face_row and 'center' in face_row and observations[ref]['center'][1]>face_row['center'][1]+.30:continue
        seed={'side':side,'reference_index':int(ref),'peak_index':i,
            'reference':float(times[ref]-times[0]),'start':float(times[ref]-times[0]),
            'peak':float(times[i]-times[0]),'release':float(times[i]-times[0]),
            'end':float(min(times[-1],times[i]+.5)-times[0]),'candidate_only':True}
        if any(s['start']-.2<=seed['start']<=s['end']+.2 for s in result):continue
        result.append(seed)
    return sorted(result,key=lambda s:s['reference_index'])

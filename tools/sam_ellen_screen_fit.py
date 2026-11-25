"""Fit palm pose, elbow plane and bounded finger hinges without changing lengths."""
import math

import numpy as np
from mathutils import Matrix, Quaternion, Vector


def hand_points(rig, side):
    points = [rig.pose.bones['Wrist_'+side].head.copy()]
    for finger in ('Thumb', 'Index', 'Middle', 'Ring', 'Little'):
        # MMD thumbs are numbered 0..2, the other fingers 1..3.
        names = [finger+'Finger'+str(j)+'_'+side for j in (1, 2, 3)]
        if finger == 'Thumb':
            names = [b.name for b in rig.pose.bones
                     if b.mmd_bone.name_j.startswith(('左' if side == 'L' else '右')+'親指')]
            names.sort(key=lambda n: len(rig.pose.bones[n].parent_recursive))
        bones = [rig.pose.bones[n] for n in names[:3]]
        points.extend([b.head.copy() for b in bones] + [bones[-1].tail.copy()])
    return points


def landmark_observations(screen, side, smoothing=.035, space='FACE',position_seconds=.045):
    # Legacy confidence is a handedness classification score, not localization
    # confidence. Identity was assigned by wrist correspondence in the worker.
    # Bridge short detection gaps offline; repeated fade-outs cause arm spasms.
    frames = screen['frames']
    from sam_ellen_arm_visibility import observed_frames,hand_landmark_mask,screen_hand
    usable=observed_frames({'screen':screen},side)
    valid = {}
    for i, row in enumerate(frames):
        hand, face = screen_hand(screen,i,side), row.get('face')
        if (not usable[i]
                or space != 'FRAME' and not (face and .005 < face.get('width', 0) < 2)):
            continue
        points = np.asarray(hand.get('points', []), dtype=float)
        mask = hand_landmark_mask(hand)
        if points.shape == (21, 2) and mask.shape == (21,) and np.isfinite(points).all() and mask.sum() >= 4:
            valid[i] = points, mask
    interpolated = {}
    indices = sorted(valid)
    for left, right in zip(indices, indices[1:]):
        gap = frames[right]['time']-frames[left]['time']
        mask = valid[left][1] & valid[right][1]
        if right-left <= 1 or gap > .24 or mask.sum() < 4:
            continue
        for i in range(left+1, right):
            w = (frames[i]['time']-frames[left]['time'])/gap
            interpolated[i] = {'points': (valid[left][0]*(1-w)+valid[right][0]*w).tolist(),
                               'in_frame': mask.tolist(), 'confidence': 1.}
    result, previous, last, started = [], None, -100., -100.
    for i, row in enumerate(frames):
        face, hand, time = row.get('face'), interpolated.get(i, screen_hand(screen,i,side) if usable[i] else None), row['time']
        if i in interpolated and space != 'FRAME' and not face:
            face = previous['face'] if previous else None
        good = (hand
                and (space == 'FRAME' or face and .005 < face.get('width', 0) < 2)
                and len(hand.get('points', [])) == 21)
        if good:
            if time-last > .18:
                started = time
            points = np.asarray(hand['points'], dtype=float)
            mask = hand_landmark_mask(hand)
            good = points.shape == (21, 2) and mask.shape == (21,) and np.isfinite(points).all() and mask.sum() >= 4
        if good:
            previous = {'points': points, 'mask': mask, 'face': face,
                        'weight': min(1., (time-started+.025)/.05)}
            last = time
            result.append(dict(previous, held=True, interpolated=True) if i in interpolated else dict(previous))
        elif previous is not None and time-last < .18:
            result.append(dict(previous, weight=previous['weight']*max(0, 1-(time-last)/.18), held=True))
        else:
            result.append(None)
    # Centered trajectories remove detector noise without a causal hand/face lag.
    from sam_ellen_temporal import trajectory
    visible=np.array([r is not None for r in result])
    edges=np.diff(np.r_[False,visible,False].astype(int))
    for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        if b-a<3:continue
        times=np.array([frames[i]['time'] for i in range(a,b)])
        raw=np.array([r['points'] for r in result[a:b]])
        for j in range(21):
            weights=np.array([float(r['mask'][j])*(.15 if r.get('held') and not r.get('interpolated') else 1.)
                              for r in result[a:b]])
            if weights.sum()<2:continue
            raw[:,j]=trajectory(times,raw[:,j],position_seconds,weights,robust=.015)
        # A binary crop mask must not abruptly change the fitted hand centre.
        # Smooth support alongside positions; only a short interval around an
        # actually measured joint can contribute outside its original mask.
        masks=np.array([r['mask'] for r in result[a:b]],dtype=float)
        support=np.clip(trajectory(times,masks,.055),0.,1.)
        for j in range(21):
            known=np.flatnonzero(masks[:,j])
            if not len(known):support[:,j]=0.;continue
            distance=np.min(abs(times[:,None]-times[known][None,:]),axis=1)
            support[:,j]*=np.clip((.12-distance)/.06,0.,1.)
        for k,i in enumerate(range(a,b)):
            result[i]['points']=raw[k]
            result[i]['joint_weights']=support[k]
    return result


def projected(points, camera):
    projection, aspect = np.asarray(camera[1], dtype=float), camera[2]
    clip = np.c_[points, np.ones(len(points))] @ projection.T
    w = np.where(abs(clip[:, 3]) < 1e-8, 1e-8, clip[:, 3])
    return np.stack(((clip[:, 0]/w+1)*.5*aspect, (1-clip[:, 1]/w)*.5), axis=1)


def rotation_vector(value):
    length = float(np.linalg.norm(value))
    return np.asarray(Quaternion(Vector(value/length), length).to_matrix()) if length > 1e-10 else np.eye(3)


def stabilize_capture(channels, data, options, observations=None):
    """Damp capture discontinuities before fitting; never filter solved IK angles."""
    from sam_ellen_temporal import quaternions
    times = np.array([r['time'] for r in data['frames']])
    relative = times-times[0]
    start, end = options.get('start', 0), options.get('end', 0)
    weight = np.clip((relative-start)/.15, 0, 1)*options['strength']
    if end > 0:
        weight *= np.clip((end-relative)/.15, 0, 1)
    reliability = (np.array([1. if o is not None and (not o.get('held') or o.get('interpolated'))
                             else .15 for o in observations]) if observations is not None else None)
    for part in ('Arm', 'Elbow', 'Wrist'):
        name = part+'_'+options['side']
        samples = channels[name]
        filtered = quaternions(times, [q for _, q in samples], .065, weights=reliability)
        channels[name] = [(f, tuple(Quaternion(q).slerp(filtered[i], float(weight[i]))))
                          if weight[i] > 0 else (f, q) for i, (f, q) in enumerate(samples)]
    # Finger directions retain their articulation; damp only the short residual
    # tremor in visible capture before the rigid hand and contact are solved.
    for name,samples in list(channels.items()):
        if not name.endswith('_'+options['side']) or not name.startswith(('Thumb','IndexFinger','MiddleFinger','RingFinger','LittleFinger')):continue
        filtered=quaternions(times,[q for _,q in samples],.035,weights=reliability)
        channels[name]=[(f,tuple(Quaternion(q).slerp(filtered[i],float(weight[i]))))
                        if weight[i]>0 and observations is not None and observations[i] is not None else (f,q)
                        for i,(f,q) in enumerate(samples)]


def stable_goals(times, targets, rotations, bends, captured, active, cameras, allowed):
    """Blend real correction offsets at visibility edges, then fit continuous goals.

    Extend only a short correction across detector loss. An inactive target is the
    capture itself, so blending that target (the old implementation) cannot fade
    an alignment correction. Longer gaps still return to the captured movement.
    """
    from sam_ellen_temporal import trajectory, quaternions
    active = np.asarray(active, dtype=bool)
    enabled = active.copy()
    visible = np.flatnonzero(active)
    if not len(visible):
        return targets, rotations, bends, enabled
    for i in np.flatnonzero(~active & allowed):
        j = int(visible[np.argmin(abs(times[visible]-times[i]))])
        distance = abs(times[j]-times[i])
        if distance >= .18:
            continue
        w = 1-distance/.18
        w = w*w*(3-2*w)
        offset = Vector(targets[j])-captured[j][0]
        targets[i] = tuple(captured[i][0]+offset*w)
        delta = Quaternion(rotations[j])@captured[j][1].inverted()
        rotations[i] = tuple(Quaternion().slerp(delta, w)@captured[i][1])
        # The bend plane is also a correction; copying a fully corrected plane
        # into the first barely active frame creates a visible elbow snap.
        bend=Vector(bends[i]).lerp(Vector(bends[j]),w)
        bends[i]=tuple(bend.normalized()) if bend.length>1e-7 else bends[i]
        enabled[i] = True
    # Include the captured boundary poses as fixed anchors. Filter depth more
    # strongly than image position: monocular depth is much less constrained.
    edges = np.diff(np.r_[False, enabled, False].astype(np.int8))
    for begin, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        first,last=begin,end-1
        begin, end = max(0, begin-1), min(len(times), end+1)
        local = times[begin:end]
        camera_points = np.array([cameras[i][0]@Vector(targets[i]) for i in range(begin, end)])
        perspective = np.array([cameras[i][3] == 'PERSP' for i in range(begin, end)])
        ray_depth = np.where(perspective, np.maximum(.05, -camera_points[:, 2]), 1.)
        image_position = camera_points[:, :2]/ray_depth[:, None]
        image_position = trajectory(local, image_position, .05, robust=.015)
        camera_points[:, 2:] = trajectory(local, camera_points[:, 2:], .14, robust=.035)
        filtered_depth = np.where(perspective, np.maximum(.05, -camera_points[:, 2]), 1.)
        camera_points[:, :2] = image_position*filtered_depth[:, None]
        filtered_rotations = quaternions(local, rotations[begin:end], .11)
        filtered_bends = trajectory(local, bends[begin:end], .10, robust=.5)
        for k, i in enumerate(range(begin, end)):
            if enabled[i] and allowed[i]:
                boundary=1.
                if first>0:boundary=min(boundary,(times[i]-times[first-1])/.20)
                if last<len(times)-1:boundary=min(boundary,(times[last+1]-times[i])/.20)
                boundary=float(np.clip(boundary,0,1));boundary=boundary*boundary*(3-2*boundary)
                fitted=cameras[i][0].inverted()@Vector(camera_points[k])
                targets[i] = tuple(captured[i][0].lerp(fitted,boundary))
                rotations[i] = tuple(captured[i][1].slerp(filtered_rotations[k],boundary))
                bends[i] = tuple(filtered_bends[k])
    return targets, rotations, bends, enabled


def fit_hand(points, target, mask, camera, depth_weight, previous=None, head_depth=None,
             match_image=False, reach=None, articulate=False, joint_weights=None):
    """Fit camera projections, with optional image size and arm reach constraints."""
    transform = camera[0]
    original = np.array([transform @ p for p in points])
    wrist, offsets = original[0].copy(), original-original[0]
    palm = np.array([0, 5, 9, 13, 17])
    anchors = palm[mask[palm]]
    if len(anchors) < 3:
        anchors = np.flatnonzero(mask)
    weights = np.full(21, .25)
    weights[palm] = 1.
    if match_image:
        weights[[4, 8, 12, 16, 20]] = .75
    if articulate:
        weights[:]=.08
        weights[palm]=1.
    weights*=np.asarray(joint_weights,dtype=float) if joint_weights is not None else mask
    # In image mode use one weighted objective for all support configurations.
    # Switching between MCP-only and all-finger centroids at 2/3 MCPs produced
    # competing depth/orientation solutions and a large visible hand tremor.
    center_weights=weights/max(1e-8,float(weights.sum()))
    image = projected(original, camera)
    continuous_center=match_image and joint_weights is not None
    target_center = np.sum(target*center_weights[:,None],axis=0) if continuous_center else target[anchors].mean(axis=0)
    # Normalize source extent to the native hand's current projection. A human's
    # hand/face ratio is not evidence that the character should reach forward.
    pairs = [(a, b) for j, a in enumerate(anchors) for b in anchors[j+1:]]
    native_span = np.median([np.linalg.norm(image[a]-image[b]) for a, b in pairs])
    source_span = np.median([np.linalg.norm(target[a]-target[b]) for a, b in pairs])
    shape = target-target_center
    if not match_image:
        shape = shape*native_span/max(source_span, 1e-6)
    regular = .015 if match_image else .16
    prior_depth = float(original[palm, 2].mean())
    palm_length = max(.01, np.linalg.norm(original[9]-original[0]))
    depth_limit = palm_length*(9. if match_image else .15)*depth_weight if camera[3] == 'PERSP' else 0.
    x = np.zeros(6)
    max_angle = math.radians(80 if match_image else 45)
    previous_xyz = previous.get('camera_wrist') if previous else None

    def evaluate(value):
        r = rotation_vector(value[3:])
        p = offsets @ r.T + wrist + value[:3]
        screen = projected(p, camera)
        center = np.sum(screen*center_weights[:,None],axis=0) if continuous_center else screen[anchors].mean(axis=0)
        error = (screen-center)-shape
        norm = np.linalg.norm(error, axis=1)
        huber = np.minimum(1., .045/np.maximum(norm, 1e-6))
        residual = (error*np.sqrt(weights*huber)[:, None]).ravel().tolist()
        residual.extend(((center-target_center)*4.).tolist())
        residual.extend((value[3:]*regular).tolist())
        residual.append((p[palm, 2].mean()-prior_depth)*(.04 if match_image else 4.))
        if head_depth is not None:
            plane, weight = head_depth
            residual.append((p[palm, 2].mean()+plane)*2.*weight)
        if previous_xyz is not None:
            residual.extend(((p[0]-previous_xyz)*.055).tolist())
        return np.asarray(residual)

    def constrain(value):
        value = value.copy()
        value[3:] *= min(1., max_angle/max(1e-8, np.linalg.norm(value[3:])))
        rotated = offsets @ rotation_vector(value[3:]).T+wrist
        if head_depth is None:
            neutral = prior_depth-rotated[palm, 2].mean()
            value[2] = np.clip(value[2], neutral-depth_limit, neutral+depth_limit)
        near_z = float(rotated[:, 2].max()) if match_image else wrist[2]
        value[2] = min(value[2], -.15-near_z)
        if reach is not None:
            center, radius = reach
            goal = transform.inverted() @ Vector(wrist+value[:3])
            offset = goal-center
            if offset.length > radius:
                goal = center+offset.normalized()*radius
                value[:3] = np.asarray(transform @ goal)-wrist
        return value

    residual = evaluate(x)
    for _ in range(28 if match_image else 14):
        columns = []
        for j in range(6):
            trial = x.copy()
            trial[j] += 1e-4
            columns.append((evaluate(trial)-residual)/1e-4)
        jac = np.stack(columns, axis=1)
        step = np.linalg.solve(jac.T@jac+np.eye(6)*.002, -jac.T@residual)
        step[:3] *= min(1., .07/max(1e-8, np.linalg.norm(step[:3])))
        step[3:] *= min(1., .14/max(1e-8, np.linalg.norm(step[3:])))
        improved = False
        for scale in (1., .5, .25):
            trial = constrain(x+step*scale)
            r = evaluate(trial)
            if np.dot(r, r) < np.dot(residual, residual):
                x, residual, improved = trial, r, True
                break
        if not improved or np.linalg.norm(step) < 1e-5:
            break
    r = rotation_vector(x[3:])
    fitted = offsets @ r.T+wrist+x[:3]
    goal = transform.inverted() @ Vector(fitted[0])
    basis = transform.to_quaternion().to_matrix()
    delta = basis.inverted() @ Matrix(r.tolist()) @ basis
    reach_limited = reach is not None and (goal-reach[0]).length >= reach[1]-1e-4
    return goal, delta.to_quaternion(), {'camera_wrist': fitted[0], 'angle': float(np.linalg.norm(x[3:])),
        'reach_limited': reach_limited,
        'palm_depth_change': float(fitted[palm, 2].mean()-prior_depth),
        'depth_limit': float(depth_limit)}


def correct_landmarks(scratch, rig, channels, data, options, cameras, report):
    from sam_ellen_retarget import correct_contacts
    from sam_ellen_screen import project
    side = options['side']
    screen = data['screen']
    observations = landmark_observations(screen, side, space=options.get('space', 'FACE'))
    from sam_ellen_chin_contact import observations as chin_observations,goal as chin_goal
    chin_support=(chin_observations(screen,side) if options.get('chin_anchor') and options.get('chin_contact',True)
                  and options.get('space')=='FRAME' else [None]*len(data['frames']))
    if options.get('mirror'):
        for support in chin_support:
            if support is not None:
                support['offset']=support['offset']*np.array([-1.,1.])
    chin_context={}
    previous, last_time = None, -100.
    metadata = {}

    def goal(index):
        nonlocal previous, last_time
        row, observation = data['frames'][index], observations[index]
        time = row['time']-data['frames'][0]['time']
        if (observation is None or time < options.get('start', 0)
                or options.get('end', 0) > 0 and time >= options['end']
                or options['strength'] <= 0):
            report.append({'time': row['time'], 'active': False})
            previous = None
            return None
        camera = cameras[index]
        eyes = [rig.pose.bones['Eye_'+s].head.copy() for s in ('L', 'R')]
        eye_center = sum(eyes, Vector())*.5
        eye_width = (project(eyes[0], camera)-project(eyes[1], camera)).length
        face = observation['face']
        source = observation['points'].copy()
        mask = observation['mask']
        if options.get('space', 'FACE') == 'FRAME':
            aspect = screen.get('aspect', data.get('video_orientation', {}).get('decoded_height_width', [16, 9])[1]
                                /data.get('video_orientation', {}).get('decoded_height_width', [16, 9])[0])
            target = source.copy()
            target[:, 0] *= camera[2]/aspect
            if options.get('mirror'):
                target[:, 0] = camera[2]-target[:, 0]
        else:
            relative = (source-np.array(face['center']))/face['width']
            if options.get('mirror'):
                relative[:, 0] *= -1
            target = np.array(project(eye_center, camera))+relative*eye_width
        points = hand_points(rig, side)
        if chin_support[index] is not None:
            anchor=options['chin_anchor']
            chin=rig.pose.bones[anchor['bone']].matrix@Vector(anchor['local'])
            chin_context[index]=(chin.copy(),points[5]-points[0],points[17]-points[5])
        before = np.array([project(p, camera) for p in points])
        anchors = np.array([0, 5, 9, 13, 17])
        anchors = anchors[mask[anchors]]
        if len(anchors) < 3:
            anchors = np.flatnonzero(mask)
        if row['time']-last_time > .2:
            previous = None
        head_depth = None
        if options.get('head_depth_enabled'):
            from sam_ellen_gestures import interval_weight
            w = interval_weight(time, options['head_depth_start'], options['head_depth_end'], .25)
            if w > 0:
                head_depth = (-(camera[0]@eye_center).z+options['head_depth_offset'], w)
        wrist = rig.pose.bones['Wrist_'+side]
        arm, elbow = [rig.pose.bones[n+'_'+side] for n in ('Arm', 'Elbow')]
        reach = (elbow.head-arm.head).length+(wrist.head-elbow.head).length
        image_mode = options.get('space') == 'FRAME'
        fitted, delta, previous_fit = fit_hand(points, target, mask, camera, options['depth'], previous, head_depth,
            match_image=image_mode, reach=(arm.head.copy(), reach*.985) if image_mode else None,
            articulate=options.get('finger_articulation',True),joint_weights=observation.get('joint_weights'))
        strength = options['strength']*observation['weight']
        # Keep the initial/end interval fades consistent with legacy alignment.
        if options.get('start',0)>0:
            strength *= min(1, max(0, (time-options['start'])/.15))
        if options.get('end', 0) > 0:
            strength *= min(1, max(0, (options['end']-time)/.15))
        fitted = wrist.head.lerp(fitted, strength)
        rotation = Quaternion().slerp(delta, strength) @ wrist.matrix.to_quaternion()
        if options.get('front'):
            camera_goal = camera[0]@fitted
            camera_goal.z += options['front']*strength
            fitted = camera[0].inverted()@camera_goal
        offset = fitted-arm.head
        outside_reach = offset.length > reach*.985
        limited = outside_reach or previous_fit['reach_limited']
        if outside_reach:
            fitted = arm.head+offset.normalized()*reach*.985
        # A conservative head exclusion volume, not mesh collision simulation.
        head = next((b for b in rig.pose.bones if b.mmd_bone.name_j == '頭'), None)
        clearance = False
        if head:
            center = eye_center+(head.head-eye_center)*.4
            radius = (eyes[0]-eyes[1]).length*.78
            v = fitted-center
            if v.length < radius:
                fitted = center+v.normalized()*radius if v.length > 1e-7 else fitted
                clearance = True
        previous, last_time = previous_fit, row['time']
        metadata[index] = (target, mask, anchors)
        report.append({'time': row['time'], 'active': True, 'weight': strength,
                       'interpolated': bool(observation.get('interpolated')),
                       'model': 'image-landmarks-v3' if image_mode else 'native-hand-proportions-v2',
                       'palm_depth_change_m': previous_fit['palm_depth_change'],
                       'depth_limit_m': previous_fit['depth_limit'],
                       'limited': limited, 'head_clearance': clearance,
                       'held': bool(observation.get('held')), 'landmarks': int(mask.sum()),
                       'before_center_error_pixels': float(np.linalg.norm(before[anchors].mean(axis=0)
                            -target[anchors].mean(axis=0))*options.get('height', 1920)),
                       'before_error_pixels': float(np.linalg.norm(before-target, axis=1)[mask].mean()*options.get('height', 1920))})
        if options.get('dual'):
            report[-1]['before_center_xy'] = before[anchors].mean(axis=0).tolist()
        return fitted, rotation

    # Fit all targets before solving any arm. Filtering joint angles afterwards
    # would pull the wrist away from its image target and loosen contact.
    if any(o is not None for o in observations):
        stabilize_capture(channels, data, options, observations)
    if options.get('space')=='FRAME' and not options.get('unified_arms'):
        from sam_ellen_arm_visibility import apply_visibility
        from sam_ellen_arm_policy import activation
        apply_visibility(scratch,rig,channels,data,side,initialize_only=True,
            allowed=activation(np.array([r['time'] for r in data['frames']]),options)>0)
    originals = {name: list(samples) for name, samples in channels.items()}
    times = np.array([row['time'] for row in data['frames']])
    targets, rotations, bends, captured, active = [], [], [], [], []
    previous_bend = None
    for i, row in enumerate(data['frames']):
        for name, samples in originals.items():
            rig.pose.bones[name].rotation_quaternion = samples[i][1]
        scratch.view_layers[0].update()
        arm, elbow, wrist = [rig.pose.bones[n+'_'+side] for n in ('Arm', 'Elbow', 'Wrist')]
        result = goal(i)
        position, orientation = result if result else (wrist.head.copy(), wrist.matrix.to_quaternion())
        captured.append((wrist.head.copy(), wrist.matrix.to_quaternion()))
        direction = (position-arm.head).normalized()
        original_direction = (wrist.head-arm.head).normalized()
        bend = elbow.head-arm.head
        bend -= original_direction*bend.dot(original_direction)
        if bend.length > .001:
            bend = original_direction.rotation_difference(direction)@bend.normalized()
        elif previous_bend is not None:
            bend = previous_bend.copy()
        else:
            bend = arm.matrix.to_quaternion()@Vector((0, 0, 1))
        bend -= direction*bend.dot(direction)
        if bend.length < 1e-6:
            bend = direction.orthogonal()
        bend.normalize()
        if result is not None and options.get('space') == 'FRAME':
            from sam_ellen_articulation import elbow_plane
            bend, elbow_report = elbow_plane(arm, elbow, wrist, position, bend, row, side, cameras[i], data, options)
            report[i].update(elbow_report)
        previous_bend = bend.copy()
        targets.append(tuple(position)); rotations.append(tuple(orientation)); bends.append(tuple(bend))
        active.append(result is not None)
        yield i+1, 3*len(times)
    relative = times-times[0]
    allowed = (relative >= options.get('start', 0)) & (options['strength'] > 0)
    if options.get('end', 0) > 0:
        allowed &= relative < options['end']
    targets, rotations, bends, enabled = stable_goals(
        times, targets, rotations, bends, captured, active, cameras, allowed)
    for i in chin_context:
        if enabled[i] and allowed[i]:
            targets[i],target,anchor,weight=chin_goal(targets[i],rotations[i],captured[i],chin_context[i],
                cameras[i],chin_support[i],options['strength'])
            report[i].update(chin_contact_weight=weight,chin_target=list(target),
                before_chin_error_pixels=(project(anchor,cameras[i])-target).length*options.get('height',1920),
                chin_anchor_source=options['chin_anchor']['source'])

    from sam_ellen_arm_chain import direction_evidence,refine_goal
    arm_evidence=direction_evidence(data,side,options) if options.get('space')=='FRAME' else None
    previous_arm=None
    def continuous_goal(i):
        nonlocal previous_arm
        if not allowed[i] or observations[i] is None:
            previous_arm=None
            return None
        position,bend=Vector(targets[i]),Vector(bends[i])
        if arm_evidence is not None:
            arm,elbow,wrist=[rig.pose.bones[n+'_'+side] for n in ('Arm','Elbow','Wrist')]
            if not enabled[i]:
                position=wrist.head.copy()
                direction=(position-arm.head).normalized();bend=elbow.head-arm.head
                bend-=direction*bend.dot(direction)
            face=data['screen']['frames'][i].get('face');front_depth=None
            if face and arm_evidence[i] is not None:
                xy=np.array(project(position,cameras[i]));center=np.array(face['center'])
                center[0]*=cameras[i][2]/data['screen']['aspect']
                if options.get('mirror'):center[0]=cameras[i][2]-center[0]
                if np.linalg.norm(xy-center)<face['width']*2.5:
                    eye=(rig.pose.bones['Eye_L'].head+rig.pose.bones['Eye_R'].head)*.5
                    front_depth=-(cameras[i][0]@eye).z+.01
            palm=None
            if i in metadata:
                target,mask,anchors=metadata[i]
                model=hand_points(rig,side)
                delta=Quaternion(rotations[i])@wrist.matrix.to_quaternion().inverted()
                offsets=np.array([delta@(model[j]-wrist.head) for j in anchors])
                palm=(offsets,target[anchors],.15 if observations[i].get('held') else 1.)
            from sam_ellen_arm_chain import torso_front_plane
            position,bend,details=refine_goal(arm,elbow,wrist,position,bend,cameras[i],arm_evidence[i],previous_arm,
                front_depth=front_depth,max_step_degrees=min(30,540*(times[i]-times[i-1] if i else 1/data['fps'])),
                palm=palm,front_plane=torso_front_plane(rig),hand_axis=Quaternion(rotations[i])@Vector((0,1,0)))
            if details.get('elbow_position') is not None:
                previous_arm=tuple(np.array(v) for v in (arm.head,details['elbow_position'],position,details['normal']))
            report[i].update(details)
        return position,Quaternion(rotations[i]),bend

    for done, total in correct_contacts(scratch, rig, channels, data,
            {'side': side, 'stable_ik': True,'anatomical':arm_evidence is not None}, goal_provider=continuous_goal):
        yield len(times)+done, 3*len(times)
    if options.get('finger_articulation', True):
        from sam_ellen_articulation import fit_fingers
        for done,total in fit_fingers(scratch,rig,channels,data,options,cameras,metadata,report):
            yield 2*len(times),3*len(times)
    if options.get('space')=='FRAME' and not options.get('unified_arms'):
        # Finger fitting changes the projected hand shape. Refit the complete
        # native palm afterwards so the arm is solved from that final shape,
        # rather than keeping a wrist pinned to an earlier pose or bad joint 0.
        from sam_ellen_hand_temporal import refit_palm
        for done,total in refit_palm(scratch,rig,channels,data,options,cameras,
                observations,metadata,allowed,arm_evidence,report):
            yield 2*len(times)+done,3*len(times)
    for i, row in enumerate(report):
        if not row['active']:
            continue
        for name, samples in channels.items():
            rig.pose.bones[name].rotation_quaternion = samples[i][1]
        scratch.view_layers[0].update()
        target, mask, anchors = metadata[i]
        actual = np.array([project(p, cameras[i]) for p in hand_points(rig, side)])
        row['after_center_error_pixels'] = float(np.linalg.norm(actual[anchors].mean(axis=0)
            -target[anchors].mean(axis=0))*options.get('height', 1920))
        row['after_error_pixels'] = float(np.linalg.norm(actual-target, axis=1)[mask].mean()*options.get('height', 1920))
        tips = np.array([4, 8, 12, 16, 20])
        tips = tips[mask[tips]]
        row['after_tip_error_pixels'] = (float(np.linalg.norm(actual[tips]-target[tips], axis=1).mean()
            *options.get('height', 1920)) if len(tips) else None)
        pairs = [(a, b) for j, a in enumerate(anchors) for b in anchors[j+1:]]
        row['after_palm_size_ratio'] = float(np.median([np.linalg.norm(actual[a]-actual[b]) for a, b in pairs])
            /max(1e-6, np.median([np.linalg.norm(target[a]-target[b]) for a, b in pairs])))
        if i in chin_context:
            row['after_chin_error_pixels']=(project(hand_points(rig,side)[5],cameras[i])-Vector(row['chin_target'])).length*options.get('height',1920)
            row['after_chin_depth_gap_m']=float((cameras[i][0]@hand_points(rig,side)[5]).z-(cameras[i][0]@chin_context[i][0]).z)
        row['trajectory_seconds'] = .05
        row['depth_trajectory_seconds'] = .14
        row['rotation_trajectory_seconds'] = .11
        row['stability_version'] = 3
        row['stable_elbow'] = True
        yield 2*len(times)+i+1, 3*len(times)


def measure_final(scratch, rig, channels, data, options, cameras, report, contact=None, punch=None):
    """Report projections after all pose overrides, using the same observations."""
    if options['side'] == 'BOTH':
        from sam_ellen_dual_screen import measure_dual
        yield from measure_dual(scratch, rig, channels, data, options, cameras, report, contact, punch)
        return
    from sam_ellen_screen import project, palm_points
    side = options['side']
    observations = landmark_observations(data['screen'], side, space=options.get('space', 'FACE'))
    for i, row in enumerate(report):
        if row.get('active'):
            for name, samples in channels.items():
                rig.pose.bones[name].rotation_quaternion = samples[i][1]
            scratch.view_layers[0].update()
            time = data['frames'][i]['time']-data['frames'][0]['time']
            row['measurement_stage'] = 'final-pose'
            row['overrides'] = [label for label, override in (('mouth-contact', contact), ('punch', punch))
                if override and override.get('side') == side and override.get('strength', 1.) > 0
                and (any(s['start'] < time < s['end'] for s in override.get('segments', []))
                     if override.get('automatic') else override.get('start', 0) < time < override.get('end', 0))]
            camera = cameras[i]
            if options.get('landmarks') and observations[i] is not None:
                obs = observations[i]
                target, mask = obs['points'].copy(), obs['mask']
                eyes = [rig.pose.bones['Eye_'+s].head for s in ('L', 'R')]
                if options.get('space') == 'FRAME':
                    target[:, 0] *= camera[2]/data['screen']['aspect']
                    if options.get('mirror'):
                        target[:, 0] = camera[2]-target[:, 0]
                else:
                    relative = (target-np.asarray(obs['face']['center']))/obs['face']['width']
                    if options.get('mirror'):
                        relative[:, 0] *= -1
                    width = (project(eyes[0], camera)-project(eyes[1], camera)).length
                    target = np.asarray(project((eyes[0]+eyes[1])*.5, camera))+relative*width
                actual = np.array([project(p, camera) for p in hand_points(rig, side)])
                anchors = np.array([0, 5, 9, 13, 17])
                anchors = anchors[mask[anchors]]
                if len(anchors) < 3:
                    anchors = np.flatnonzero(mask)
                height = options.get('height', 1920)
                for key in ('after_error_pixels', 'after_center_error_pixels', 'after_tip_error_pixels'):
                    if key in row:
                        row['fit_'+key] = row[key]
                row['after_error_pixels'] = float(np.linalg.norm(actual-target, axis=1)[mask].mean()*height)
                row['after_center_error_pixels'] = float(np.linalg.norm(actual[anchors].mean(0)-target[anchors].mean(0))*height)
                if options.get('dual'):
                    row['final_center_xy'] = actual[anchors].mean(0).tolist()
                    row['target_center_xy'] = target[anchors].mean(0).tolist()
                tips = np.array([4, 8, 12, 16, 20])
                tips = tips[mask[tips]]
                row['after_tip_error_pixels'] = float(np.linalg.norm(actual[tips]-target[tips], axis=1).mean()*height) if len(tips) else None
                pairs = [(a, b) for j, a in enumerate(anchors) for b in anchors[j+1:]]
                row['after_palm_size_ratio'] = float(np.median([np.linalg.norm(actual[a]-actual[b]) for a, b in pairs])
                    /max(1e-6, np.median([np.linalg.norm(target[a]-target[b]) for a, b in pairs])))
            elif 'target_xy' in row:
                center = sum(palm_points(rig, side), Vector())/5
                row['after_error_eye_widths'] = (project(center, camera)-Vector(row['target_xy'])).length/row['eye_width']
        yield i+1, len(report)

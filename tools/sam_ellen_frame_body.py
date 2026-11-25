"""Bounded upper-body image fitting, followed by the existing arm solver."""
import math

import numpy as np
from mathutils import Quaternion, Vector


def rotation(value):
    angle = float(np.linalg.norm(value))
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = value/angle
    k = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3)+math.sin(angle)*k+(1-math.cos(angle))*(k@k)


def relative_face_depth(times, faces, reference):
    """Infer relative distance, not human-to-character absolute face dimensions."""
    from sam_ellen_temporal import trajectory
    times = np.asarray(times, dtype=float)
    sizes, weights = np.zeros(len(times)), np.zeros(len(times))
    for i, face in enumerate(faces):
        if not face:
            continue
        width, size = face.get('width', 0), face.get('scale', face.get('width', 0))
        if not np.isfinite([width, size]).all() or not .005 < width < 2 or not .005 < size < 2:
            continue
        sizes[i] = math.log(size)
        # Side-on eyes are a weaker distance cue, even with landmark-z compensation.
        weights[i] = np.clip((width/size-.3)/.45, .05, 1.)
    valid = np.flatnonzero(weights > 0)
    anchors = [i for i in reference if weights[i] > 0]
    if not anchors or len(valid) < 2:
        return np.ones(len(times)), np.zeros(len(times))
    sizes = np.interp(times, times[valid], sizes[valid])
    filtered = trajectory(times, sizes, .08, weights, robust=.04)
    baseline = float(np.median(filtered[anchors]))
    # Reject impulsive depth changes in log space, without clipping sustained
    # approaches to an arbitrary 1.4x. Physical near-plane limits live in the
    # character trajectory, after scene scale is known.
    delta=filtered-baseline
    for i in range(1,len(delta)):
        step=max(0.,times[i]-times[i-1])*1.5
        delta[i]=np.clip(delta[i],delta[i-1]-step,delta[i-1]+step)
    ratio = np.exp(np.clip(delta, math.log(.15), math.log(6.)))
    # Bridge brief losses, but do not invent depth through extended occlusion.
    insertion = np.searchsorted(times[valid], times)
    left = times[valid[np.clip(insertion-1, 0, len(valid)-1)]]
    right = times[valid[np.clip(insertion, 0, len(valid)-1)]]
    distance = np.minimum(abs(times-left), abs(times-right))
    reliability = np.interp(times, times[valid], weights[valid])*np.clip(1-distance/.25, 0, 1)
    return ratio, reliability


def stabilize_capture(channels,data,options,rig):
    """Filter the input before camera-relative head pose is reapplied."""
    from sam_ellen_retarget import body_bones,gaze_bones
    from sam_ellen_temporal import quaternions
    times=np.array([r['time'] for r in data['frames']])
    relative=times-times[0]
    from sam_ellen_arm_policy import activation
    weight=activation(times,options)
    for bone in [*body_bones(rig),gaze_bones(rig)['首']]:
        values=channels[bone.name]
        filtered=quaternions(times,[q for _,q in values],.09)
        channels[bone.name]=[(f,tuple(Quaternion(q).slerp(filtered[i],float(weight[i]))))
                             if weight[i]>0 else (f,q) for i,(f,q) in enumerate(values)]


def fit_body(scratch, rig, channels, data, options, cameras, report):
    from sam_ellen_retarget import body_bones, gaze_bones, set_global_rotation
    from sam_ellen_screen_fit import projected
    from sam_ellen_temporal import trajectory
    from sam_ellen_camera_fit import reference_indices
    spine = body_bones(rig)
    gaze = gaze_bones(rig)
    neck, head = gaze['首'], gaze['頭']
    eyes = [gaze['左目'], gaze['右目']]
    shoulders = [rig.pose.bones.get('Arm_'+s) for s in ('L', 'R')]
    if not spine or not all(shoulders):
        raise ValueError('Coordinated alignment needs upper-body and shoulder bones')
    changed = [*spine, neck, head]
    faces = [r.get('face') for r in data['screen']['frames']]
    times = np.array([r['time'] for r in data['frames']])
    source_aspect = data['screen']['aspect']
    source_height = data.get('video_orientation', {}).get('decoded_height_width', [0])[0]
    limit = math.radians(options.get('body_limit', 25))
    neck_limit = math.radians(12)
    size = 3*(len(spine)+1)
    previous = np.zeros(size)
    height = options.get('height', 1920)
    last_time = None
    contexts, corrections = [], []
    originals = {name: list(samples) for name, samples in channels.items()}
    try:
        reference = reference_indices(data)
    except ValueError:
        reference = [i for i, f in enumerate(faces) if f][:15]
    ratios, depth_weights = relative_face_depth(times, faces, reference)
    anchors = []
    for i in reference:
        for name, samples in originals.items():
            rig.pose.bones[name].rotation_quaternion = samples[i][1]
        scratch.view_layers[0].update()
        anchors.append(np.mean([b.head[:] for b in eyes], axis=0))
    anchor = np.median(anchors, axis=0) if anchors else None

    def constrain(value):
        lengths = np.linalg.norm(value[:-3].reshape(-1, 3), axis=1)
        value[:-3] *= min(1., limit/max(1e-10, float(lengths.sum())))
        value[-3:] *= min(1., neck_limit/max(1e-10, np.linalg.norm(value[-3:])))
        return value

    # Missing/cropped shoulders have continuously reduced influence.
    shoulder_samples = np.zeros((len(times), 2, 2))
    shoulder_weights = np.zeros((len(times), 2))
    if source_height:
        for i, row in enumerate(data['frames']):
            if not row.get('image_points'):
                continue
            for side, index in enumerate((5, 6)):
                point = np.asarray(row['image_points'][index][:2], dtype=float)/source_height
                if np.isfinite(point).all():
                    shoulder_samples[i, side] = point
                    margin = min(point[0], source_aspect-point[0], point[1], 1-point[1])
                    shoulder_weights[i, side] = np.clip(margin/.04, 0, 1)
        for side in range(2):
            if shoulder_weights[:, side].any():
                shoulder_samples[:, side] = trajectory(times, shoulder_samples[:, side], .06,
                    shoulder_weights[:, side], robust=.035)
        shoulder_weights = np.clip(trajectory(times, shoulder_weights, .04), 0, 1)
    for i, row in enumerate(data['frames']):
        for name, samples in originals.items():
            rig.pose.bones[name].rotation_quaternion = samples[i][1]
        scratch.view_layers[0].update()
        face, camera = faces[i], cameras[i]
        time = row['time']-times[0]
        strength = options['strength']*min(1., max(0., (time-options.get('start', 0))/.15))
        if options.get('end', 0) > 0:
            strength *= min(1., max(0., (options['end']-time)/.15))
        if not face or strength <= 0:
            contexts.append(None)
            corrections.append(previous.copy() if face is None and last_time is not None
                               and row['time']-last_time < .2 else np.zeros(size))
            yield i+1, 2*len(times)
            continue
        # Centered filtering removes detector jitter without delaying contact gestures.
        nearby = np.flatnonzero(abs(times-times[i]) < .15)
        valid = [j for j in nearby if faces[j]]
        weights = np.exp(-.5*((times[valid]-times[i])/.045)**2)
        target = np.average([faces[j]['center'] for j in valid], axis=0, weights=weights)
        target[0] *= camera[2]/source_aspect
        width = float(np.average([faces[j]['width'] for j in valid], weights=weights))
        width *= camera[2]/source_aspect
        if options.get('mirror'):
            target[0] = camera[2]-target[0]
        global_q = {b.name: b.matrix.to_quaternion() for b in changed}
        pivots = np.array([b.head[:] for b in spine])
        neck_pivot, head_point = np.array(neck.head), np.array(head.head)
        eye_offsets = np.array([b.head-head.head for b in eyes])
        shoulder_points = np.array([b.head[:] for b in shoulders])
        axes = np.array(camera[0].to_quaternion().inverted().to_matrix())
        anchor_depth = -float((camera[0]@Vector(anchor)).z) if anchor is not None else 0.
        use_depth = (not options.get('framing_track') and camera[3] == 'PERSP'
                     and anchor_depth > .05 and depth_weights[i] > 0)
        target_depth = anchor_depth/ratios[i] if use_depth else 0.
        shoulder_targets = []
        if source_height:
            for side in range(2):
                weight = shoulder_weights[i, side]
                if weight > .001:
                    point = shoulder_samples[i, side].copy()
                    point[0] *= camera[2]/source_aspect
                    if options.get('mirror'):
                        point[0] = camera[2]-point[0]
                    shoulder_targets.append((side, point, weight))

        def geometry(value):
            rotations = [rotation(axes@v) for v in value[:-3].reshape(-1, 3)]
            # Compose rotations from the distal pivot to the proximal pivot.
            def move(points):
                points = points.copy()
                for pivot, r in reversed(list(zip(pivots, rotations))):
                    points = (points-pivot)@r.T+pivot
                return points
            n, h = move(np.array([neck_pivot, head_point]))
            h = rotation(axes@value[-3:])@(h-n)+n
            return np.r_[h[None]+eye_offsets, move(shoulder_points)]

        def image_points(value):
            points = geometry(value)
            transformed = np.c_[points, np.ones(len(points))]@np.asarray(camera[0]).T
            return projected(transformed[:, :3], camera)

        original_image = image_points(np.zeros(size))
        original_points = geometry(np.zeros(size))
        original_depth = -float((np.c_[original_points, np.ones(4)]@np.asarray(camera[0]).T)[:2, 2].mean())
        target = original_image[:2].mean(0)*(1-strength)+target*strength
        # Perspective depth carries relative scale. Do not force anime eye spacing
        # to equal the actor's eye spacing, which previously saturated the spine.
        width = float(np.linalg.norm(original_image[0]-original_image[1]))
        if use_depth:
            target_depth = original_depth*(1-strength)+target_depth*strength
            width *= original_depth/max(.03, target_depth)

        def residual(value):
            image = image_points(value)
            terms = [*((.08 if options.get('framing_track') else 3)*(image[:2].mean(0)-target))]
            if use_depth:
                points = geometry(value)
                depth = -float((np.c_[points, np.ones(4)]@np.asarray(camera[0]).T)[:2, 2].mean())
                terms.append(2*depth_weights[i]*(depth-target_depth)/anchor_depth)
                terms.append(10*max(0., .04-depth)/anchor_depth)
            for index, point, weight in shoulder_targets:
                delta = image[index+2]-point
                if options.get('framing_track'):
                    # Camera translation handles framing; fit posture relative to
                    # the face instead of forcing the neck to follow its pixels.
                    delta -= image[:2].mean(0)-target
                terms.extend(.35*delta*strength*weight)
            return np.asarray(terms)

        if last_time is None or row['time']-last_time > .2:
            previous *= 0
        last_time = row['time']
        value = previous.copy()
        pose_prior = .015 if options.get('framing_track') else .0005
        temporal_prior = .006 if options.get('framing_track') else .002

        def objective(value):
            return np.dot(residual(value), residual(value))+pose_prior*np.dot(value,value)+temporal_prior*np.dot(value-previous,value-previous)

        for _ in range(20):
            r = residual(value)
            columns = []
            for j in range(size):
                trial = value.copy()
                trial[j] += 1e-4
                columns.append((residual(trial)-r)/1e-4)
            jac = np.stack(columns, axis=1)
            step = np.linalg.solve(jac.T@jac+(pose_prior+temporal_prior)*np.eye(size),
                                   -jac.T@r-pose_prior*value-temporal_prior*(value-previous))
            step *= min(1., .12/max(1e-10, np.linalg.norm(step)))
            old_cost = objective(value)
            improved = False
            for factor in (1., .5, .25, .1):
                trial = constrain(value+step*factor)
                if objective(trial) < old_cost:
                    value, improved = trial, True
                    break
            if not improved or np.linalg.norm(step) < 1e-6:
                break
        # Keep a continuous solution even when the residual changes very little.
        previous = value.copy()
        corrections.append(value.copy())
        contexts.append((global_q, axes, original_image, target, width, len(shoulder_targets),
                         original_depth, target_depth, use_depth))
        yield i+1, 2*len(times)

    from sam_ellen_temporal import bridge_missing
    supported=np.array([c is not None for c in contexts])
    corrections=bridge_missing(times,corrections,supported)
    corrections = trajectory(times, corrections, .09, robust=math.radians(6))
    for i, row in enumerate(data['frames']):
        for name, samples in originals.items():
            rig.pose.bones[name].rotation_quaternion = samples[i][1]
        scratch.view_layers[0].update()
        context = contexts[i]
        camera = cameras[i]
        if context is None:
            global_q = {b.name: b.matrix.to_quaternion() for b in changed}
            axes = np.array(camera[0].to_quaternion().inverted().to_matrix())
            points = np.array([b.head[:] for b in eyes]+[b.head[:] for b in shoulders])
            original_image = projected((np.c_[points,np.ones(4)]@np.asarray(camera[0]).T)[:,:3],camera)
            target, width, shoulder_count = original_image[:2].mean(0), 0., 0
            original_depth = -float((np.c_[points,np.ones(4)]@np.asarray(camera[0]).T)[:2,2].mean())
            target_depth, use_depth = original_depth, False
        else:
            global_q, axes, original_image, target, width, shoulder_count, original_depth, target_depth, use_depth = context
        value = constrain(corrections[i].copy())
        # Missing measurements retain the continuous correction. The explicit
        # user interval, rather than detection presence, controls activation.
        time = row['time']-times[0]
        envelope = min(1., max(0., (time-options.get('start', 0))/.15))
        if options.get('end', 0) > 0:
            envelope *= min(1., max(0., (options['end']-time)/.15))
        value *= envelope
        cumulative = Quaternion()
        for bone, v in zip(spine, value[:-3].reshape(-1, 3)):
            angle = float(np.linalg.norm(v))
            step = Quaternion(Vector(axes@v/angle), angle) if angle > 1e-10 else Quaternion()
            cumulative = cumulative@step
            set_global_rotation(bone, cumulative@global_q[bone.name])
            scratch.view_layers[0].update()
        angle = float(np.linalg.norm(value[-3:]))
        nq = Quaternion(Vector(axes@value[-3:]/angle), angle) if angle > 1e-10 else Quaternion()
        set_global_rotation(neck, nq@cumulative@global_q[neck.name])
        scratch.view_layers[0].update()
        # Preserve the captured/gaze head orientation while the spine carries it.
        set_global_rotation(head, global_q[head.name])
        scratch.view_layers[0].update()
        actual_points = np.array([b.head[:] for b in eyes]+[b.head[:] for b in shoulders])
        transformed = np.c_[actual_points,np.ones(4)]@np.asarray(camera[0]).T
        actual = projected(transformed[:,:3],camera)
        if not np.isfinite(actual).all():
            raise ValueError('Non-finite coordinated body trajectory')
        frame = channels[spine[0].name][i][0]
        for b in changed:
            q = b.rotation_quaternion.copy()
            if i and q.dot(Quaternion(channels[b.name][i-1][1])) < 0:
                q.negate()
            channels[b.name][i] = frame, tuple(q)
        report.append({'time': row['time'], 'active': bool(envelope>0),
            'observed': context is not None,'measurement_held': context is None,
            'model': 'coordinated-upper-body-relative-depth-v3', 'target_center': target.tolist(),
            'target_width': width, 'shoulder_count': shoulder_count,
            'before_center_px': float(np.linalg.norm(original_image[:2].mean(0)-target)*height),
            'after_center_px': float(np.linalg.norm(actual[:2].mean(0)-target)*height),
            'before_width_px': float(abs(np.linalg.norm(original_image[0]-original_image[1])-width)*height),
            'after_width_px': float(abs(np.linalg.norm(actual[0]-actual[1])-width)*height),
            'smoothing_seconds': .09,
            'relative_face_scale': float(ratios[i]), 'depth_active': bool(use_depth),
            'depth_weight': float(depth_weights[i]),
            'before_depth': original_depth, 'target_depth': target_depth,
            'after_depth': -float(transformed[:2,2].mean()),
            'reference_indices': reference,
            'spine_degrees': math.degrees(float(np.linalg.norm(value[:-3].reshape(-1,3),axis=1).sum())),
            'neck_degrees': math.degrees(float(np.linalg.norm(value[-3:]))),
            'limited': bool(np.linalg.norm(value[:-3].reshape(-1,3),axis=1).sum() > limit*.99
                            or np.linalg.norm(value[-3:]) > neck_limit*.99)})
        yield len(times)+i+1, 2*len(times)

"""Rotation-only MHR-to-MMD retargeting, preserving target rest geometry."""
import json
import math
from datetime import datetime

import bpy
from mathutils import Matrix, Quaternion, Vector

# Joint indices verified against MHR's parent tree and mhr70 keypoint mapping.
BODY = {'上半身': (34, None), '@UpperBody1': (35, None), '上半身2': (36, None),
        '首': (110, None), '頭': (113, None)}
for side, offset in [('右', 0), ('左', 36)]:
    BODY.update({side+'肩': (38+offset, None), side+'腕': (39+offset, 40+offset),
                 side+'ひじ': (40+offset, 41+offset), side+'手首': (42+offset, 52+offset)})
    for label, first in [('親指', 61), ('人指', 56), ('中指', 52), ('薬指', 48), ('小指', 44)]:
        for i in range(3):
            number = '０１２'[i] if label == '親指' else '１２３'[i]
            BODY[side+label+number] = (first+offset+i, first+offset+i+1)


def validate_take(data):
    if data.get('protocol') != 'SAM3D_BLENDER_TAKE' or data.get('version') != 1:
        raise ValueError('Unsupported SAM take format')
    fps = data.get('fps')
    if not isinstance(fps, (int, float)) or not math.isfinite(fps) or not 0 < fps <= 240:
        raise ValueError('Invalid source frame rate')
    rows = data.get('frames')
    if not isinstance(rows, list) or not 2 <= len(rows) <= 100000:
        raise ValueError('A motion take needs at least two samples')
    previous = -1
    for row in [data['reference']] + rows:
        if len(row['positions']) != 127 or len(row['rotations']) != 127:
            raise ValueError('Expected the 127-joint MHR skeleton')
        for position, rotation in zip(row['positions'], row['rotations']):
            if len(position) != 3 or len(rotation) != 3 or any(len(v) != 3 for v in rotation):
                raise ValueError('Invalid skeleton dimensions')
            if not all(math.isfinite(v) for v in position + [v for axis in rotation for v in axis]):
                raise ValueError('Nonfinite skeleton values')
        if 'time' in row:
            if not math.isfinite(row['time']) or row['time'] <= previous:
                raise ValueError('Sample times must increase')
            previous = row['time']
    return data


def bone_map(rig):
    names = {bone.mmd_bone.name_j: bone for bone in rig.pose.bones
             if getattr(bone, 'mmd_bone', None) and bone.mmd_bone.name_j}
    mapping = {}
    for name, pair in BODY.items():
        bone = rig.pose.bones.get(name[1:]) if name.startswith('@') else names.get(name)
        if bone:
            mapping[bone.name] = pair
    if not all(names.get(side+part) for side in ('左', '右') for part in ('腕', 'ひじ', '手首')):
        raise ValueError('Select the Ellen/MMD character armature, not the source skeleton')
    return dict(sorted(mapping.items(), key=lambda item: len(rig.pose.bones[item[0]].parent_recursive)))


def safe_pose_basis(bone, rows):
    """Do not preserve collapsed/nonfinite secondary-physics poses in backups.

    Older takes can contain these matrices even after their active action was
    repaired. Valid transforms (including deliberate body scaling) are kept.
    """
    basis = Matrix(rows)
    finite = all(math.isfinite(v) for row in basis for v in row)
    secondary = getattr(bone, 'wiggle_enable', False) or 'Hair' in bone.name
    collapsed = any(basis.to_3x3().col[i].length < .01 for i in range(3))
    if secondary and (not finite or collapsed):
        return Matrix.Identity(4)
    if not finite:
        raise ValueError('骨骼姿态包含无效数值：'+bone.name)
    return basis


def snapshot(rig, scene):
    anim = rig.animation_data
    return {'pose': {bone.name: {'basis': [list(row) for row in safe_pose_basis(bone, bone.matrix_basis)], 'mode': bone.rotation_mode}
                     for bone in rig.pose.bones},
            'action': anim.action.name if anim and anim.action else None,
            'slot': anim.action_slot.identifier if anim and anim.action_slot else None,
            'nla': {track.name: track.mute for track in anim.nla_tracks} if anim else {},
            'range': [scene.frame_start, scene.frame_end, scene.frame_current],
            'fps': [scene.render.fps, scene.render.fps_base]}


def safe_base_pose(rig, pose):
    return {name: {**row, 'basis': [list(axis) for axis in safe_pose_basis(rig.pose.bones[name], row['basis'])]}
            if name in rig.pose.bones else row for name, row in pose.items()}


def repair_secondary_action(rig, action):
    """Repair only invalid secondary transforms left by old take baselines."""
    if not action:
        return 0
    prefixes = tuple(b.path_from_id()+'.' for b in rig.pose.bones
                     if getattr(b, 'wiggle_enable', False) or 'Hair' in b.name)
    repaired = 0
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for fc in bag.fcurves:
                    if not fc.data_path.startswith(prefixes):
                        continue
                    prop = fc.data_path.rsplit('.', 1)[-1]
                    if prop not in ('location', 'scale', 'rotation_quaternion', 'rotation_euler'):
                        continue
                    default = 1. if prop == 'scale' or (prop == 'rotation_quaternion' and fc.array_index == 0) else 0.
                    changed = False
                    for key in fc.keyframe_points:
                        if not math.isfinite(key.co.y) or (prop == 'scale' and abs(key.co.y) < .01):
                            key.co.y = key.handle_left.y = key.handle_right.y = default
                            changed = True
                            repaired += 1
                    if changed:
                        fc.update()
    return repaired


def restore(rig, scene, state):
    # Validate before changing the action, including legacy JSON snapshots.
    poses = [(rig.pose.bones[name], row['mode'], safe_pose_basis(rig.pose.bones[name], row['basis']))
             for name, row in state['pose'].items() if name in rig.pose.bones]
    live = getattr(scene, 'wiggle_enable', None)
    if live:
        scene.wiggle_enable = False
    try:
        _restore_pose(rig, scene, state, poses)
    finally:
        if live:
            # Enabling simulation rebuilds and resets its cache from the
            # restored pose instead of reusing velocities from the SAM take.
            scene.wiggle_enable = True


def _restore_pose(rig, scene, state, poses):
    anim = rig.animation_data_create()
    anim.action = bpy.data.actions.get(state['action']) if state['action'] else None
    repair_secondary_action(rig, anim.action)
    if anim.action and state['slot']:
        slot = next((s for s in anim.action.slots if s.identifier == state['slot']), None)
        if slot:
            anim.action_slot = slot
    for track in anim.nla_tracks:
        if track.name in state['nla']:
            track.mute = state['nla'][track.name]
    for bone, mode, basis in poses:
        bone.rotation_mode = mode
        bone.matrix_basis = basis
    scene.render.fps, scene.render.fps_base = state['fps']
    scene.frame_start, scene.frame_end, frame = state['range']
    scene.frame_set(frame)


def put_curve(action, rig, bone, prop, values):
    if prop == 'rotation_quaternion':
        # q and -q are the same key pose, but component F-curves interpolate
        # through zero if their signs alternate. Align every committed series,
        # including poses produced by later image/IK correction passes.
        aligned=[];previous=None
        for frame,value in values:
            q=Quaternion(value).normalized()
            if previous is not None and q.dot(previous)<0:q.negate()
            aligned.append((frame,tuple(q)));previous=q.copy()
        values=aligned
    path = bone.path_from_id(prop)
    for index in range(len(values[0][1])):
        curve = action.fcurve_ensure_for_datablock(rig, path, index=index)
        curve.keyframe_points.add(len(values))
        curve.keyframe_points.foreach_set('co', [v for frame, value in values for v in (frame, value[index])])
        for key in curve.keyframe_points:
            key.interpolation = 'LINEAR'
        curve.update()


def contact_weight(time, options):
    start, end = options['start'], options['end']
    if time <= start or time >= end:
        return 0.0
    fade = min(options['fade'], (end-start)/2)
    t = min(1.0, (time-start)/fade, (end-time)/fade) if fade > 0 else 1.0
    return options['strength'] * t*t*(3-2*t)


def mouth_landmark(rig):
    """The original model's upper/lower central lip joints define contact."""
    head = next((b for b in rig.pose.bones if b.mmd_bone.name_j == '頭'), None)
    anchor = rig.get('sam_mouth_anchor')
    if head is not None and anchor is not None and len(anchor) == 3:
        return head.name, Vector(anchor)
    lips = [rig.pose.bones.get(n) for n in ('lip05Joint', 'lip14Joint')]
    if head is None or not all(lips):
        raise ValueError('This character has no supported mouth landmark')
    point = (lips[0].bone.head_local + lips[1].bone.head_local)*.5
    return head.name, head.bone.matrix_local.inverted() @ point


def contact_point(rig, side):
    fingers = [rig.pose.bones['IndexFinger2_'+side], rig.pose.bones['MiddleFinger2_'+side]]
    return (fingers[0].matrix.translation + fingers[1].matrix.translation)*.5


def mouth_target(rig, options):
    head, local = mouth_landmark(rig)
    # Ellen faces -Y. Offsets are in anatomical, head-relative axes (metres).
    rest_delta = Vector((options.get('x', 0), -.018-options.get('front', 0), options.get('z', 0)))
    local += rig.pose.bones[head].bone.matrix_local.to_3x3().inverted() @ rest_delta
    return rig.pose.bones[head].matrix @ local


def set_global_rotation(bone, rotation):
    parent = bone.parent
    kwargs = {'parent_matrix': parent.matrix,
              'parent_matrix_local': parent.bone.matrix_local} if parent else {}
    matrix = rotation.to_matrix().to_4x4()
    matrix.translation = bone.matrix.translation
    bone.rotation_quaternion = bone.bone.convert_local_to_pose(
        matrix, bone.bone.matrix_local, invert=True, **kwargs).to_quaternion().normalized()


def gaze_bones(rig):
    names = {b.mmd_bone.name_j: b for b in rig.pose.bones if getattr(b, 'mmd_bone', None)}
    if not all(name in names for name in ('頭', '首', '左目', '右目', '両目')):
        raise ValueError('此人物缺少头部或眼睛控制骨骼')
    return {name: names[name] for name in ('頭', '首', '左目', '右目', '両目')}

def camera_samples(scene, target_rig, camera, data):
    """Read evaluated camera/character transforms without advancing physics."""
    frame, subframe = scene.frame_current, scene.frame_subframe
    wiggle = getattr(scene, 'wiggle_enable', None)
    samples = []
    try:
        if wiggle is not None:
            scene.wiggle_enable = False
        for row in data['frames']:
            from sam_ellen_sequence import timeline_frame
            time = timeline_frame(data, row['time'])
            scene.frame_set(int(time), subframe=time-int(time))
            depsgraph = bpy.context.evaluated_depsgraph_get()
            world = target_rig.evaluated_get(depsgraph).matrix_world
            position = camera.evaluated_get(depsgraph).matrix_world.translation
            samples.append(world.inverted() @ position)
        return samples
    finally:
        scene.frame_set(frame, subframe=subframe)
        if wiggle is not None:
            scene.wiggle_enable = wiggle


def gaze_drift(time, speed=.12, seed=17, randomness=1.0):
    """Bounded C2 random drift; a fixed seed makes scrubbing/rendering stable."""
    def noise(position, salt):
        index = math.floor(position)
        fraction = position-index
        blend = fraction**3*(fraction*(fraction*6-15)+10)

        def value(cell):
            number = (cell*0x9E3779B1 + int(seed)*0x85EBCA77 + salt) & 0xFFFFFFFF
            number = ((number ^ (number >> 16))*0x7FEB352D) & 0xFFFFFFFF
            number = ((number ^ (number >> 15))*0x846CA68B) & 0xFFFFFFFF
            return 2*((number ^ (number >> 16))/0xFFFFFFFF)-1

        return (1-blend)*value(index)+blend*value(index+1)

    phase = time*speed*3
    # Smooth time warping varies turn intervals, without frame-wise randomness.
    phase += .4*noise(phase*.37, 113)
    random_value = (.65*noise(phase, 271)+.25*noise(phase*.47, 521)
                    + .10*noise(phase*.19, 941))
    regular_phase = time*speed*2*math.pi
    regular_value = .70*math.sin(regular_phase+.7)+.30*math.sin(regular_phase*.43+2.1)
    return (1-randomness)*regular_value+randomness*random_value


def smooth_rotations(samples, fps, seconds):
    """Centered Gaussian quaternion mean: suppress jitter without causal lag.

    Time-cell weights keep irregularly sampled captures from biasing the mean.
    Quaternion signs are aligned to each center to avoid q/-q cancellation.
    """
    if seconds <= 0 or len(samples) < 3:
        return list(samples)
    times = [frame/fps for frame, _ in samples]
    rotations = [Quaternion(q).normalized() for _, q in samples]
    widths = [(times[1]-times[0])/2]
    widths.extend((times[i+1]-times[i-1])/2 for i in range(1, len(times)-1))
    widths.append((times[-1]-times[-2])/2)
    result = []
    left, right = 0, 0
    for i, time in enumerate(times):
        while times[left] < time-3*seconds:
            left += 1
        while right < len(times) and times[right] <= time+3*seconds:
            right += 1
        center = rotations[i]
        mean = [0., 0., 0., 0.]
        for j in range(left, right):
            weight = math.exp(-.5*((times[j]-time)/seconds)**2)*widths[j]
            if center.dot(rotations[j]) < 0:
                weight = -weight
            for axis in range(4):
                mean[axis] += weight*rotations[j][axis]
        q = Quaternion(mean).normalized()
        if result and q.dot(Quaternion(result[-1][1])) < 0:
            q.negate()
        result.append((samples[i][0], tuple(q)))
    return result


def body_bones(rig):
    names = {b.mmd_bone.name_j: b for b in rig.pose.bones if getattr(b, 'mmd_bone', None)}
    bones = [names.get('上半身'), rig.pose.bones.get('UpperBody1'), names.get('上半身2')]
    return list(dict.fromkeys(b for b in bones if b))


def correct_body(channels, rig, data, options):
    """Small additive torso rotations; roots/hips/legs never receive offsets."""
    bones = body_bones(rig)
    times = [row['time']-data['frames'][0]['time'] for row in data['frames']]
    originals = {b.name: list(channels[b.name]) for b in bones}
    # Measure actual captured torso activity over a time window, not frame noise.
    smoothed = {name: smooth_rotations(values, data['fps'], .25)
                for name, values in originals.items()}
    left, right = 0, 0
    weights = [1/len(bones)]*len(bones)
    amplitude = math.radians(options['amplitude'])*options['strength']
    seed, speed = options['seed'], options['speed']
    for i, time in enumerate(times):
        while left < i and times[left] < time-.3:
            left += 1
        while right+1 < len(times) and times[right] < time+.3:
            right += 1
        span = times[right]-times[left]
        velocity = 0.
        for values in smoothed.values():
            q1, q2 = Quaternion(values[left][1]), Quaternion(values[right][1])
            velocity += 2*math.acos(min(1., abs(q1.dot(q2))))/max(span, 1e-6)
        retained = 1/(1+(velocity/math.radians(12))**2)
        roll = gaze_drift(time, speed, seed, options['randomness'])
        yaw = .55*gaze_drift(time, speed*.83, seed+101, options['randomness'])
        pitch = .25*gaze_drift(time, speed*.67, seed+211, options['randomness'])
        for bone, weight in zip(bones, weights):
            rest = bone.bone.matrix_local.to_quaternion().inverted()
            delta = (Quaternion(rest @ Vector((0, 0, 1)), yaw*amplitude*weight*retained)
                     @ Quaternion(rest @ Vector((0, 1, 0)), roll*amplitude*weight*retained)
                     @ Quaternion(rest @ Vector((1, 0, 0)), pitch*amplitude*weight*retained))
            frame, value = originals[bone.name][i]
            channels[bone.name][i] = (frame, tuple((Quaternion(value) @ delta).normalized()))
        yield i+1, len(times)


def correct_gaze(scratch, rig, channels, data, options, cameras, eye_base=None, sample_pose=None,
                 independent_eyes=False):
    from sam_ellen_gaze import merged_ranges, contact_weight
    from sam_ellen_sequence import timeline_frame
    away_ranges = merged_ranges(options.get('away_ranges', []))
    away_fade = options.get('away_fade', 0)
    bones = gaze_bones(rig)
    head, neck = bones['頭'], bones['首']
    eyes = [bones['左目'], bones['右目']]
    eye_control = bones['両目']
    original = {name: list(values) for name, values in channels.items()}
    # Eye contact is independent from head motion. Missing mode in older
    # settings also preserves capture; forced head tracking is opt-in.
    track_head = options.get('head_tracking', False)
    changed = [eye_control, *eyes]
    if track_head:
        changed = [head, neck, *changed]
    baseline = {bone.name: bone.rotation_quaternion.copy() for bone in changed}
    for name, base in (eye_base or {}).items():
        if name in baseline:
            baseline[name] = Matrix(base['basis']).to_quaternion()
    for bone in changed:
        channels.setdefault(bone.name, [])
    head_rest = head.bone.matrix_local.to_quaternion()
    center = sum((b.bone.head_local for b in eyes), Vector())*.5
    eye_center_local = head.bone.matrix_local.inverted() @ center
    previous = {}

    def store_sample(bone, i, frame, q):
        if bone.name in previous and q.dot(previous[bone.name]) < 0:
            q.negate()
        previous[bone.name] = q.copy()
        sample = (frame, tuple(q))
        if bone.name in original:
            channels[bone.name][i] = sample
        else:
            channels[bone.name].append(sample)
    noise = options.get('noise_enabled', False)
    noise_amplitude = math.radians(max(0.0, min(12.0, options.get('noise_amplitude', 0.0))))
    noise_speed = max(.05, min(2.0, options.get('noise_speed', .28)))
    noise_head = max(0.0, min(1.0, options.get('noise_head', .35)))
    noise_eyes = max(0.0, min(1.0, options.get('noise_eyes', .75)))
    noise_seed = options.get('noise_seed', 17)
    noise_randomness = options.get('noise_randomness', .75)
    center_rotation = Quaternion()
    if track_head and options.get('center_motion'):
        # Remove the take's static head bias, retaining its frame-to-frame motion.
        reference_q = None
        summed = Vector((0., 0., 0., 0.))
        for i in range(len(data['frames'])):
            for name, samples in original.items():
                rig.pose.bones[name].rotation_quaternion = samples[i][1]
            scratch.view_layers[0].update()
            body = neck.parent
            delta = body.matrix.to_quaternion() @ body.bone.matrix_local.to_quaternion().inverted()
            q = (delta @ head_rest).inverted() @ head.matrix.to_quaternion()
            if reference_q is None:
                reference_q = q.copy()
            if q.dot(reference_q) < 0:
                q.negate()
            summed += Vector(tuple(q))
        center_rotation = Quaternion(tuple(summed)).normalized()
    for i, row in enumerate(data['frames']):
        if sample_pose:
            sample_pose(i)
        weight = contact_weight(timeline_frame(data, row['time']), away_ranges, away_fade)
        frame = 1+(row['time']-data['frames'][0]['time'])*data['fps']
        elapsed = row['time']-data['frames'][0]['time']
        noise_yaw = noise_amplitude*gaze_drift(elapsed, noise_speed, noise_seed, noise_randomness) if noise else 0.0
        for name, samples in original.items():
            rig.pose.bones[name].rotation_quaternion = samples[i][1]
        for bone in changed:
            if bone.name not in original:
                bone.rotation_quaternion = baseline[bone.name]
        captured = {bone.name: bone.rotation_quaternion.copy() for bone in changed}
        scratch.view_layers[0].update()
        if weight == 0:
            for bone in changed:
                store_sample(bone, i, frame, captured[bone.name])
            yield i+1, len(data['frames'])
            continue
        # Neutralize the MMD shared-eye additive controller only in the new take.
        eye_control.rotation_quaternion = Quaternion()
        scratch.view_layers[0].update()
        if track_head:
            captured_head = head.matrix.to_quaternion()
            body = neck.parent
            body_delta = body.matrix.to_quaternion() @ body.bone.matrix_local.to_quaternion().inverted()
            residual = (body_delta @ head_rest).inverted() @ captured_head
            if options.get('center_motion'):
                residual = center_rotation.inverted() @ residual
            if residual.w < 0:
                residual.negate()
            retained = Quaternion().slerp(residual, options['motion'])
            if retained.angle > math.radians(45):
                retained = Quaternion().slerp(retained, math.radians(45)/retained.angle)
            neck.rotation_quaternion = Quaternion().slerp(neck.rotation_quaternion.normalized(), options['motion'])
            for _ in range(3):
                eye_center = head.matrix @ eye_center_local
                direction = body_delta.inverted() @ (cameras[i]-eye_center)
                if direction.length < 1e-6:
                    raise ValueError('摄像机与人物眼睛重合，请拉开距离')
                yaw = max(-math.radians(50), min(math.radians(50), math.atan2(direction.x, -direction.y)))
                pitch = max(-math.radians(35), min(math.radians(35), math.atan2(direction.z, math.hypot(direction.x, direction.y))))
                head_yaw = yaw + noise_yaw*noise_head
                limited = Vector((math.sin(head_yaw)*math.cos(pitch), -math.cos(head_yaw)*math.cos(pitch), math.sin(pitch)))
                desired = body_delta @ limited.to_track_quat('-Y', 'Z') @ head_rest @ retained
                set_global_rotation(head, desired)
                scratch.view_layers[0].update()
        head_delta = head.matrix.to_quaternion() @ head_rest.inverted()
        for bone in eyes:
            forward = head_delta @ Vector((0, -1, 0))
            neutral = head_delta @ bone.bone.matrix_local.to_quaternion()
            direction = cameras[i]-bone.matrix.translation
            if direction.length < 1e-6:
                raise ValueError('摄像机与人物眼睛重合，请拉开距离')
            if noise:
                # Rotate the eye target itself, so the eye still has a precise
                # target and does not accumulate a separate post-rotation drift.
                # The head uses its smaller ratio above; the eyes can follow more.
                direction = Quaternion(Vector((0, 0, 1)), noise_yaw*noise_eyes) @ direction
            swing = forward.rotation_difference(direction.normalized())
            # Flat anime irises leave their eye sockets with large rotations.
            eye_limit = math.radians(options.get('eye_limit_degrees', 20))
            if swing.angle > eye_limit:
                swing = Quaternion().slerp(swing, eye_limit/swing.angle)
            swing = Quaternion().slerp(swing, options['eyes'])
            set_global_rotation(bone, swing @ neutral)
            if not independent_eyes:
                scratch.view_layers[0].update()
        for bone in changed:
            q = bone.rotation_quaternion.copy()
            if weight < 1:
                q = captured[bone.name].slerp(q, weight)
            store_sample(bone, i, frame, q)
        yield i+1, len(data['frames'])


def continuous_contacts(scratch, rig, channels, data, side, goal_provider, anatomical=False):
    """Solve a two-link arm with an explicit, continuous elbow bend plane."""
    arm, elbow, wrist = [rig.pose.bones[n+'_'+side] for n in ('Arm', 'Elbow', 'Wrist')]
    twist=rig.pose.bones.get('HandTwist_'+side) if anatomical else None
    if twist and twist.name not in channels:
        channels[twist.name]=[(f,tuple(Quaternion())) for f,_ in channels[arm.name]]
    originals = {name: list(values) for name, values in channels.items()}
    previous = {}
    view = scratch.view_layers[0]
    previous_hand=None
    hinge_state={}
    for i, row in enumerate(data['frames']):
        for name, samples in originals.items():
            rig.pose.bones[name].rotation_quaternion = samples[i][1]
        view.update()
        result = goal_provider(i)
        if result is None:
            previous_hand=None;hinge_state={}
        if result is not None:
            goal, wrist_rotation, bend = result
            a, e, w = arm.head.copy(), elbow.head.copy(), wrist.head.copy()
            upper, lower = (e-a).length, (w-e).length
            vector = goal-a
            direction = vector.normalized() if vector.length > 1e-7 else (w-a).normalized()
            minimum=(math.sqrt(upper**2+lower**2-2*upper*lower*math.cos(math.radians(35)))
                     if anatomical else abs(upper-lower)+1e-5)
            maximum=(math.sqrt(upper**2+lower**2-2*upper*lower*math.cos(math.radians(165)))
                     if anatomical else upper+lower-1e-5)
            distance = max(minimum, min(maximum, vector.length))
            goal = a+direction*distance
            bend = bend-direction*bend.dot(direction)
            if bend.length < 1e-7:
                bend = direction.orthogonal()
            along = (distance*distance+upper*upper-lower*lower)/(2*distance)
            elbow_goal = a+direction*along+bend.normalized()*math.sqrt(max(0., upper*upper-along*along))
            if anatomical:
                from sam_ellen_arm_chain import anatomical_rotations
                if previous_hand is not None:
                    dot=min(1,abs(previous_hand.dot(wrist_rotation.normalized())))
                    angle=2*math.acos(dot)
                    dt=row['time']-data['frames'][i-1]['time'] if i else 1/data['fps']
                    limit=math.radians(900*dt)
                    if angle>limit:wrist_rotation=previous_hand.slerp(wrist_rotation,limit/angle)
                anatomical_rotations(arm,elbow,wrist,a,elbow_goal,goal,wrist_rotation,set_global_rotation,view,hinge_state)
                previous_hand=wrist.matrix.to_quaternion().copy()
                if (wrist.head-goal).length>.002 and twist:
                    twist.rotation_quaternion=Quaternion();view.update()
                    set_global_rotation(wrist,wrist_rotation);view.update()
                if (wrist.head-goal).length>.002:
                    raise ValueError('Anatomical arm solver did not reach its target')
                for bone in (arm,elbow,wrist,*([twist] if twist else [])):
                    q=bone.rotation_quaternion.copy()
                    if bone.name in previous and q.dot(previous[bone.name])<0:q.negate()
                    previous[bone.name]=q.copy();channels[bone.name][i]=channels[bone.name][i][0],tuple(q)
                yield i+1,len(data['frames'])
                continue
            upper_rotation = (e-a).rotation_difference(elbow_goal-a)@arm.matrix.to_quaternion()
            set_global_rotation(arm, upper_rotation)
            view.update()
            # The elbow's lower-link vector has already inherited the upper-arm
            # turn. Solve from this updated direction, not the pre-turn vector.
            elbow_now = elbow.head.copy()
            wrist_now = wrist.head.copy()
            lower_rotation = (wrist_now-elbow_now).rotation_difference(goal-elbow_goal)@elbow.matrix.to_quaternion()
            set_global_rotation(elbow, lower_rotation)
            view.update()
            set_global_rotation(wrist, wrist_rotation)
            view.update()
            if (wrist.head-goal).length > .002:
                raise ValueError('Stable arm solver did not reach its target')
        for bone in (arm, elbow, wrist):
            q = bone.rotation_quaternion.copy()
            if bone.name in previous and q.dot(previous[bone.name]) < 0:
                q.negate()
            previous[bone.name] = q.copy()
            channels[bone.name][i] = channels[bone.name][i][0], tuple(q)
        yield i+1, len(data['frames'])


def correct_contacts(scratch, rig, channels, data, options, goal_provider=None):
    """Use Blender IK on a disposable two-link rig, keeping wrist orientation."""
    if options.get('stable_ik') and goal_provider:
        yield from continuous_contacts(scratch, rig, channels, data, options['side'], goal_provider,
                                       anatomical=options.get('anatomical',False))
        return
    side = options['side']
    arm, elbow, wrist = [rig.pose.bones[n+'_'+side] for n in ('Arm', 'Elbow', 'Wrist')]
    arm_data = bpy.data.armatures.new('SAM Contact Solver')
    solver = bpy.data.objects.new('SAM Contact Solver', arm_data)
    target = bpy.data.objects.new('SAM Contact Goal', None)
    scratch.collection.objects.link(solver)
    scratch.collection.objects.link(target)
    view = scratch.view_layers[0]
    try:
        with bpy.context.temp_override(scene=scratch, view_layer=view, object=solver,
                                       active_object=solver, selected_objects=[solver],
                                       selected_editable_objects=[solver]):
            view.objects.active = solver
            solver.select_set(True, view_layer=view)
            bpy.ops.object.mode_set(mode='EDIT')
            upper = arm_data.edit_bones.new('Upper')
            upper.head, upper.tail = arm.bone.head_local, elbow.bone.head_local
            upper.align_roll(arm.bone.matrix_local.to_3x3().col[2])
            lower = arm_data.edit_bones.new('Lower')
            lower.head, lower.tail = elbow.bone.head_local, wrist.bone.head_local
            lower.align_roll(elbow.bone.matrix_local.to_3x3().col[2])
            lower.parent, lower.use_connect = upper, True
            bpy.ops.object.mode_set(mode='OBJECT')
        ik = solver.pose.bones['Lower'].constraints.new('IK')
        ik.target, ik.chain_count, ik.use_stretch, ik.iterations = target, 2, False, 64
        for b in solver.pose.bones:
            b.ik_stretch = 0
        changed = (arm.name, elbow.name, wrist.name)
        originals = {name: list(values) for name, values in channels.items()}
        previous = {}
        for i, row in enumerate(data['frames']):
            weight = 1 if goal_provider else contact_weight(row['time']-data['frames'][0]['time'], options)
            if weight > 0:
                for name, samples in originals.items():
                    rig.pose.bones[name].rotation_quaternion = samples[i][1]
                view.update()
                wrist_rotation = wrist.matrix.to_quaternion()
                goal = (goal_provider(i) if goal_provider else
                        wrist.matrix.translation + (mouth_target(rig, options)-contact_point(rig, side))*weight)
                if goal is None:
                    yield i+1, len(data['frames'])
                    continue
                if isinstance(goal, tuple):
                    goal, wrist_rotation = goal
                target.location = goal
                solver.location = arm.matrix.translation - arm.bone.head_local
                ik.mute = True
                view.update()
                for helper, source in zip(('Upper', 'Lower'), (arm, elbow)):
                    matrix = source.matrix.copy()
                    matrix.translation -= solver.location
                    solver.pose.bones[helper].matrix = matrix
                ik.mute = False
                view.update()
                for bone, helper in ((arm, 'Upper'), (elbow, 'Lower'), (wrist, None)):
                    desired = (solver.pose.bones[helper].matrix.to_quaternion() if helper else wrist_rotation)
                    parent = bone.parent
                    kwargs = {'parent_matrix': parent.matrix,
                              'parent_matrix_local': parent.bone.matrix_local} if parent else {}
                    matrix = desired.to_matrix().to_4x4()
                    matrix.translation = bone.matrix.translation
                    local = bone.bone.convert_local_to_pose(matrix, bone.bone.matrix_local, invert=True, **kwargs)
                    bone.rotation_quaternion = local.to_quaternion().normalized()
                    view.update()
                for name in changed:
                    frame = channels[name][i][0]
                    channels[name][i] = (frame, tuple(rig.pose.bones[name].rotation_quaternion))
            for name in changed:
                frame, value = channels[name][i]
                q = Quaternion(value)
                if name in previous and q.dot(previous[name]) < 0:
                    q.negate()
                previous[name] = q.copy()
                channels[name][i] = (frame, tuple(q))
            yield i+1, len(data['frames'])
    finally:
        bpy.data.objects.remove(solver, do_unlink=True)
        bpy.data.objects.remove(target, do_unlink=True)
        bpy.data.armatures.remove(arm_data)


def bake_steps(scene, rig, data, smoothing=0.0, capture_mode='BOTH', inactive_pose='DOWN', contact=None, gaze=None, screen=None, punch=None, head_smoothing=0.0, body_support=None, base_pose=None, torso_capture=None, head_pose=None):
    """Yield progress while solving; return a new Action on generator completion."""
    validate_take(data)
    if data.get('screen'):
        from sam_ellen_hand_tracking import prepare as prepare_tracking
        prepare_tracking(data['screen'])
    mode=(screen or {}).get('camera_mode','LEGACY')
    if mode=='HANDHELD':
        raise ValueError('手持机位尚需背景/镜头运动输入；目前不能可靠分离人物与相机，请使用固定机位视频')
    if mode not in ('FIXED','LEGACY'):raise ValueError('未知拍摄方式')
    fixed=mode=='FIXED' and screen.get('space')=='FRAME'
    if fixed and data.get('_sam_timeline'):
        raise ValueError('固定机位位移目前仅支持整段重新生成，追加片段需独立转场与地面衔接；已保留原动作')
    location_channels={}
    if fixed:
        screen=dict(screen, unified_arms=True)
    if not math.isfinite(head_smoothing) or not 0 <= head_smoothing <= .5:
        raise ValueError('Invalid head smoothing time')
    if capture_mode not in ('BOTH', 'SELFIE_LEFT', 'SELFIE_RIGHT', 'UPPER_BOTH', 'PHONE_LEFT', 'PHONE_RIGHT') or inactive_pose not in ('DOWN', 'CURRENT'):
        raise ValueError('Unsupported selfie preset')
    mapping = bone_map(rig)
    grip = {}
    if capture_mode in ('PHONE_LEFT', 'PHONE_RIGHT'):
        from sam_ellen_phone import finger_rotations
        grip = finger_rotations(rig, 'L' if capture_mode == 'PHONE_LEFT' else 'R')
        if screen: screen = dict(screen, finger_articulation=False)
    if screen:
        if not math.isfinite(screen['strength']) or not 0 <= screen['strength'] <= 1:
            raise ValueError('Invalid screen alignment strength')
        if screen['strength'] == 0:
            screen = None
    coordinated = bool(screen and screen.get('coordinated_body') and screen.get('space') == 'FRAME'
                       and screen.get('landmarks'))
    if coordinated and sum(bool(r.get('face')) for r in data.get('screen', {}).get('frames', [])) < 2:
        coordinated = False
        screen = dict(screen, coordinated_body=False, calibrate_camera=False)
    if fixed and (not coordinated or not screen.get('calibrate_camera')):
        raise ValueError('固定机位需要开启整体身体对齐、一次基础构图校准，以及足够的可见脸部观测')
    if coordinated:
        if not math.isfinite(screen.get('body_limit', 25)) or not 1 <= screen.get('body_limit', 25) <= 25:
            raise ValueError('Invalid coordinated body angle limit')
        torso_capture = dict(torso_capture or {'strength': .75, 'screen_strength': .65,
            'max_angle': 15., 'smoothing': .12, 'camera': screen.get('camera', '')})
        if screen.get('calibrate_camera') and not data.get('_sam_timeline'):
            selected = bpy.data.objects.get(screen.get('camera', '')) or scene.camera
            original_camera = bpy.data.objects.get(selected.get('sam_original_camera', '')) if selected else None
            if original_camera:
                # Reapplying starts from the retained baseline, never the last result.
                screen = dict(screen, camera=original_camera.name)
                torso_capture = dict(torso_capture, camera=original_camera.name)
                if gaze:
                    gaze = dict(gaze, camera=original_camera.name)
                if head_pose:
                    head_pose = dict(head_pose, camera=original_camera.name)
    torso_cameras, torso_report = None, []
    if torso_capture:
        for key, low, high in (('strength', 0, 1), ('screen_strength', 0, 1),
                               ('max_angle', 1, 25), ('smoothing', .01, .5)):
            value = torso_capture[key]
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError('Invalid captured torso settings')
        if not body_bones(rig):
            raise ValueError('此人物缺少上半身控制骨骼')
        from sam_ellen_screen import camera_frames
        camera = bpy.data.objects.get(torso_capture.get('camera', '')) or scene.camera
        if camera is None or camera.type != 'CAMERA':
            raise ValueError('请选择人物画面的摄像机')
        torso_cameras = camera_frames(scene, rig, camera, data, observation_space=True)
    if body_support:
        for key, low, high in (('strength', 0, 1), ('amplitude', 0, 6),
                               ('speed', .05, .5), ('randomness', 0, 1)):
            value = body_support[key]
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError('Invalid body sway settings')
        if type(body_support['seed']) is not int or not 0 <= body_support['seed'] <= 1000000:
            raise ValueError('Invalid body sway seed')
        if not body_bones(rig):
            raise ValueError('此人物缺少上半身控制骨骼')
    cameras = None
    if gaze:
        from sam_ellen_gaze import merged_ranges
        merged_ranges(gaze.get('away_ranges', []))
        if type(gaze.get('away_fade', 0)) is not int or not 0 <= gaze.get('away_fade', 0) <= 120:
            raise ValueError('不看镜头过渡帧数必须在 0–120 之间')
        camera = bpy.data.objects.get(gaze.get('camera', '')) if gaze.get('camera') else scene.camera
        if camera is None or camera.type != 'CAMERA':
            raise ValueError('请先选择目标摄像机')
        if not all(math.isfinite(gaze[k]) and 0 <= gaze[k] <= 1 for k in ('motion', 'eyes')):
            raise ValueError('Invalid gaze settings')
        for key, low, high in (('noise_amplitude', 0, 12),
                               ('noise_speed', .05, 2), ('noise_head', 0, 1),
                               ('noise_eyes', 0, 1), ('noise_randomness', 0, 1)):
            value = gaze.get(key, 0 if key == 'noise_amplitude' else 1)
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError('Invalid gaze noise settings')
        seed = gaze.get('noise_seed', 17)
        if type(seed) is not int or not 0 <= seed <= 1000000:
            raise ValueError('Invalid gaze random seed')
        gaze_bones(rig)
        cameras = camera_samples(scene, rig, camera, data)
    screen_cameras = None
    screen_report = []
    head_report, head_cameras = [], None
    if head_pose:
        from sam_ellen_screen import camera_frames
        camera = bpy.data.objects.get(head_pose.get('camera', '')) or scene.camera
        if camera is None or camera.type != 'CAMERA':
            raise ValueError('请选择头部捕捉摄像机')
        head_cameras = camera_frames(scene, rig, camera, data, observation_space=True)
    punch_cameras = None
    punch_report = []
    if punch:
        from sam_ellen_punch import validate_punch
        from sam_ellen_screen import camera_frames
        validate_punch(data, punch, capture_mode)
        camera = bpy.data.objects.get(punch.get('camera', '')) if punch.get('camera') else scene.camera
        if camera is None or camera.type != 'CAMERA':
            raise ValueError('请选择出拳目标摄像机')
        punch_cameras = camera_frames(scene, rig, camera, data, observation_space=True)
    if screen:
        from sam_ellen_screen import validate_screen, camera_frames
        validate_screen(data, screen['side'], screen.get('space', 'FACE'), coordinated)
        if screen.get('head_depth_enabled'):
            from sam_ellen_gestures import interval_weight
            interval_weight(0, screen['head_depth_start'], screen['head_depth_end'])
            if not math.isfinite(screen['head_depth_offset']) or abs(screen['head_depth_offset']) > .2:
                raise ValueError('Invalid head depth offset')
        if not all(math.isfinite(screen[k]) for k in ('strength', 'depth', 'front')) or not all(0 <= screen[k] <= 1 for k in ('strength', 'depth')):
            raise ValueError('Invalid screen alignment settings')
        if (not all(math.isfinite(screen.get(k, 0)) for k in ('start', 'end'))
                or screen.get('start', 0) < 0 or screen.get('end', 0) < 0
                or screen.get('end', 0) > 0 and screen['end'] <= screen.get('start', 0)):
            raise ValueError('画面对齐结束时间必须晚于开始时间')
        camera = bpy.data.objects.get(screen.get('camera', '')) if screen.get('camera') else scene.camera
        if camera is None or camera.type != 'CAMERA':
            raise ValueError('请选择画面对齐的目标摄像机')
        if capture_mode == 'SELFIE_LEFT' and screen['side'] != 'L' or capture_mode == 'SELFIE_RIGHT' and screen['side'] != 'R':
            raise ValueError('画面对齐手与自拍预设不一致')
        screen_cameras = camera_frames(scene, rig, camera, data, observation_space=True)
    passes = 1+bool(contact)+bool(gaze)+bool(screen)+bool(punch)+bool(body_support)+bool(torso_capture)+coordinated+bool(head_pose)
    names_j = {b.name: b.mmd_bone.name_j for b in rig.pose.bones}
    active_side = {'SELFIE_LEFT': '左', 'SELFIE_RIGHT': '右'}.get(capture_mode)
    if contact:
        if (contact['side'] not in ('L', 'R') or contact['end'] <= contact['start']
                or not 0 <= contact['strength'] <= 1 or contact['fade'] < 0
                or contact['start'] < 0 or not all(math.isfinite(contact.get(k, 0))
                    for k in ('start', 'end', 'fade', 'strength', 'x', 'front', 'z'))):
            raise ValueError('Invalid mouth correction settings')
        if active_side and contact['side'] != ('L' if active_side == '左' else 'R'):
            raise ValueError('捂嘴手与自拍预设不一致，请选择同一只手')
        mouth_landmark(rig)
    inactive = set()
    if active_side:
        inactive = {name for name in mapping if names_j[name].startswith(('左', '右'))
                    and not names_j[name].startswith(active_side)}
        # Torso capture is an explicit opt-in for cropped selfies.
        mapping = {name: pair for name, pair in mapping.items()
                   if torso_capture or names_j[name] in ('首', '頭') or names_j[name].startswith(('左', '右'))}
        if inactive_pose == 'CURRENT':
            mapping = {name: pair for name, pair in mapping.items() if name not in inactive}
    state = snapshot(rig, scene)
    original_scene_camera = scene.camera
    render_camera_binding = None
    render_camera = None
    calibrated = None
    calibration_report = None
    eye_base = {}
    current_action = rig.animation_data.action if rig.animation_data else None
    stable_pose = json.loads(current_action.get('sam_camera_match_base', 'null')) if current_action else None
    # Older take metadata predates newly rebuilt MMD costume helper bones.
    # Keep its original base for existing bones, and initialize new helpers
    # from the current repaired rig rather than failing at action commit.
    pose_base = safe_base_pose(rig, {**state['pose'], **(base_pose or stable_pose or {})})
    torso_base = json.loads(current_action.get('sam_torso_base', '{}')) if current_action else {}
    if torso_capture and not torso_base:
        torso_base = {b.name: pose_base[b.name] for b in body_bones(rig)}
    body_base = json.loads(current_action.get('sam_body_base', '{}')) if current_action else {}
    if body_support and not body_base:
        body_base = {b.name: pose_base[b.name] for b in body_bones(rig)}
    if current_action and current_action.get('sam_gaze_eye_base'):
        eye_base = json.loads(current_action['sam_gaze_eye_base'])
    if gaze and not eye_base:
        eye_base = {b.name: pose_base[b.name] for key, b in gaze_bones(rig).items()
                    if key in ('両目', '左目', '右目')}
    target_rig = rig
    if screen and screen.get('space')=='FRAME' and screen.get('landmarks') and screen.get('chin_contact',True):
        from sam_ellen_chin_contact import chin_anchor
        screen=dict(screen,chin_anchor=chin_anchor(target_rig))
    scratch = bpy.data.scenes.new('SAM Retarget Scratch')
    rig = target_rig.copy()
    scratch.collection.objects.link(rig)
    rig.parent = None
    rig.matrix_world = Matrix.Identity(4)
    rig.animation_data_clear()
    if base_pose or stable_pose:
        for name, base in pose_base.items():
            if name in rig.pose.bones:
                rig.pose.bones[name].matrix_basis = Matrix(base['basis'])
    # Start reapplication from the pre-assistance pose; never accumulate sway.
    for name, base in body_base.items():
        if name in rig.pose.bones:
            rig.pose.bones[name].matrix_basis = Matrix(base['basis'])
    for bone in rig.pose.bones:
        phone_constraint = bone.constraints.get('SAM Phone Grip')
        if phone_constraint:
            phone_constraint.mute = True
        for constraint in bone.constraints:
            if getattr(constraint, 'target', None) == target_rig:
                constraint.target = rig
    scratch.view_layers[0].update()
    wiggle = getattr(scene, 'wiggle_enable', None)
    ground_reference={side:float(rig.pose.bones['AnkleD_'+side].head.z) for side in ('L','R')
        if 'AnkleD_'+side in rig.pose.bones and 'LegIK_'+side in rig.pose.bones} if fixed else {}
    channels = {name: [] for name in mapping}
    reference = [Matrix(value) for value in data['reference']['rotations']]
    previous = {}
    action = None
    committed = False
    try:
        for bone in rig.pose.bones:
            bone.rotation_mode = 'QUATERNION'
        for i, row in enumerate(data['frames']):
            positions = [Vector(p) for p in row['positions']]
            cache = {}
            evaluated_helpers = False

            def pose_matrix(bone):
                if bone.name not in cache:
                    kwargs = {'parent_matrix': pose_matrix(bone.parent),
                              'parent_matrix_local': bone.parent.bone.matrix_local} if bone.parent else {}
                    cache[bone.name] = bone.bone.convert_local_to_pose(
                        bone.matrix_basis, bone.bone.matrix_local, **kwargs)
                return cache[bone.name]

            for name, (source, child) in mapping.items():
                bone = rig.pose.bones[name]
                if child is not None and not evaluated_helpers:
                    # Only shoulder-cancellation helpers need dependency-graph
                    # evaluation. Remaining FK uses Blender's own bone API.
                    scratch.view_layers[0].update()
                    cache = {b.name: b.matrix.copy() for b in rig.pose.bones
                             if any(not c.mute and c.influence for c in b.constraints)}
                    evaluated_helpers = True
                # Global delta retains the source's twist. Swing alignment removes
                # A/T-pose offsets without scaling any target limb or moving roots.
                desired = Matrix(row['rotations'][source]) @ reference[source].transposed() @ bone.bone.matrix_local.to_3x3()
                if name in inactive:
                    # Neutral fingers and a lowered arm, never the unseen SAM arm.
                    desired = bone.bone.matrix_local.to_3x3().copy()
                    part = names_j[name][1:]
                    direction = None
                    sign = 1 if names_j[name].startswith('左') else -1
                    if part == '腕':
                        direction = Vector((sign*.18, 0, -1))
                    elif part in ('ひじ', '手首'):
                        direction = Vector((sign*.08, -.15, -1))
                    if direction is not None:
                        desired = (desired @ Vector((0, 1, 0))).rotation_difference(direction).to_matrix() @ desired
                elif active_side and names_j[name].endswith('肩'):
                    desired = bone.bone.matrix_local.to_3x3().copy()
                elif child is not None:
                    direction = positions[child] - positions[source]
                    if direction.length > 1e-6:
                        swing = (desired @ Vector((0, 1, 0))).rotation_difference(direction)
                        desired = swing.to_matrix() @ desired
                parent = bone.parent
                kwargs = {'parent_matrix': pose_matrix(parent), 'parent_matrix_local': parent.bone.matrix_local} if parent else {}
                current = bone.bone.convert_local_to_pose(bone.matrix_basis, bone.bone.matrix_local, **kwargs)
                target = desired.to_4x4()
                target.translation = current.translation
                local = bone.bone.convert_local_to_pose(target, bone.bone.matrix_local, invert=True, **kwargs)
                rotation = local.to_quaternion().normalized()
                if name in grip:
                    rotation = grip[name].copy()
                if name in previous and rotation.dot(previous[name]) < 0:
                    rotation.negate()
                bone.rotation_quaternion = rotation
                previous[name] = rotation.copy()
                cache.pop(name, None)
                pose_matrix(bone)
                channels[name].append((1 + (row['time']-data['frames'][0]['time'])*data['fps'], tuple(rotation)))
            yield i+1, len(data['frames'])*passes

        # Smooth capture before IK; smoothing IK output would loosen contact.
        if smoothing > 0:
            for name, samples in channels.items():
                channels[name] = [(frame, tuple(Quaternion(q).slerp(
                    Quaternion(samples[max(0, i-1)][1]).slerp(Quaternion(samples[min(len(samples)-1, i+1)][1]), .5),
                    min(.8, smoothing)))) for i, (frame, q) in enumerate(samples)]
        if head_smoothing > 0:
            for name, samples in list(channels.items()):
                if names_j[name] in ('首', '頭'):
                    channels[name] = smooth_rotations(samples, data['fps'], head_smoothing)
        if body_base:
            for bone in body_bones(rig):
                if bone.name not in channels:
                    q = tuple(Matrix(body_base[bone.name]['basis']).to_quaternion())
                    channels[bone.name] = [(1+time*data['fps'], q) for time in
                        (row['time']-data['frames'][0]['time'] for row in data['frames'])]
        completed_passes = 1
        if torso_capture:
            from sam_ellen_torso import correct_torso
            capture_options = dict(torso_capture, screen_strength=0.) if coordinated else torso_capture
            for done, total in correct_torso(scratch, rig, channels, data, capture_options,
                                             torso_cameras, torso_base, torso_report):
                yield completed_passes*len(data['frames'])+done, total*passes
            completed_passes += 1
        if body_support:
            for done, total in correct_body(channels, rig, data, body_support):
                yield completed_passes*len(data['frames'])+done, total*passes
            completed_passes += 1
        if coordinated:
            from sam_ellen_frame_body import stabilize_capture
            stabilize_capture(channels,data,screen,rig)
        head_input = {name: list(samples) for name, samples in channels.items()}
        if head_pose:
            from sam_ellen_head_pose import correct_head_pose
            for done, total in correct_head_pose(scratch, rig, channels, data, head_pose, head_cameras, head_report):
                yield completed_passes*len(data['frames'])+done, total*passes
            completed_passes += 1
        gaze_input = {name: list(samples) for name, samples in channels.items()}
        if gaze:
            studio = bool(scene.get('CharacterStudio'))
            gaze = dict(gaze, center_motion=studio and gaze.get('head_tracking', False),
                        eye_limit_degrees=5 if studio else 20)
            for done, total in correct_gaze(scratch, rig, channels, data, gaze, cameras, eye_base=eye_base):
                yield completed_passes*len(data['frames'])+done, total*passes
            completed_passes += 1
        if coordinated and screen.get('calibrate_camera'):
            from sam_ellen_camera_fit import calibrated_camera, camera_source
            from sam_ellen_screen import camera_frames
            old_alignment_camera = screen.get('camera', '')
            calibrated, calibration_report = calibrated_camera(scene, target_rig, scratch, rig, channels, data, screen)
            from sam_ellen_camera_fit import bind_render_camera
            render_camera, render_camera_binding = bind_render_camera(scene, calibrated)
            if fixed:
                from sam_ellen_camera_fit import framing_driver_world
                # Calibrate the actual presentation view, including its local
                # adjustment. Never apply that extra distance twice.
                calibrated.matrix_world=framing_driver_world(scene,calibrated,render_camera,calibrated.matrix_world.copy())
                bpy.context.view_layer.update()
            if punch and punch.get('automatic'):
                calibrated.data.clip_start=min(calibrated.data.clip_start,.01)
            screen = dict(screen, camera=render_camera.name)
            torso_capture = dict(torso_capture, camera=render_camera.name)
            screen_cameras = camera_frames(scene, target_rig, render_camera, data, observation_space=True)
            calibration_report['render_camera'] = render_camera.name
            calibration_report['presentation_included_in_solve'] = True
            punch_target = bpy.data.objects.get(punch.get('camera', '')) if punch else None
            alignment_source = bpy.data.objects.get(old_alignment_camera) or original_scene_camera
            if punch and (not punch.get('camera') or punch_target == alignment_source
                          or punch_target and camera_source(punch_target) == camera_source(alignment_source)):
                punch = dict(punch, camera=render_camera.name)
                punch_cameras = screen_cameras
            if head_pose:
                head_pose = dict(head_pose, camera=render_camera.name)
                channels = {name: list(samples) for name, samples in head_input.items()}
                head_report = []
                for done, total in correct_head_pose(scratch, rig, channels, data, head_pose, screen_cameras, head_report):
                    yield completed_passes*len(data['frames']), len(data['frames'])*passes
                gaze_input = {name: list(samples) for name, samples in channels.items()}
            if gaze:
                gaze = dict(gaze, camera=render_camera.name)
                cameras = camera_samples(scene, target_rig, render_camera, data)
                channels = gaze_input
                for done, total in correct_gaze(scratch, rig, channels, data, gaze, cameras, eye_base=eye_base):
                    yield completed_passes*len(data['frames']), len(data['frames'])*passes
        coordinated_report = []
        framing_report = []
        if coordinated and fixed:
            from sam_ellen_fixed_capture import fit_subject,fixed_camera_check
            fixed_check=fixed_camera_check(scene,data)
            if max(fixed_check.values())>1e-6:
                raise ValueError('固定机位的最终摄像机仍存在动画或约束位移；请先取消视图摄像机的镜头动画')
            location_channels=fit_subject(scene,target_rig,scratch,rig,channels,data,screen,screen_cameras,framing_report)
            if punch and punch.get('camera')==screen.get('camera'):punch_cameras=screen_cameras
            completed_passes+=1
        elif coordinated:
            from sam_ellen_frame_body import fit_body
            tracking = bool(calibrated and not calibration_report.get('reused_for_append'))
            screen = dict(screen, framing_track=tracking)
            for done, total in fit_body(scratch, rig, channels, data, screen, screen_cameras, coordinated_report):
                yield completed_passes*len(data['frames'])+done*len(data['frames'])/total, len(data['frames'])*passes
            if tracking:
                from sam_ellen_framing import track_framing
                for done, total in track_framing(scene, target_rig, scratch, rig, channels, data,
                        screen, calibrated, screen_cameras, framing_report):
                    yield completed_passes*len(data['frames'])+done*len(data['frames'])/total, len(data['frames'])*passes
                # Re-read the actual hierarchy after keying its parent. The
                # committed camera and the solver must share every transform.
                screen_cameras = camera_frames(scene, target_rig, render_camera, data, observation_space=True)
                if punch and punch.get('camera') == render_camera.name:
                    punch_cameras = screen_cameras
            completed_passes += 1
        arm_reference={name:list(samples) for name,samples in channels.items()}
        close_range_report=[]
        data.pop('_sam_close_targets',None)
        if coordinated and screen:
            from sam_ellen_close_range import coordinate_close_hands
            close_exclusions={contact['side']} if contact else set()
            if punch and not punch.get('automatic'):close_exclusions.add(punch['side'])
            close_options=dict(screen,exclude_sides=sorted(close_exclusions))
            for done,total in coordinate_close_hands(scratch,rig,channels,data,close_options,screen_cameras,close_range_report):
                yield completed_passes*len(data['frames']),len(data['frames'])*passes
        if screen:
            from sam_ellen_screen import correct_screen
            for done, total in correct_screen(scratch, rig, channels, data, screen, screen_cameras, screen_report):
                yield completed_passes*len(data['frames'])+done*len(data['frames'])/total, len(data['frames'])*passes
            completed_passes += 1
        if contact:
            for done, total in correct_contacts(scratch, rig, channels, data, contact):
                yield completed_passes*len(data['frames'])+done, total*passes
            completed_passes += 1
        if punch:
            from sam_ellen_punch import correct_punch
            for done, total in correct_punch(scratch, rig, channels, data, punch, punch_cameras, punch_report, target_rig):
                yield completed_passes*len(data['frames'])+done, total*passes
            completed_passes += 1

        visibility_report={}
        projection_report=[]
        motion_report=[]
        if screen and screen.get('space')=='FRAME' and fixed:
            from sam_ellen_final_arms import finish_arms
            image_report=[]
            result=yield from finish_arms(scene,scratch,rig,target_rig,channels,data,screen,
                screen_cameras,contact,punch,motion_report)
            stable_sides,protected,allowed=result
        elif screen and screen.get('space')=='FRAME':
            from sam_ellen_arm_visibility import apply_visibility,protected_frames
            from sam_ellen_arm_policy import excluded,activation
            image_report=[]
            # Explicit manual pose overrides keep their authored intent. Image
            # refinement is for observed capture and automatic gesture repair.
            exclusions={contact['side']} if contact else set()
            if punch and not punch.get('automatic'):exclusions.add(punch['side'])
            image_options=dict(screen,exclude_sides=sorted(exclusions))
            final_sides=('L','R') if screen['side']=='BOTH' else (screen['side'],)
            protected={side:protected_frames(data,side,contact,punch) for side in final_sides}
            from sam_ellen_fist_final import refine_final_fist
            if punch and not excluded(image_options,punch['side']):
                for done,total in refine_final_fist(scratch,rig,target_rig,channels,data,punch,screen_cameras,image_report):
                    yield len(data['frames'])*passes,len(data['frames'])*passes
            # Verified video silhouettes are image evidence even when the
            # optional authored punch preset is disabled.
            from sam_punch_evidence import detect as detect_fists
            for side in final_sides:
                if excluded(image_options,side) or punch and punch['side']==side:continue
                segments,_=detect_fists(data['screen'],side)
                if not any(segment.get('silhouette') for segment in segments):continue
                automatic=dict(side=side,automatic=True,strength=screen.get('strength',1.),mirror=screen.get('mirror',False))
                for done,total in refine_final_fist(scratch,rig,target_rig,channels,data,automatic,screen_cameras,image_report):
                    yield len(data['frames'])*passes,len(data['frames'])*passes
            if screen.get('landmarks'):
                from sam_ellen_projection import close_projection
                # Fit against the final camera using the early arm reference.
                # The final coupled trajectory follows this image proposal.
                for done,total in close_projection(scratch,rig,target_rig,channels,data,image_options,screen_cameras,projection_report,protected,arm_reference):
                    yield len(data['frames'])*passes,len(data['frames'])*passes
            # Image searches must finish before temporal fitting; a subsequent
            # per-frame refit would reintroduce their discontinuous branches.
            from sam_ellen_motion_continuity import stabilize_motion
            for done,total in stabilize_motion(scratch,rig,channels,data,image_options,screen_cameras,motion_report,
                    repair_losses=True,reference_channels=arm_reference,protected=protected):
                yield len(data['frames'])*passes,len(data['frames'])*passes
            from sam_ellen_palm_orientation import align_palms
            for done,total in align_palms(scratch,rig,channels,data,image_options,screen_cameras,motion_report):
                yield len(data['frames'])*passes,len(data['frames'])*passes
            # Only this final state/hinge pass owns loss, entry and speed limits.
            # Only residual temporal fitting with the same length, wrist and
            # torso safeguards may follow; never run another image-space IK.
            image_weights=activation([r['time'] for r in data['frames']],image_options)
            allowed=image_weights>0
            for side in final_sides:
                if excluded(image_options,side):continue
                visibility_report[side]=[]
                apply_visibility(scratch,rig,channels,data,side,protected[side],visibility_report[side],
                    cameras=screen_cameras,allowed=allowed,mirror=image_options.get('mirror',False),image_weights=image_weights)
            from sam_ellen_arm_stability import stabilize_arms
            stable_sides = tuple(s for s in final_sides if not excluded(image_options, s))
            for side, done, total in stabilize_arms(scene, scratch, rig, channels, data,
                    protected=protected, allowed=allowed, sides=stable_sides, report=motion_report):
                yield len(data['frames'])*passes,len(data['frames'])*passes
            from sam_ellen_arm_trajectory import solve_trajectory
            for side,done,total in solve_trajectory(scratch,rig,channels,data,screen_cameras,
                    image_options,protected=protected,allowed=allowed,sides=stable_sides,report=motion_report):
                yield len(data['frames'])*passes,len(data['frames'])*passes
            from sam_ellen_arm_stability import stabilize_twist
            for side,done,total in stabilize_twist(scratch,rig,channels,data,
                    protected=protected,allowed=allowed,sides=stable_sides,report=motion_report):
                yield len(data['frames'])*passes,len(data['frames'])*passes
        else:
            image_report=[]

        if screen_report:
            from sam_ellen_screen_fit import measure_final
            for done, total in measure_final(scratch, rig, channels, data, screen, screen_cameras,
                                             screen_report, contact, punch):
                yield len(data['frames'])*passes, len(data['frames'])*passes

        action = bpy.data.actions.new('SAM Take ' + datetime.now().strftime('%Y%m%d_%H%M%S'))
        action.use_fake_user = True
        action['sam_close_range_report']=json.dumps(close_range_report)
        anim = target_rig.animation_data_create()
        if anim.action:
            anim.action.use_fake_user = True
        anim.action = action
        for track in anim.nla_tracks:
            track.mute = True
        last = channels[next(iter(channels))][-1][0]
        for bone in target_rig.pose.bones:
            bone.rotation_mode = 'QUATERNION'
            base = safe_pose_basis(bone, eye_base.get(bone.name, pose_base[bone.name])['basis'])
            samples = channels.get(bone.name)
            put_curve(action, target_rig, bone, 'rotation_quaternion', samples or [(1, tuple(base.to_quaternion())), (last, tuple(base.to_quaternion()))])
            put_curve(action, target_rig, bone, 'location', location_channels.get(bone.name) or [(1, tuple(base.translation)), (last, tuple(base.translation))])
            # Preserve existing scale as a constant, never animate it from SAM.
            put_curve(action, target_rig, bone, 'scale', [(1, tuple(base.to_scale())), (last, tuple(base.to_scale()))])
        if screen and screen.get('space')=='FRAME':
            from sam_ellen_arm_curve import smooth_keys
            from sam_ellen_arm_policy import PREFIXES
            for side in stable_sides:
                locked=protected[side]|~allowed
                for prefix in PREFIXES:smooth_keys(action,target_rig.pose.bones[prefix+side],locked)
        action['sam_video'] = data['video']
        action['sam_job_id'] = data['job_id']
        action['sam_source_start_seconds'] = data['frames'][0]['time']
        action['sam_sample_count'] = len(data['frames'])
        action['sam_retarget'] = 'fixed camera / grounded root translation / conservative pelvis / unified arms' if fixed else 'rotation-only / upper body / fixed roots and legs'
        action['sam_camera_mode']=mode
        action['sam_location_bones']=json.dumps(list(location_channels))
        if fixed:action['sam_fixed_camera_check']=json.dumps(fixed_check)
        action['sam_semantic_report']=json.dumps(data.get('_sam_semantic_report',{}))
        action['sam_capture_mode'] = capture_mode
        action['sam_inactive_pose'] = inactive_pose
        action['sam_mouth_contact'] = json.dumps(contact, ensure_ascii=False)
        action['sam_camera_gaze'] = json.dumps(gaze, ensure_ascii=False)
        action['sam_head_smoothing'] = head_smoothing
        action['sam_head_pose'] = json.dumps(head_pose, ensure_ascii=False)
        action['sam_head_pose_report'] = json.dumps(head_report)
        action['sam_body_support'] = json.dumps(body_support, ensure_ascii=False)
        action['sam_torso_capture'] = json.dumps(torso_capture, ensure_ascii=False)
        if torso_capture or screen and screen.get('landmarks') or stable_pose:
            action['sam_camera_match_base'] = json.dumps(pose_base)
        action['sam_torso_report'] = json.dumps(torso_report)
        if torso_base:
            action['sam_torso_base'] = json.dumps(torso_base)
        if body_base:
            action['sam_body_base'] = json.dumps(body_base, ensure_ascii=False)
        action['sam_screen_alignment'] = json.dumps(screen, ensure_ascii=False)
        action['sam_screen_report'] = json.dumps(screen_report)
        action['sam_image_refinement'] = json.dumps(image_report)
        action['sam_arm_visibility_report']=json.dumps(visibility_report)
        action['sam_arm_entry']=json.dumps({'version':2,'reports':data.get('_sam_entry_report',[]),
            'direction_repairs':data.get('_sam_direction_repairs',[])})
        action['sam_body_clearance']=json.dumps(data.get('_sam_body_clearance',[]))
        action['sam_elbow_evidence']=json.dumps(data.get('_sam_elbow_evidence',[]))
        action['sam_projection_report']=json.dumps(projection_report)
        action['sam_motion_continuity']=json.dumps({'version':2,'reports':motion_report})
        action['sam_arm_policy']=json.dumps({'version':2,'reference':'before first hand image fit',
            'final_owner':'coupled image / whole-chain clearance / continuous trajectory',
            'frames':data.get('_sam_arm_policy',[])})
        if projection_report:action['sam_projection_closure']=json.dumps({'version':2,'camera':screen['camera']})
        action['sam_coordinated_body_report'] = json.dumps(coordinated_report)
        action['sam_framing_report'] = json.dumps(framing_report)
        action['sam_camera_calibration'] = json.dumps(calibration_report)
        action['sam_punch'] = json.dumps(punch, ensure_ascii=False)
        action['sam_punch_report'] = json.dumps(punch_report)
        action['sam_punch_limited_samples'] = sum(row.get('limited', False) for row in punch_report)
        if eye_base:
            action['sam_gaze_eye_base'] = json.dumps(eye_base, ensure_ascii=False)
        action['sam_mapping'] = json.dumps(mapping, ensure_ascii=False)
        action['sam_wilor'] = json.dumps(data.get('wilor_refinement'), ensure_ascii=False)
        action['sam_finger_reliability'] = json.dumps(data.get('finger_reliability'), ensure_ascii=False)
        action['sam_gesture_assistance'] = json.dumps(data.get('gesture_assistance'), ensure_ascii=False)
        scene.render.fps = round(data['fps'])
        scene.render.fps_base = round(data['fps']) / data['fps']
        scene.frame_start, scene.frame_end = 1, int(math.ceil(last-1e-7))
        if wiggle is not None:
            scene.wiggle_enable = False
        scene.frame_set(1)
        if fixed and ground_reference:
            from sam_ellen_fixed_capture import ground_deformed_feet
            action['sam_ground_reference']=json.dumps(ground_reference)
            action['sam_ground_report']=json.dumps(ground_deformed_feet(scene,target_rig,data,ground_reference))
        if screen and screen.get('landmarks') and screen.get('space') == 'FRAME' and not data.get('_sam_timeline'):
            from sam_ellen_validation import validate_final, summary, require_safe_arms
            final_report=[]
            for done,total in validate_final(scene,target_rig,data,screen,final_report):
                yield len(data['frames'])*passes,len(data['frames'])*passes
            if fixed:
                from sam_ellen_arm_curve import linearize_unsafe_edges
                interpolation_repairs=linearize_unsafe_edges(action,target_rig,final_report)
                if interpolation_repairs:
                    action['sam_interpolation_repairs']=json.dumps(interpolation_repairs)
                    final_report=[]
                    for done,total in validate_final(scene,target_rig,data,screen,final_report):
                        yield len(data['frames'])*passes,len(data['frames'])*passes
            action['sam_final_validation']=json.dumps(summary(final_report))
            action['sam_final_scene_report']=json.dumps(final_report)
            mouth_failures=[(row.get('frame'),side) for row in final_report
                for side,hand in row.get('hands',{}).items()
                if hand.get('semantic_intent')=='mouth-cover'
                and hand.get('semantic_phase','hold')=='hold'
                and hand.get('semantic_weight',0.)>=.45
                and not hand.get('mouth_cover_pass',False)]
            if mouth_failures:
                sample='、'.join('%g帧%s'%(frame,side) for frame,side in mouth_failures[:5])
                raise ValueError('捂嘴动作未在最终摄像机中形成遮挡与前后关系（%s），已保留原动作；请检查角色可达性和镜头对齐'%sample)
            require_safe_arms(final_report,sides=stable_sides,
                active_times=([r['time'] for r in data['frames']],allowed))
            for body_row, final_row in zip(coordinated_report, (r for r in final_report if not r.get('intermediate'))):
                if 'face_center_px' in final_row and 'after_center_px' in body_row:
                    body_row['pose_after_center_px']=body_row['after_center_px']
                    body_row['after_center_px']=final_row['face_center_px']
                    body_row['measurement_stage']='committed-evaluated-scene'
            action['sam_coordinated_body_report']=json.dumps(coordinated_report)
        # Commit settings only after safety succeeds. A rejected first bake
        # removes its new presentation camera and must retain the old pointers.
        if calibrated and hasattr(scene, 'sam_ellen'):
            scene.sam_ellen.screen_camera = render_camera
            if gaze:
                scene.sam_ellen.gaze_camera = render_camera
        committed = True
        return action
    finally:
        bpy.data.objects.remove(rig, do_unlink=True)
        bpy.data.scenes.remove(scratch)
        if wiggle is not None:
            scene.wiggle_enable = wiggle
        if not committed:
            if render_camera_binding:
                from sam_ellen_camera_fit import restore_render_camera
                restore_render_camera(scene, render_camera_binding)
            scene.camera = original_scene_camera
            if calibrated and not calibration_report.get('reused_for_append'):
                camera_data = calibrated.data
                bpy.data.objects.remove(calibrated, do_unlink=True)
                if camera_data.users == 0:
                    bpy.data.cameras.remove(camera_data)
            restore(target_rig, scene, state)
            if action:
                bpy.data.actions.remove(action)

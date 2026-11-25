"""Final temporal fitting of complete arm hinges, after image and safety passes."""
import math

import numpy as np
from mathutils import Quaternion, Vector


def bounded_roll_curve(values, steps, locked):
    """Fit a bounded physical driver, retaining exact authored intervals.

    Quaternion equivalence modulo 2*pi does not apply to MMD fractional twist
    helpers. Keep the signed scalar and project both directions in time, so a
    dropped hand cannot introduce an artificial forearm spin.
    """
    values = np.asarray(values, float)
    steps = np.asarray(steps, float)
    locked = np.asarray(locked, bool)
    result = values.copy()
    editable = np.flatnonzero(~locked)
    runs = np.split(editable, np.flatnonzero(np.diff(editable) > 1)+1)
    limit = math.radians(170.)
    for run in runs:
        if not len(run):
            continue
        first, last = int(run[0]), int(run[-1])
        left, right = max(0, first-1), min(len(values)-1, last+1)
        target = values[left:right+1]
        fixed = locked[left:right+1]
        bounds = steps[left:right]
        lo = np.full(len(target), -limit)
        hi = np.full(len(target), limit)
        lo[fixed] = hi[fixed] = target[fixed]
        for i in range(1, len(target)):
            lo[i] = max(lo[i], lo[i-1]-bounds[i-1])
            hi[i] = min(hi[i], hi[i-1]+bounds[i-1])
        for i in range(len(target)-2, -1, -1):
            lo[i] = max(lo[i], lo[i+1]-bounds[i])
            hi[i] = min(hi[i], hi[i+1]+bounds[i])
        if np.any(lo > hi+1e-7):
            raise ValueError('前臂扭转无法平滑连接受保护动作，已保留原动作')
        forward = np.clip(target, lo, hi)
        backward = forward.copy()
        for i in range(1, len(target)):
            forward[i] = np.clip(forward[i], max(lo[i], forward[i-1]-bounds[i-1]),
                                 min(hi[i], forward[i-1]+bounds[i-1]))
        for i in range(len(target)-2, -1, -1):
            backward[i] = np.clip(backward[i], max(lo[i], backward[i+1]-bounds[i]),
                                  min(hi[i], backward[i+1]+bounds[i]))
        fitted = (forward+backward)*.5
        result[first:last+1] = fitted[first-left:last-left+1]
    return result


def stabilize_twist(scratch, rig, channels, data, protected=None, allowed=None,
                    sides=('L', 'R'), report=None):
    """Reconcile twist AFTER the last hinge solve, retaining palms and fingers.

    Earlier visibility limits can be invalidated by later hinge/roll refits.
    Reserve a rotation budget for the final inherited elbow motion as well as
    the controller itself. The evaluated-scene safety guard still runs after
    commit; this is a pose correction, not a relaxed validation threshold.
    """
    from sam_ellen_retarget import set_global_rotation
    times = np.asarray([r['time'] for r in data['frames']], float)
    count = len(times)
    if count < 2:
        return
    allowed = np.ones(count, bool) if allowed is None else np.asarray(allowed, bool)
    original = {name: list(rows) for name, rows in channels.items()}
    view = scratch.view_layers[0]

    def load(i):
        for name, rows in original.items():
            rig.pose.bones[name].rotation_quaternion = rows[i][1]
        view.update()

    for side in sides:
        control_name, wrist_name = 'HandTwist_'+side, 'Wrist_'+side
        if control_name not in channels or wrist_name not in channels:
            continue
        controller, wrist = rig.pose.bones[control_name], rig.pose.bones[wrist_name]
        locked = (np.asarray(protected.get(side, np.zeros(count, bool)), bool)
                  if protected else np.zeros(count, bool)) | ~allowed
        rolls, bases, palms = [], [], []
        for i in range(count):
            load(i)
            local = controller.rotation_quaternion.normalized()
            if local.w < 0:
                local.negate()
            roll = local.to_swing_twist('Y')[1]
            rolls.append(roll)
            bases.append(controller.matrix.to_quaternion().normalized() @ local.inverted())
            palms.append(wrist.matrix.to_quaternion().copy())
            if i % 80 == 0:
                yield side, i, count*2
        inherited = np.array([2*math.acos(min(1., abs(a.dot(b))))
                              for a, b in zip(bases[:-1], bases[1:])])
        # Triangle inequality bounds the complete physical rotation, including
        # the MMD helper bones inheriting fractions of this local driver.
        steps = np.minimum(np.radians(600)*np.diff(times),
                           np.maximum(0., np.radians(1100)*np.diff(times)-inherited))
        fitted = bounded_roll_curve(rolls, steps, locked)
        changes = []
        for i in np.flatnonzero(abs(fitted-np.asarray(rolls)) > 1e-7):
            load(int(i))
            controller.rotation_quaternion = Quaternion((0, 1, 0), float(fitted[i]))
            view.update()
            set_global_rotation(wrist, palms[i])
            view.update()
            for name in (control_name, wrist_name):
                channels[name][i] = channels[name][i][0], tuple(rig.pose.bones[name].rotation_quaternion)
            changes.append(int(i)+1)
            if i % 80 == 0:
                yield side, count+int(i), count*2
        if report is not None:
            report.append({'model': 'final-physical-twist-continuity-v1', 'side': side,
                           'changed_frames': changes, 'preserved': 'palm pose / arm endpoints / fingers / protected keys'})


def phone_weights(scene, data, side):
    from sam_ellen_phone_ranges import ranges
    from sam_ellen_sequence import timeline_frame
    settings = getattr(scene, 'sam_ellen', None)
    times = np.asarray([r['time'] for r in data['frames']], float)
    weights = np.zeros(len(times))
    if not settings or not settings.phone_enabled or settings.phone_side != side:
        return weights
    for first, last in ranges(settings):
        frames = np.asarray([timeline_frame(data, t) for t in times])
        # Ramp the extra damping inside the range, never change its visibility.
        distance = np.minimum(frames-first, last-frames)/data['fps']
        w = np.clip(distance/.18, 0., 1.)
        weights = np.maximum(weights, w*w*(3-2*w))
    return weights


def stabilize_arms(scene, scratch, rig, channels, data, protected=None,
                   allowed=None, sides=('L', 'R'), report=None):
    """Filter torso-relative hinge, flexion and palm together, preserving lengths.

    A hand's classification score is not localization confidence. Partial crop
    support gets less weight; genuine low-frequency motion gets more. Explicit
    contact/fist intervals and their boundary poses remain exact anchors.
    """
    from sam_ellen_arm_visibility import screen_hand, hand_landmark_mask
    from sam_ellen_arm_chain import frame_rotation, anatomical_rotations, bounded_wrist_rotation
    from sam_ellen_body_clearance import torso_geometry, torso_envelope, arm_clearance
    from sam_ellen_arm_policy import wrist_limit, PREFIXES
    from sam_ellen_retarget import set_global_rotation
    from sam_ellen_temporal import quaternions, trajectory

    times = np.asarray([r['time'] for r in data['frames']], float)
    count = len(times)
    if count < 3:
        return
    allowed = np.ones(count, bool) if allowed is None else np.asarray(allowed, bool)
    geometry = torso_geometry(rig)
    original = {name: list(rows) for name, rows in channels.items()}
    view = scratch.view_layers[0]
    chest = rig.pose.bones.get('UpperBody2') or rig.pose.bones.get('UpperBody1')
    if chest is None:
        return

    def load(i):
        for name, rows in original.items():
            rig.pose.bones[name].rotation_quaternion = rows[i][1]
        view.update()

    for side in sides:
        if not all(p+side in channels for p in PREFIXES):
            continue
        a, e, w = [rig.pose.bones[p+'_'+side] for p in ('Arm', 'Elbow', 'Wrist')]
        controller = rig.pose.bones['HandTwist_'+side]
        anchors = (np.asarray(protected.get(side, np.zeros(count, bool)), bool)
                   if protected else np.zeros(count, bool)) | ~allowed
        amount = np.ones(count)
        if anchors.any():
            distance = np.min(abs(times[:, None]-times[anchors][None, :]), axis=1)
            amount = np.clip(distance/.20, 0., 1.)
            amount = amount*amount*(3-2*amount)
        frames, flexes, palms, rolls, positions = [], [], [], [], []
        support = np.zeros(count)
        for i in range(count):
            load(i)
            body = chest.matrix.to_quaternion() @ chest.bone.matrix_local.to_quaternion().inverted()
            u, v = e.head-a.head, w.head-e.head
            frames.append(body.inverted() @ frame_rotation(u, u.cross(v)))
            flexes.append(u.angle(v))
            palms.append(body.inverted() @ w.matrix.to_quaternion())
            q = controller.rotation_quaternion.copy()
            if q.w < 0:
                q.negate()
            rolls.append(q.to_swing_twist('Y')[1])
            positions.append(body.inverted() @ (w.head-a.head))
            hand = screen_hand(data.get('screen', {}), i, side)
            mask = hand_landmark_mask(hand)
            support[i] = float(np.mean(mask)) if hand else 0.
            if i % 80 == 0:
                yield side, i, count*2
        coarse = trajectory(times, np.asarray(positions), .14)
        speed = np.linalg.norm(np.gradient(coarse, times, axis=0), axis=1)
        # Fit the complete motion first: frame-wise IK corrections must not be
        # allowed to masquerade as fast intentional movement in this weighting.
        weights = .65 + 1.35*support + 6*np.clip(speed/.65, 0., 1.)**2
        grip = phone_weights(scene, data, side)
        filtered_frames = quaternions(times, frames, .065, weights)
        filtered_flexes = trajectory(times, flexes, .065, weights)
        filtered_palms = quaternions(times, palms, .055, weights)
        # This is a bounded MMD controller, not a freely spinning orientation.
        # Its fractional helper bones make roll+2*pi a different skin pose.
        # Unwrapping then clamping would break protected interval boundaries.
        filtered_rolls = trajectory(times, rolls, .075, weights)
        if grip.any():
            phone_frames = quaternions(times, frames, .095, weights)
            phone_flexes = trajectory(times, flexes, .095, weights)
            phone_palms = quaternions(times, palms, .085, weights)
            phone_rolls = trajectory(times, rolls, .10, weights)
            for i, gain in enumerate(grip):
                filtered_frames[i] = filtered_frames[i].slerp(phone_frames[i], float(gain))
                filtered_flexes[i] = (1-gain)*filtered_flexes[i] + gain*phone_flexes[i]
                filtered_palms[i] = filtered_palms[i].slerp(phone_palms[i], float(gain))
                filtered_rolls[i] = (1-gain)*filtered_rolls[i] + gain*phone_rolls[i]
        hinge = {}
        changes = []
        for i in range(count):
            if amount[i] <= 0:
                hinge = {}
                continue
            load(i)
            body = chest.matrix.to_quaternion() @ chest.bone.matrix_local.to_quaternion().inverted()
            frame = body @ frames[i].slerp(filtered_frames[i], float(amount[i]))
            flex = float((1-amount[i])*flexes[i] + amount[i]*filtered_flexes[i])
            # Preserve the original endpoint-safe elbow range; avoid a straight
            # singularity even if a filtered flexion overshoots at a boundary.
            flex = float(np.clip(flex, math.radians(15), math.radians(145)))
            hand = body @ palms[i].slerp(filtered_palms[i], float(amount[i]))
            aa = a.head.copy()
            upper, lower = (e.head-aa).length, (w.head-e.head).length
            old_wrist = w.head.copy()
            u = frame @ Vector((0, 1, 0))
            v = Quaternion(frame @ Vector((1, 0, 0)), flex) @ u
            ee, ww = aa + u*upper, aa + u*upper + v*lower
            envelope = torso_envelope(rig, geometry)
            margin = arm_clearance(aa, ee, ww, envelope)
            # This is a smooth proposal. The coupled trajectory owner below
            # handles clearance together with image and transition constraints;
            # projecting individual frames here can undo the temporal fit.
            elbow = math.degrees((aa-ee).angle(ww-ee))
            hand = bounded_wrist_rotation(hand, ww-ee, wrist_limit(elbow))
            anatomical_rotations(a, e, w, aa, ee, ww, hand, set_global_rotation, view, hinge)
            palm = w.matrix.to_quaternion().copy()
            roll = (1-amount[i])*rolls[i] + amount[i]*filtered_rolls[i]
            controller.rotation_quaternion = Quaternion((0, 1, 0),
                float(np.clip(roll, -math.radians(170), math.radians(170))))
            view.update()
            set_global_rotation(w, palm)
            view.update()
            for prefix in PREFIXES:
                name = prefix+side
                channels[name][i] = channels[name][i][0], tuple(rig.pose.bones[name].rotation_quaternion)
            changes.append({'frame': i+1, 'wrist_shift_m': (w.head-old_wrist).length,
                            'clearance_m': margin, 'phone_weight': float(grip[i])})
            if i % 80 == 0:
                yield side, count+i, count*2
        if report is not None:
            report.append({'model': 'final-hinge-residual-stability-v1', 'side': side,
                'frames': count, 'protected_frames': (np.flatnonzero(anchors)+1).tolist(),
                'changes': changes, 'time_filter': 'centered / torso-relative / crop and motion weighted'})

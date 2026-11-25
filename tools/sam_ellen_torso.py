"""Captured torso leaning with feet fixed and bounded face-trajectory correction."""
import math

import numpy as np
from mathutils import Matrix, Quaternion, Vector


def correct_torso(scratch, rig, channels, data, options, cameras, base, report):
    from sam_ellen_retarget import body_bones, smooth_rotations, set_global_rotation
    from sam_ellen_screen import project, unproject
    bones = body_bones(rig)
    originals = {b.name: smooth_rotations(channels[b.name], data['fps'], options['smoothing']) for b in bones}
    neutral = {b.name: Quaternion(originals[b.name][0][1]) for b in bones}
    poses = {b.name: Matrix(base[b.name]['basis']).to_quaternion() for b in bones}
    faces = [r.get('face') for r in data.get('screen', {}).get('frames', [])]
    valid = [f for f in faces[:max(1, round(data.get('sample_fps', data['fps'])*.5))] if f] if faces else []
    if not valid:
        valid = [f for f in faces if f][:5]
    reference = (np.median([f['center'] for f in valid], axis=0),
                 float(np.median([f.get('scale', f['width']) for f in valid]))) if valid else None
    max_angle = math.radians(options['max_angle'])
    previous_delta = np.zeros(3)
    previous_target, last = None, None
    anchor, anchor_width, anchor_depth = None, None, None
    for i, row in enumerate(data['frames']):
        for name, values in channels.items():
            rig.pose.bones[name].rotation_quaternion = values[i][1]
        # Remove the take's initial torso bias; keep the user's within-take movement.
        for bone in bones:
            delta = neutral[bone.name].rotation_difference(Quaternion(originals[bone.name][i][1]))
            if delta.w < 0:
                delta.negate()
            delta = Quaternion().slerp(delta, options['strength'])
            axis, angle = delta.to_axis_angle()
            if angle > max_angle/len(bones):
                delta = Quaternion(axis, max_angle/len(bones))
            bone.rotation_quaternion = poses[bone.name]@delta
        scratch.view_layers[0].update()
        eyes = [rig.pose.bones['Eye_'+s] for s in ('L', 'R')]
        center = lambda: (eyes[0].head+eyes[1].head)*.5
        before = center().copy()
        camera = cameras[i]
        if anchor is None:
            anchor = project(before, camera)
            anchor_width = (project(eyes[0].head, camera)-project(eyes[1].head, camera)).length
            anchor_depth = -(camera[0]@before).z
        face = faces[i] if i < len(faces) else None
        active = bool(face and reference and options['screen_strength'] > 0)
        limited, error_before, error_after = False, 0., 0.
        if active:
            rel = (np.array(face['center'])-reference[0])/max(reference[1], .005)
            if options.get('mirror'):
                rel[0] *= -1
            xy = anchor+Vector(rel)*anchor_width
            scale = face.get('scale', face['width'])/max(reference[1], .005)
            scale = max(.8, min(1.25, scale))
            depth = anchor_depth/scale if camera[3] == 'PERSP' else anchor_depth
            desired = unproject(xy, depth, camera)
            if previous_target is not None:
                alpha = 1-math.exp(-(row['time']-last)/max(.01, options['smoothing']))
                desired = previous_target.lerp(desired, alpha)
            previous_target, last = desired.copy(), row['time']
            desired = before.lerp(desired, options['screen_strength'])
            initial = {b.name: b.rotation_quaternion.copy() for b in bones}
            axes = camera[0].to_quaternion().inverted().to_matrix()
            pivots = [np.array(b.head) for b in bones]
            global_rotations = {b.name: b.matrix.to_quaternion() for b in bones}

            def evaluate(value):
                # Float64 forward kinematics avoids noisy float32 scene derivatives.
                vector = np.asarray(axes)@value/len(bones)
                angle = float(np.linalg.norm(vector))
                if angle > 1e-10:
                    x, y, z = vector/angle
                    k = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
                    rotation = np.eye(3)+math.sin(angle)*k+(1-math.cos(angle))*(k@k)
                else:
                    rotation = np.eye(3)
                point = np.array(before)
                for pivot in reversed(pivots):
                    point = rotation@(point-pivot)+pivot
                return point-np.array(desired)

            value = previous_delta.copy()
            error_before = (before-desired).length
            residual = evaluate(value)
            for _ in range(8):
                jac = []
                for j in range(3):
                    trial = value.copy()
                    trial[j] += .0001
                    jac.append((evaluate(trial)-residual)/.0001)
                j = np.stack(jac, axis=1)
                # Prefer the capture; the screen constraint only provides a bounded correction.
                reg = .002
                step = np.linalg.solve(j.T@j+np.eye(3)*reg, -j.T@residual-reg*(value-previous_delta*.6))
                step *= min(1., .08/max(1e-8, np.linalg.norm(step)))
                value += step
                norm = np.linalg.norm(value)
                limited |= norm > max_angle
                value *= min(1., max_angle/max(1e-8, norm))
                residual = evaluate(value)
            if np.linalg.norm(residual) > error_before:
                value = np.zeros(3)
                residual = evaluate(value)
            angle = float(np.linalg.norm(value))
            step_rotation = Quaternion(axes@Vector(value/angle), angle/len(bones)) if angle > 1e-9 else Quaternion()
            cumulative = Quaternion()
            for bone in bones:
                cumulative = step_rotation@cumulative
                set_global_rotation(bone, cumulative@global_rotations[bone.name])
                scratch.view_layers[0].update()
            if (center()-desired).length > error_before:
                for bone in bones:
                    bone.rotation_quaternion = initial[bone.name]
                scratch.view_layers[0].update()
                value = np.zeros(3)
            previous_delta = value.copy()
            error_after = (center()-desired).length
        else:
            previous_delta *= 0
        frame = channels[bones[0].name][i][0]
        for bone in bones:
            q = bone.rotation_quaternion.copy()
            if i and q.dot(Quaternion(channels[bone.name][i-1][1])) < 0:
                q.negate()
            channels[bone.name][i] = frame, tuple(q)
        report.append({'time': row['time'], 'active': active, 'limited': bool(limited),
                       'before_error_m': error_before, 'after_error_m': error_after,
                       'head_position': list(center())})
        yield i+1, len(data['frames'])

"""Short camera-punch repair using captured fist shape and non-stretch arm IK."""
import bisect
import math

from mathutils import Quaternion, Vector


def validate_punch(data, options, mode):
    side = options.get('side')
    if side not in ('L', 'R') or mode == 'SELFIE_LEFT' and side != 'L' or mode == 'SELFIE_RIGHT' and side != 'R':
        raise ValueError('出拳手与自拍预设不一致')
    if options.get('automatic'):
        from sam_punch_evidence import detect
        screen = data.get('screen', {})
        if (screen.get('video') != data['video'] or len(screen.get('frames', [])) != len(data['frames'])
                or any(abs(a['time']-b['time']) > .0001 for a,b in zip(screen.get('frames', []),data['frames']))):
            raise ValueError('自动出拳需要与视频时间轴一致的画面关键点，请重新识别')
        if not 0 <= options.get('strength', -1) <= 1:
            raise ValueError('Invalid punch strength')
        options['segments'], _ = detect(screen, side)
        return
    keys = ('start', 'peak', 'release', 'end', 'reference', 'distance', 'x', 'y', 'strength')
    if not all(isinstance(options.get(k), (int, float)) and math.isfinite(options[k]) for k in keys):
        raise ValueError('Invalid punch settings')
    duration = data['frames'][-1]['time']-data['frames'][0]['time']
    if not 0 <= options['start'] < options['peak'] <= options['release'] < options['end'] <= duration:
        raise ValueError('请按开始 < 击中 <= 停留结束 < 收回结束设置，且不要超过视频长度')
    if not 0 <= options['reference'] < options['peak'] or options['reference'] > duration:
        raise ValueError('握拳参考时间应位于击中之前')
    if not .05 <= options['distance'] <= 2 or not 0 <= options['strength'] <= 1:
        raise ValueError('Invalid punch distance or strength')
    if not 0 <= options['x'] <= 1 or not 0 <= options['y'] <= 1:
        raise ValueError('Invalid punch screen target')


def ease(t):
    t = max(0., min(1., t))
    return t*t*(3-2*t)


def punch_weight(time, options):
    start, peak, release, end = [options[k] for k in ('start', 'peak', 'release', 'end')]
    if time <= start or time >= end:
        return 0.
    if time < peak:
        value = ease((time-start)/(peak-start))
    elif time <= release:
        value = 1.
    else:
        value = 1-ease((time-release)/(end-release))
    return value*options['strength']


def sample_at(samples, time):
    if len(samples) == 1:
        return Quaternion(samples[0][1])
    frames = [v[0] for v in samples]
    i = max(0, min(len(samples)-2, bisect.bisect_right(frames, time)-1))
    a, b = samples[i], samples[i+1]
    t = max(0., min(1., (time-a[0])/max(1e-8, b[0]-a[0])))
    return Quaternion(a[1]).slerp(Quaternion(b[1]), t)


def ray_reach(xy, depth, offset, shoulder, reach, camera):
    """Keep the fist on its image ray when a depth exceeds the real arm reach."""
    from sam_ellen_screen import unproject
    origin = unproject(xy, 0., camera)-offset
    ray = unproject(xy, 1., camera)-offset-origin
    delta = origin-shoulder
    a, b, c = ray.dot(ray), 2*ray.dot(delta), delta.dot(delta)-reach*reach
    disc = b*b-4*a*c
    if disc >= 0:
        lo, hi = (-b-math.sqrt(disc))/(2*a), (-b+math.sqrt(disc))/(2*a)
        lo = max(.08, lo)
        if hi >= lo:
            actual = max(lo, min(hi, depth))
            return origin+ray*actual, abs(actual-depth) > 1e-5, True
    # No depth can satisfy this ray. Report the remaining screen error explicitly.
    closest = origin+ray*max(.08, -ray.dot(delta)/a)
    direction = closest-shoulder
    return shoulder+direction.normalized()*reach, True, False


def image_goal(points, target, camera):
    """Fit knuckles rather than confusing the wrist with the visible fist center."""
    import numpy as np
    from sam_punch_evidence import ANCHORS
    from sam_ellen_screen import project, unproject
    from sam_ellen_screen_fit import fit_hand
    anchors = list(ANCHORS)
    desired = np.asarray(target, dtype=float)
    center = Vector(desired[anchors].mean(0))
    radius = float(np.sqrt(np.mean(np.sum((desired[anchors]-np.array(center))**2, axis=1))))
    wrist = points[0]
    offset = sum((points[j]-wrist for j in anchors), Vector())/len(anchors)
    near = max(.08, max((camera[0].to_3x3()@(p-wrist-offset)).z for p in points)+.04)
    lo, hi = near, max(10., -(camera[0]@wrist).z*4)
    translated = points
    for _ in range(28):
        depth = (lo+hi)*.5
        goal = unproject(center, depth, camera)-offset
        # Centering a projected 3-D cluster needs a small perspective correction.
        for _ in range(3):
            translated = [p-wrist+goal for p in points]
            projected_center = sum((project(translated[j], camera) for j in anchors), Vector((0., 0.)))/len(anchors)
            goal += unproject(center, depth, camera)-unproject(projected_center, depth, camera)
        image = np.array([project(p-wrist+goal, camera) for p in points])
        actual_radius = float(np.sqrt(np.mean(np.sum((image[anchors]-image[anchors].mean(0))**2, axis=1))))
        if actual_radius > radius:
            lo = depth
        else:
            hi = depth
    translated = [p-wrist+goal for p in points]
    mask = np.zeros(21, dtype=bool); mask[anchors] = True
    fitted, rotation, _ = fit_hand(translated, desired, mask, camera, 100., match_image=True)
    # fit_hand rotates about the translated wrist, so its result is already the
    # complete wrist position, not a displacement to add to the captured wrist.
    return fitted, rotation


def correct_punch(scratch, rig, channels, data, options, cameras, report, source_rig=None):
    if options.get('automatic') and data.get('screen', {}).get('punch_tracking', {}).get('mask_version') == 2:
        from sam_ellen_punch_mesh import correct_video_punch
        yield from correct_video_punch(scratch, rig, source_rig, channels, data, options, cameras, report)
        return
    import numpy as np
    from sam_ellen_retarget import bone_map, correct_contacts, set_global_rotation
    from sam_ellen_screen import project
    from sam_ellen_screen_fit import hand_points
    from sam_punch_evidence import ANCHORS, detect
    from sam_ellen_sequence import timeline_frame
    side = options['side']; automatic = options.get('automatic', False)
    view = scratch.view_layers[0]
    originals = {name: list(values) for name, values in channels.items()}
    times = [r['time']-data['frames'][0]['time'] for r in data['frames']]
    segments, observations = (detect(data['screen'], side) if automatic else ([options], [None]*len(times)))
    options['segments'] = segments
    if automatic and any(c[3] != 'PERSP' for c in cameras):
        raise ValueError('自动出拳大小拟合需要透视摄像机，请先切换摄像机类型')
    wrist, arm, elbow = [rig.pose.bones[n+'_'+side] for n in ('Wrist', 'Arm', 'Elbow')]
    reach = ((arm.bone.head_local-elbow.bone.head_local).length
             +(elbow.bone.head_local-wrist.bone.head_local).length)*.985
    offset = 36 if side == 'L' else 0
    fingers = {name for name, (index, _) in bone_map(rig).items()
               if any(first+offset <= index < first+offset+3 for first in (61, 56, 52, 48, 44))}
    goals, metadata = [], []
    for i, time in enumerate(times):
        segment = next((s for s in segments if s['start'] < time < s['end']), None)
        observation = observations[i]
        faded = 1.
        if automatic and segment and observation is None:
            prior = next((j for j in range(i-1, -1, -1) if observations[j] is not None), None)
            if prior is not None and times[i]-times[prior] < .12:
                observation = dict(observations[prior], source='short-fade',
                                   seed_time=data['frames'][prior]['time'])
                faded = 1-ease((times[i]-times[prior])/.12)
        weight = (min(ease((time-segment['start'])/.10), ease((segment['end']-time)/.12))*options['strength']
                  if automatic and segment else punch_weight(time, options) if not automatic else 0.)
        weight *= faded
        row = {'time': time, 'active': False, 'model': 'image-fist-v2' if automatic else 'manual-fist-v2'}
        if weight <= 0 or automatic and observation is None:
            row['reason'] = 'no-image-evidence' if segment else 'outside-detected-event'
            metadata.append(row); goals.append(None); continue
        for name, samples in channels.items():
            rig.pose.bones[name].rotation_quaternion = samples[i][1]
        view.update()
        before = wrist.head.copy(); rotation = wrist.matrix.to_quaternion().normalized()
        # Preserve real elbow/shoulder motion. Only bridge unreliable finger
        # articulation, using the tracked image patch's own reliable seed.
        reference = observation.get('seed_time', data['frames'][0]['time']+segment['reference'])-data['frames'][0]['time'] if automatic else segment['reference']
        hold = not automatic or observation['source'] != 'landmarks'
        if hold:
            frame = timeline_frame(data, data['frames'][0]['time']+reference)
            for name in fingers:
                f, q = channels[name][i]
                seed = sample_at(originals[name], frame)
                channels[name][i] = f, tuple(Quaternion(q).slerp(seed, weight))
                rig.pose.bones[name].rotation_quaternion = channels[name][i][1]
            # Recover wrist orientation at the same seed without changing the arm.
            for name, samples in originals.items():
                rig.pose.bones[name].rotation_quaternion = sample_at(samples, frame)
            view.update(); seed_rotation = wrist.matrix.to_quaternion().normalized()
            for name, samples in channels.items():
                rig.pose.bones[name].rotation_quaternion = samples[i][1]
            view.update(); set_global_rotation(wrist, seed_rotation); view.update()
        points = hand_points(rig, side); camera = cameras[i]
        if automatic:
            target = np.array(observation['points'], dtype=float)
            target[:, 0] *= camera[2]/data['screen']['aspect']
            if options.get('mirror'):
                target[:, 0] = camera[2]-target[:, 0]
            requested, delta = image_goal(points, target, camera)
            aimed = delta@wrist.matrix.to_quaternion().normalized()
            fist_offset = sum((delta@(points[j]-points[0]) for j in ANCHORS), Vector())/len(ANCHORS)
            target_xy = Vector(target[list(ANCHORS)].mean(0))
            depth = -(camera[0]@(requested+fist_offset)).z
        else:
            aimed = wrist.matrix.to_quaternion().normalized()
            fist_offset = sum((points[j]-points[0] for j in ANCHORS), Vector())/len(ANCHORS)
            target_xy = Vector((options['x']*camera[2], options['y']))
            depth = options['distance']
        requested, limited, ray_possible = ray_reach(target_xy, depth, fist_offset, arm.head, reach, camera)
        target_position = before.lerp(requested, weight)
        target_rotation = rotation.slerp(aimed, weight)
        # A continuous bend plane prevents the alternate IK elbow branch jumping.
        direction = (before-arm.head).normalized()
        bend = elbow.head-arm.head; bend -= direction*bend.dot(direction)
        goals.append((target_position, target_rotation, bend))
        row.update(active=True, weight=weight, limited=limited, ray_possible=ray_possible,
                   requested_depth=float(depth), target=list(target_position),
                   target_xy=[target_xy.x/camera[2], target_xy.y],
                   source=observation['source'] if automatic else 'manual',
                   partial=bool(observation.get('size_lower_bound')) if automatic else False)
        metadata.append(row)
    # The existing continuous solver retains bone lengths and the captured elbow
    # side. No head/torso bridge is applied from a fist's occlusion alone.
    yield from correct_contacts(scratch, rig, channels, data, {'side': side, 'stable_ik': True},
                               goal_provider=lambda i: goals[i])
    for i, row in enumerate(metadata):
        if row['active']:
            for name, samples in channels.items():
                rig.pose.bones[name].rotation_quaternion = samples[i][1]
            view.update(); points = hand_points(rig, side)
            xy = sum((project(points[j], cameras[i]) for j in ANCHORS), Vector((0., 0.)))/len(ANCHORS)
            row['after_xy'] = [xy.x/cameras[i][2], xy.y]
            diff = Vector(row['after_xy'])-Vector(row['target_xy']); diff.x *= cameras[i][2]
            row['center_error_pixels'] = diff.length*options.get('height', 1920)
            row['after_depth'] = -(cameras[i][0]@wrist.head).z
            row['ik_error'] = (wrist.head-Vector(row['target'])).length
            if automatic and observations[i] is not None:
                actual = np.array([project(points[j], cameras[i]) for j in ANCHORS])
                desired = np.array(observations[i]['points'])[list(ANCHORS)]
                desired[:, 0] *= cameras[i][2]/data['screen']['aspect']
                actual_size = np.sqrt(np.mean(np.sum((actual-actual.mean(0))**2, axis=1)))
                desired_size = np.sqrt(np.mean(np.sum((desired-desired.mean(0))**2, axis=1)))
                row['size_ratio'] = float(actual_size/max(1e-6, desired_size))
        report.append(row)



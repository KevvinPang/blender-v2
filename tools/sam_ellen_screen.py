"""Camera-relative wrist IK, preserving the camera, body and captured fingers."""
import math

import bpy
from mathutils import Vector


class MissingScreenObservations(ValueError):
    """Valid observations file with insufficient visible face/hand samples."""


def usable_observation(face, hand):
    return (face and hand and .005 < face['width'] < 2
            and (.002 < hand['span'] < 2 or len(hand.get('points', [])) == 21)
            and 0 <= hand['confidence'] <= 1)


def validate_screen(data, side, space='FACE', coordinated=False):
    if side == 'BOTH':
        from sam_ellen_dual_screen import validate_dual
        return validate_dual(data, space, coordinated)
    if side not in ('L', 'R'):
        raise ValueError('请选择艾莲左手或右手')
    screen = data.get('screen', {})
    if (screen.get('protocol') != 'ELLEN_SCREEN_OBSERVATIONS' or screen.get('version') != 1
            or screen.get('video') != data['video'] or screen.get('fps') != data['fps']
            or len(screen.get('frames', [])) != len(data['frames'])):
        raise ValueError('此缓存没有画面对齐数据，请重新识别视频')
    visible, faces = 0, 0
    if 'aspect' in screen and (not math.isfinite(screen['aspect']) or not .05 < screen['aspect'] < 10):
        raise ValueError('Invalid source image aspect')
    for row, body in zip(screen['frames'], data['frames']):
        if abs(row['time']-body['time']) > .0001:
            raise ValueError('画面对齐数据与身体时间轴不一致')
        face, hand = row.get('face'), row.get('hands', {}).get(side)
        if face:
            values = [*face['center'], face['width']]
            if len(values) != 3 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
                raise ValueError('Invalid screen face landmarks')
            if 'scale' in face and (not math.isfinite(face['scale']) or not .005 < face['scale'] < 2):
                raise ValueError('Invalid face scale')
            faces += .005 < face['width'] < 2
        if not hand:
            continue
        values = [*hand['palm'], *hand['wrist'], hand['span'], hand['confidence']]
        if len(values) != 6 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            raise ValueError('Invalid screen hand landmarks')
        if not 0 <= hand['confidence'] <= 1:
            raise ValueError('Invalid screen landmark confidence')
        if 'points' in hand:
            points, mask = hand['points'], hand.get('in_frame', [True]*21)
            if (len(points) != 21 or len(mask) != 21
                    or any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in points)
                    or any(type(v) is not bool for v in mask)):
                raise ValueError('Invalid screen hand keypoints')
        # A side-on palm can collapse to a line. Drop that observation, not the take.
        visible += bool((space == 'FRAME' and len(hand.get('points', [])) == 21
                         and sum(hand.get('in_frame', [True]*21)) >= 4)
                        or usable_observation(face, hand))
    if visible < 2 and not (coordinated and faces >= 2):
        raise MissingScreenObservations('没有识别到所选手与脸同时出镜，请检查左右手预设或改用上半身视频')
    return screen


def camera_frames(scene, rig, camera, data, *, observation_space=False):
    """Physical output view by default; explicit source-space view for solving."""
    frame, subframe = scene.frame_current, scene.frame_subframe
    physics = getattr(scene, 'wiggle_enable', None)
    result = []
    try:
        if physics is not None:
            scene.wiggle_enable = False
        for row in data['frames']:
            from sam_ellen_sequence import timeline_frame
            f = timeline_frame(data, row['time'])
            scene.frame_set(int(f), subframe=f-int(f))
            graph = bpy.context.evaluated_depsgraph_get()
            cam = camera.evaluated_get(graph)
            if cam.data.type not in ('PERSP', 'ORTHO'):
                raise ValueError('画面对齐只支持透视或正交摄像机')
            render = scene.render
            projection = cam.calc_matrix_camera(graph, x=render.resolution_x, y=render.resolution_y,
                scale_x=render.pixel_aspect_x, scale_y=render.pixel_aspect_y)
            transform = cam.matrix_world.inverted() @ rig.evaluated_get(graph).matrix_world
            aspect = render.resolution_x*render.pixel_aspect_x/(render.resolution_y*render.pixel_aspect_y)
            if observation_space:
                from sam_ellen_shot import mapping, observation_projection
                projection = observation_projection(projection, aspect, mapping(cam))
            result.append((transform, projection, aspect, cam.data.type))
        return result
    finally:
        scene.frame_set(frame, subframe=subframe)
        if physics is not None:
            scene.wiggle_enable = physics


def project(point, camera):
    transform, projection, aspect, _ = camera
    value = projection @ (transform @ point).to_4d()
    if abs(value.w) < 1e-8:
        raise ValueError('人物与摄像机位置过近')
    return Vector(((value.x/value.w+1)*.5*aspect, (1-value.y/value.w)*.5))


def unproject(screen, depth, camera):
    transform, projection, aspect, _ = camera
    inverse = projection.inverted()
    xy = (2*screen.x/aspect-1, 1-2*screen.y)
    a = inverse @ Vector((*xy, -1, 1))
    b = inverse @ Vector((*xy, 1, 1))
    a, b = a.xyz/a.w, b.xyz/b.w
    ray = b-a
    point = a+ray*((-depth-a.z)/ray.z)
    return transform.inverted() @ point


def palm_points(rig, side):
    return [rig.pose.bones['Wrist_'+side].matrix.translation.copy(),
            *[rig.pose.bones[name+'Finger1_'+side].matrix.translation.copy()
              for name in ('Index', 'Middle', 'Ring', 'Little')]]


def observations(screen, side, smoothing=.055):
    result, previous, last, started = [], None, None, None
    for row in screen['frames']:
        face, hand, time = row.get('face'), row.get('hands', {}).get(side), row['time']
        if usable_observation(face, hand) and hand['confidence'] >= .65:
            position = (Vector(hand['palm'])-Vector(face['center']))/face['width']
            size = hand['span']/face['width']
            if previous and time-last <= .16:
                alpha = 1-math.exp(-(time-last)/smoothing)
                position = previous[0].lerp(position, alpha)
                size = previous[1]+(size-previous[1])*alpha
            else:
                started = time
            previous = (position, size)
            last = time
            weight = min(1, (time-started+.04)/.12)
        elif previous is not None and time-last < .25:
            weight = max(0, 1-(time-last)/.25)
        else:
            result.append(None)
            continue
        result.append((*previous, weight))
    return result


def correct_screen(scratch, rig, channels, data, options, cameras, report):
    if options['side'] == 'BOTH':
        from sam_ellen_dual_screen import correct_dual
        yield from correct_dual(scratch, rig, channels, data, options, cameras, report)
        return
    if options.get('landmarks') and data['screen'].get('features_version', 1) >= 2:
        from sam_ellen_screen_fit import correct_landmarks
        yield from correct_landmarks(scratch, rig, channels, data, options, cameras, report)
        return
    from sam_ellen_retarget import correct_contacts
    side = options['side']
    source = observations(validate_screen(data, side), side)

    def goal(index):
        observation = source[index]
        if observation is None:
            report.append({'time': data['frames'][index]['time'], 'active': False})
            return None
        relative, size, visibility = observation
        time = data['frames'][index]['time']-data['frames'][0]['time']
        if time < options.get('start', 0) or options.get('end', 0) > 0 and time > options['end']:
            report.append({'time': data['frames'][index]['time'], 'active': False})
            return None
        fade = .15
        visibility *= min(1, max(0, (time-options.get('start', 0))/fade))
        if options.get('end', 0) > 0:
            visibility *= min(1, max(0, (options['end']-time)/fade))
        if options['strength']*visibility <= 1e-8:
            report.append({'time': data['frames'][index]['time'], 'active': False})
            return None
        if options['mirror']:
            relative = Vector((-relative.x, relative.y))
        eyes = [rig.pose.bones['Eye_'+s].matrix.translation.copy() for s in ('L', 'R')]
        eye_center = (eyes[0]+eyes[1])*.5
        camera = cameras[index]
        eye_width = (project(eyes[0], camera)-project(eyes[1], camera)).length
        if eye_width < 1e-6:
            raise ValueError('人物正对摄像机侧面，无法可靠对齐')
        desired_xy = project(eye_center, camera)+relative*eye_width
        palm = palm_points(rig, side)
        center = sum(palm, Vector())/len(palm)
        transform = camera[0]
        depth = -(transform@center).z
        face_depth = -(transform@eye_center).z
        if min(depth, face_depth) <= 0:
            raise ValueError('人物必须位于摄像机前方')
        if camera[3] == 'PERSP':
            current_span = (project(palm[1], camera)-project(palm[-1], camera)).length
            estimate = depth*current_span/max(.01, size*eye_width)
            estimate = max(.15, min(face_depth+.25, estimate))
            depth += (estimate-depth)*options['depth']
            depth = max(.15, depth-options['front'])
            # Conservative face clearance, not a replacement for mesh collision.
            if relative.length < 1.5:
                depth = min(depth, max(.15, face_depth-.08))
            if options.get('head_depth_enabled'):
                from sam_ellen_gestures import interval_weight
                weight = interval_weight(time, options['head_depth_start'], options['head_depth_end'], .25)
                head_plane = max(.15, face_depth + options['head_depth_offset'])
                depth += (head_plane-depth)*weight
        desired_palm = unproject(desired_xy, depth, camera)
        wrist = rig.pose.bones['Wrist_'+side]
        strength = options['strength']*visibility
        target = wrist.matrix.translation+(desired_palm-center)*strength
        arm = rig.pose.bones['Arm_'+side]
        elbow = rig.pose.bones['Elbow_'+side]
        reach = (arm.bone.head_local-elbow.bone.head_local).length+(elbow.bone.head_local-wrist.bone.head_local).length
        delta = target-arm.matrix.translation
        limited = delta.length > reach*.985
        if limited:
            target = arm.matrix.translation+delta.normalized()*reach*.985
        report.append({'time': data['frames'][index]['time'], 'active': True,
                       'weight': strength, 'limited': limited, 'target_xy': list(desired_xy),
                       'before_error_eye_widths': (project(center, camera)-desired_xy).length/eye_width,
                       'desired_depth': depth, 'eye_width': eye_width})
        return target

    yield from correct_contacts(scratch, rig, channels, data, {'side': side}, goal_provider=goal)
    # Measure the generated channels, not the requested (possibly unreachable) goals.
    for i, row in enumerate(report):
        if not row['active']:
            continue
        for name, samples in channels.items():
            rig.pose.bones[name].rotation_quaternion = samples[i][1]
        scratch.view_layers[0].update()
        center = sum(palm_points(rig, side), Vector())/5
        row['after_error_eye_widths'] = (project(center, cameras[i])-Vector(row['target_xy'])).length/row['eye_width']

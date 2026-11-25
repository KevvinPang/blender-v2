"""Bake timed facial capture into Ellen's existing, editable expression shapes."""
import json
import math
from datetime import datetime
from types import SimpleNamespace

import bpy
import xr_ellen_bridge as xr
from sam_ellen_retarget import bake_steps, snapshot, restore


def face_object(scene, rig):
    return next((o for o in scene.objects if o.type == 'MESH' and o.data.shape_keys
                 and any(m.type == 'ARMATURE' and m.object == rig for m in o.modifiers)), None)


def validate_face_take(data, body):
    if data.get('protocol') != 'ELLEN_FACE_TAKE' or data.get('version') != 1:
        raise ValueError('Invalid face take protocol')
    rows = data.get('frames')
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError('面部采样不足，请重新识别面部')
    if data.get('video') != body.get('video') or abs(data.get('fps', 0)-body['fps']) > .001:
        raise ValueError('面部与身体不是同一段视频或时间轴')
    previous = -1
    for row in rows:
        time = row['time']
        if not math.isfinite(time) or time <= previous or not isinstance(row['active'], bool):
            raise ValueError('Invalid face sample times')
        if not body['frames'][0]['time']-.001 <= time <= body['frames'][-1]['time']+.001:
            raise ValueError('面部采样超出身体片段')
        previous = time
        if len(row['blendshapes']) > 60 or any(not isinstance(k, str) or not isinstance(v, (float, int))
                or not math.isfinite(v) or not 0 <= v <= 1 for k, v in row['blendshapes'].items()):
            raise ValueError('Invalid facial blendshape values')
    if not any(row['active'] for row in rows):
        raise ValueError('视频未识别到人脸，原表情保留')


def face_state(mesh, scene=None):
    keys = mesh.data.shape_keys
    ad = keys.animation_data
    state = {'mesh': mesh.name, 'action': ad.action.name if ad and ad.action else None,
            'slot': ad.action_slot.identifier if ad and ad.action_slot else None,
            'nla': {t.name: t.mute for t in ad.nla_tracks} if ad else {},
            'values': {k.name: k.value for k in keys.key_blocks}}
    if hasattr(mesh, 'ellen_expression_track'):
        track = mesh.ellen_expression_track
        state['track'] = {name: getattr(track, name) for name in
                          ('base_values', 'base_slot', 'generated_slot', 'status')}
        state['track'].update({name: getattr(track, name).name if getattr(track, name) else None
                               for name in ('target', 'base_action', 'generated_action')})
    if scene:
        state['scene'] = {'target': scene.ellen_expression_object.name if scene.ellen_expression_object else None,
                          'xr_mode': scene.xr_ellen.face_mode,
                          'xr_take': scene.xr_ellen.face_take.name if scene.xr_ellen.face_take else None}
    return state


def restore_face(state, scene=None):
    mesh = bpy.data.objects.get(state['mesh'])
    if not mesh or not mesh.data.shape_keys:
        return
    keys = mesh.data.shape_keys
    ad = keys.animation_data_create()
    ad.action = bpy.data.actions.get(state['action']) if state['action'] else None
    if ad.action and state['slot']:
        slot = next((s for s in ad.action.slots if s.identifier == state['slot']), None)
        if slot:
            ad.action_slot = slot
    for t in ad.nla_tracks:
        if t.name in state['nla']:
            t.mute = state['nla'][t.name]
    for name, value in state['values'].items():
        if name in keys.key_blocks:
            keys.key_blocks[name].value = value
    if 'track' in state:
        track = mesh.ellen_expression_track
        for name, value in state['track'].items():
            if name == 'target':
                value = bpy.data.shape_keys.get(value) if value else None
            elif name in ('base_action', 'generated_action'):
                value = bpy.data.actions.get(value) if value else None
            setattr(track, name, value)
    if scene and 'scene' in state:
        saved = state['scene']
        scene.ellen_expression_object = bpy.data.objects.get(saved['target']) if saved['target'] else None
        scene.xr_ellen.face_mode = saved['xr_mode']
        scene.xr_ellen.face_take = bpy.data.actions.get(saved['xr_take']) if saved['xr_take'] else None


def mapped_samples(face, mouth_gain, expression_gain, blink, smoothing, mouth_model=None):
    from sam_ellen_speech import mouth_weights
    previous, last_valid, last_values = None, None, None
    settings = SimpleNamespace(mouth_gain=mouth_gain)
    samples = []
    for row in face['frames']:
        time = row['time']
        if row['active']:
            arkit = {name[0].upper()+name[1:]: value for name, value in row['blendshapes'].items() if name}
            values = xr.tracked_face_weights({'arkit': arkit}, settings)
            values.update(mouth_weights(row, mouth_gain, mouth_model))
            for name in values:
                if name not in xr.MOUTH_NAMES:
                    values[name] = min(1, values[name]*expression_gain)
            if not blink:
                values['ウィンク２'] = values['ウィンク２右'] = 0
            last_valid, last_values = time, values.copy()
        elif last_values is not None:
            # Brief tracking losses hold, longer losses return gently to neutral.
            decay = max(0, 1-max(0, time-last_valid-.15)/.25)
            values = {name: value*decay for name, value in last_values.items()}
        else:
            values = {name: 0 for name in xr.FACE_NAMES}
        values.pop(xr.SHADOW_CONTROL, None)
        if previous and smoothing > 0:
            dt = time-samples[-1][0]
            for name in values:
                tau = smoothing*(.45 if name in xr.MOUTH_NAMES or 'ウィンク' in name else 1)
                alpha = 1-math.exp(-dt/max(.001, tau))
                if name != xr.TEETH_CONTROL:
                    values[name] = previous.get(name, 0)+(values[name]-previous.get(name, 0))*alpha
        previous = values.copy()
        samples.append((time, values))
    return samples


def bake_face(scene, rig, data, settings, timeline=None):
    from sam_ellen_speech import calibration, ensure_evidence, VERSION as speech_version
    validate_face_take(data['face'], data)
    face = ensure_evidence(data['face'], settings.take_path)
    validate_face_take(face, data)
    mesh = face_object(scene, rig)
    if not mesh:
        raise ValueError('找不到此人物的面部网格')
    keys = mesh.data.shape_keys
    if any(o != mesh and o.type == 'MESH' and o.data.shape_keys == keys for o in bpy.data.objects):
        raise ValueError('面部形态键被其他网格共享，请先设为单用户')
    names = set(xr.MOUTH_NAMES if settings.face_mode == 'LIPS' else xr.TRACKED_NAMES)
    names &= set(keys.key_blocks.keys())
    if not all(name in names for name in ('あ', 'う', 'お')):
        raise ValueError('当前人物尚未配置口型映射')
    ad = keys.animation_data_create()
    if any(not t.mute for t in ad.nla_tracks):
        raise ValueError('面部 NLA 正在播放，请先静音后生成')
    paths = {keys.key_blocks[n].path_from_id('value') for n in names}
    if any(c.data_path in paths for c in ad.drivers):
        raise ValueError('面部控制已有驱动器，暂不覆盖')
    before = face_state(mesh, scene)
    old = ad.action
    slot = ad.action_slot.identifier if ad.action_slot else None
    source = old
    track = mesh.ellen_expression_track
    if old and old == track.generated_action and track.base_action:
        source = track.base_action
        base_slot = next((s for s in source.slots if s.handle == track.base_slot), None)
        slot = base_slot.identifier if base_slot else None
    action = source.copy() if source else bpy.data.actions.new('SAM Facial Capture')
    source_slot = next((s for s in source.slots if s.identifier == slot), None) if source and slot else None
    action.name = 'SAM Face '+datetime.now().strftime('%Y%m%d_%H%M%S')
    action.use_fake_user = True
    actions_before = {a.as_pointer() for a in bpy.data.actions if a != action}
    try:
        if old:
            old.use_fake_user = True
        ad.action = action
        if slot:
            ad.action_slot = next(s for s in action.slots if s.identifier == slot)
        mouth_model = calibration(mesh, rig)
        samples = mapped_samples(face, settings.face_mouth_gain, settings.face_expression_gain,
                                 settings.face_blink, settings.face_smoothing, mouth_model)
        start = data['frames'][0]['time']
        for name in names:
            curve = action.fcurve_ensure_for_datablock(keys, keys.key_blocks[name].path_from_id('value'))
            curve.keyframe_points.clear()
            for modifier in list(curve.modifiers):
                curve.modifiers.remove(modifier)
            curve.mute = False
            curve.keyframe_points.add(len(samples))
            curve.keyframe_points.foreach_set('co', [v for time, values in samples
                for v in (1+(time-start)*data['fps'], values.get(name, 0))])
            for point in curve.keyframe_points:
                point.interpolation = 'LINEAR'
            curve.update()
        action['sam_face_job'] = data['job_id']
        action['sam_face_model'] = face['model']
        action['sam_face_mode'] = settings.face_mode
        action['sam_face_sample_count'] = len(samples)
        action['sam_speech_version'] = speech_version
        action['sam_mouth_calibration'] = json.dumps(mouth_model)
        if timeline:
            from sam_ellen_sequence import merge_action
            temporary = action
            action = merge_action(keys, source, source_slot, temporary, data['fps'], timeline, paths)
            bpy.data.actions.remove(temporary)
        if hasattr(scene, 'ellen_expression_object'):
            import ellen_expression_timeline as expr
            scene.ellen_expression_object = mesh
            if (any(item.enabled for item in track.items) or track.global_speech_gain != 1
                    or track.speech_speed != 1):
                # Keep the preceding generated action available for rollback/restore.
                track.generated_action = None
                expr.rebuild(scene)
        scene.xr_ellen.face_mode = 'OFF'
        scene.xr_ellen.face_take = ad.action
        return ad.action
    except Exception:
        restore_face(before, scene)
        for created in list(bpy.data.actions):
            if created.as_pointer() not in actions_before and created.users <= int(created.use_fake_user):
                bpy.data.actions.remove(created)
        raise


def combined_steps(scene, rig, data, settings, contact, gaze, screen=None, punch=None, body_support=None, reapply=False):
    from sam_ellen_sequence import make_plan, merge_action
    from sam_ellen_phone_ranges import ranges as phone_ranges
    if settings.phone_enabled:
        phone_ranges(settings)
    audio_settings = scene.sam_ellen_editing
    audio_lipsync = audio_settings.video_audio and audio_settings.video_audio_lipsync
    facial = settings.face_mode != 'OFF' and bool(data.get('face'))
    mesh = face_object(scene, rig) if facial or audio_lipsync else None
    if facial:
        validate_face_take(data['face'], data)
        if mesh is None:
            raise ValueError('找不到人物面部网格')
    state = snapshot(rig, scene)
    before = face_state(mesh, scene) if mesh else None
    saved_settings = (settings.face_take, settings.face_status, settings.previous_face_state)
    saved_cameras = scene.camera, settings.screen_camera, settings.gaze_camera
    # Include existing facial keys in the sequence length even if capture is off.
    timeline = make_plan(scene, rig, face_object(scene, rig), settings, reapply)
    if timeline:
        if 'pose' not in timeline:
            timeline['pose'] = state['pose']
        data = dict(data, _sam_timeline=timeline)
        if reapply:
            from sam_ellen_camera_fit import append_input_camera
            # Rebuild from the input camera. Reusing already corrected keys
            # adds this clip's correction twice, including old dropout jumps.
            for label,option in (('screen',screen),('gaze',gaze),('punch',punch)):
                if option is None:continue
                selected=__import__('bpy').data.objects.get(option.get('camera','')) or scene.camera
                baseline=append_input_camera(selected,timeline['start'])
                if baseline != selected:
                    updated=dict(option,camera=baseline.name)
                    if label=='screen':screen=updated
                    elif label=='gaze':gaze=updated
                    else:punch=updated
    old_body = rig.animation_data.action if rig.animation_data else None
    old_slot = rig.animation_data.action_slot if rig.animation_data else None
    checkpoint = None
    if timeline:
        if reapply and old_body and old_body.get('sam_previous_clip'):
            checkpoint = json.loads(old_body['sam_previous_clip'])
        elif not reapply:
            checkpoint_mesh = face_object(scene, rig)
            checkpoint = {'body': state,
                          'face': face_state(checkpoint_mesh, scene) if checkpoint_mesh else None,
                          'settings': {name: getattr(settings, name) for name in
                                       ('take_path', 'video', 'job_id', 'face_status')}}
            # The file picker may already point at the incoming video. Derive
            # previous capture metadata from the action actually being replaced.
            checkpoint['settings'].update(
                take=old_body.name if old_body else None,
                face_take=settings.face_take.name if settings.face_take else None,
                take_path=old_body.get('sam_take_path', '') if old_body else '',
                video=old_body.get('sam_video', '') if old_body else '',
                job_id=old_body.get('sam_job_id', '') if old_body else '')
    body = None
    success = False
    try:
        body = yield from bake_steps(scene, rig, data, settings.smoothing, settings.capture_mode,
                                     settings.inactive_pose, contact, gaze, screen, punch,
                                     head_smoothing=settings.head_smoothing, body_support=body_support,
                                     base_pose=timeline['pose'] if timeline else None,
                                     torso_capture=__import__('sam_ellen_bridge').torso_options(settings),
                                     head_pose=__import__('sam_ellen_bridge').head_pose_options(settings))
        if timeline:
            import math
            temporary = body
            body = merge_action(rig, old_body, old_slot, temporary, data['fps'], timeline)
            bpy.data.actions.remove(temporary)
            body['sam_sequence'] = json.dumps(timeline)
            scene.render.fps, scene.render.fps_base = timeline['render_fps'], timeline['fps_base']
            scene.frame_start = timeline['frame_start']
            last = timeline['start']+(data['frames'][-1]['time']-data['frames'][0]['time'])*timeline['fps']
            scene.frame_end = max(timeline['frame_end'], math.ceil(last-1e-7))
            scene.frame_set(timeline['start'])
        body_range = scene.frame_start, scene.frame_end
        if facial:
            settings.face_take = bake_face(scene, rig, data, settings, timeline)
            if not settings.previous_face_state:
                settings.previous_face_state = json.dumps(before, ensure_ascii=False)
            settings.face_status = f'面部已生成：{data["face"]["detected_samples"]}/{len(data["face"]["frames"])} 次识别'
        elif settings.face_mode != 'OFF':
            settings.face_status = '此缓存没有面部数据，请点击“补充 / 重新识别面部”'
        else:
            settings.face_status = '未改动面部动画'
        # Manual expression blocks may outlive a newly replaced, shorter take.
        # Keep those editable blocks, but play only the new video's duration.
        if not timeline:
            scene.frame_start, scene.frame_end = body_range
        scene.frame_set(scene.frame_current)
        body['sam_take_path'] = settings.take_path
        body['sam_video'] = data.get('video', '')
        if checkpoint:
            body['sam_previous_clip'] = json.dumps(checkpoint, ensure_ascii=False)
        from sam_ellen_video_audio import attach
        audio_strip = attach(scene, rig, data, body, old_body, timeline)
        from sam_ellen_phone_ranges import apply as apply_phone
        apply_phone(scene)
        if screen and screen.get('landmarks') and screen.get('space') == 'FRAME' and (timeline or settings.phone_enabled):
            from sam_ellen_validation import validate_final, summary
            final_report=[]
            for done,total in validate_final(scene,rig,data,screen,final_report):
                yield done,total
            body['sam_final_validation']=json.dumps(summary(final_report))
            body['sam_final_scene_report']=json.dumps(final_report)
        if audio_lipsync and audio_strip:
            from sam_ellen_audio_lipsync import bake_capture
            try:
                bake_capture(scene, rig, audio_strip.sound.filepath, timeline)
                settings.face_take = mesh.data.shape_keys.animation_data.action
                if not settings.previous_face_state:
                    settings.previous_face_state = json.dumps(before, ensure_ascii=False)
                settings.face_status = '已根据视频声音生成本段口型'
                audio_settings.status += '；'+settings.face_status
            except Exception as error:
                # Keep the successful body, sound and video facial capture.
                message = '音频口型生成失败，原口型保留：'+str(error)
                settings.face_status = message
                audio_settings.status += '；'+message
        success = True
        return body
    finally:
        if not success:
            scene.camera, settings.screen_camera, settings.gaze_camera = saved_cameras
            restore(rig, scene, state)
            if before:
                restore_face(before, scene)
            settings.face_take, settings.face_status, settings.previous_face_state = saved_settings
            if body and body.users <= 1:
                bpy.data.actions.remove(body)

"""One production profile for every scene opened in the project Blender.

These are runtime defaults, not animation data. Loading/saving a file never
rebakes an action. Generation and reapplication use this same public profile.
"""
import json

REVISION = 30
FRAMING_MODEL = 'fixed-camera-v4 / source-guided closeup gestures / distributed upper-arm roll / grounded root'
PROFILE = {
    'screen_enabled': True,
    'screen_landmarks': True,
    'screen_space': 'FRAME',
    'screen_body': True,
    'screen_camera_calibrate': True,
    'screen_chin_contact': True,
    'screen_strength': 1.,
    'screen_depth': .7,
    'screen_front': 0.,
    'screen_start': 0.,
    'screen_end': 0.,
    'screen_body_limit': 20.,
    'torso_enabled': True,
    'torso_screen_strength': 0.,
    'torso_strength': .75,
    'torso_limit': 15.,
    'torso_smoothing': .12,
    'hand_model': 'WILOR',
    'finger_temporal': True,
    'sample_fps': 30,
    'smoothing': .15,
    'head_pose_enabled': True,
    'gaze_head_tracking': False,
    'body_enabled': False,
}


def apply(scene):
    import bpy
    from fix_ellen_collar import protected_file
    from sam_ellen_retarget import bone_map

    if protected_file(bpy.data.filepath):
        return {'status': 'protected', 'changes': []}
    s = scene.sam_ellen
    if not s.use_shared_pipeline:
        return {'status': 'custom', 'changes': []}
    rig = s.rig
    if rig is None:
        candidates = []
        for obj in scene.objects:
            if obj.type != 'ARMATURE':
                continue
            try:
                bone_map(obj)
            except (ValueError, KeyError):
                continue
            candidates.append(obj)
        if len(candidates) == 1:
            rig = candidates[0]
        else:
            return {'status': 'choose-rig', 'changes': []}
    if rig.name not in scene.objects:
        return {'status': 'choose-rig', 'changes': []}
    bone_map(rig)
    changes = []

    def put(name, value):
        old = getattr(s, name)
        same = abs(old-value) < 1e-6 if isinstance(value, float) else old == value
        if not same:
            setattr(s, name, value)
            changes.append(name)

    put('rig', rig)
    if s.capture_mode == 'BOTH':
        # Upgrade the legacy two-hand enum without erasing clip-specific aids.
        aids = {k: getattr(s, k) for k in
                ('mouth_enabled', 'punch_enabled', 'head_depth_enabled', 'gesture_mode')}
        put('capture_mode', 'UPPER_BOTH')
        for k, value in aids.items():
            put(k, value)
    for name, value in PROFILE.items():
        put(name, value)
    if s.capture_mode in ('SELFIE_LEFT', 'SELFIE_RIGHT'):
        put('screen_side', 'L' if s.capture_mode == 'SELFIE_LEFT' else 'R')
    camera = scene.camera
    if camera is not None and camera.name in scene.objects:
        put('screen_camera', camera)
    else:
        put('screen_camera', None)
    return {'status': 'ready' if camera else 'choose-camera', 'changes': changes}


def signature(scene):
    from sam_ellen_bridge import screen_options, torso_options, head_pose_options,punch_options,contact_options,gesture_options,gaze_options,body_options
    s = scene.sam_ellen
    return {
        'revision': REVISION,
        'shared': s.use_shared_pipeline,
        'capture_mode': s.capture_mode,
        'screen': screen_options(s),
        'torso': torso_options(s),
        'head': head_pose_options(s),
        'hand_model': s.hand_model,
        'finger_temporal': s.finger_temporal,
        'smoothing':s.smoothing,
        'punch':punch_options(s),'contact':contact_options(s),
        'gesture':gesture_options(s),'gaze':gaze_options(s),'body':body_options(s),
        'phone':{'enabled':s.phone_enabled,'side':s.phone_side,'fade':s.phone_fade_frames,
                 'ranges':[[r.start_frame,r.end_frame,r.enabled] for r in s.phone_ranges]},
        'source_hash':source_hash(),
    }


_source_hash=None
def source_hash():
    """Fingerprint the loaded production revision; old actions become stale."""
    global _source_hash
    if _source_hash is None:
        import hashlib
        from pathlib import Path
        h=hashlib.sha256()
        root=Path(__file__).parent
        paths=set(root.glob('sam_ellen*.py'))|{root/'sam_punch_evidence.py',root/'sam_hand_depth_worker.py',
            root/'sam_arm_evidence.py',root/'sam_punch_video_worker.py',root/'sam_entry_video_worker.py',
            root/'enhance_sam_take_with_wilor.py',root/'wilor_temporal.py'}
        for path in sorted(paths):
            h.update(path.name.encode());h.update(path.read_bytes())
        _source_hash=h.hexdigest()
    return _source_hash


def stamp(scene, action):
    action['sam_pipeline_signature'] = json.dumps(signature(scene), ensure_ascii=False)


def action_status(scene):
    """Distinguish loaded configuration from what was actually baked."""
    s = scene.sam_ellen
    rig = s.rig
    action = rig.animation_data.action if rig and rig.animation_data else None
    if not action or not action.get('sam_job_id'):
        return '配置已就绪，导入视频即可'
    try:
        alignment = json.loads(action.get('sam_screen_alignment', 'null'))
        if s.screen_enabled and not alignment:
            return '当前动作尚未画面对齐，请重新适配'
        if alignment and s.capture_mode in ('UPPER_BOTH', 'PHONE_LEFT', 'PHONE_RIGHT') and alignment.get('side') != 'BOTH':
            return '当前动作仅对齐一只手，请重新适配'
        saved = json.loads(action.get('sam_pipeline_signature', 'null'))
        closure=json.loads(action.get('sam_projection_closure','null'))
        current=signature(scene)
        if saved!=current and not (closure and closure.get('review_signature')==current):
            return '当前动作使用旧流程或设置，请重新适配'
        validation=json.loads(action.get('sam_final_validation','null'))
        if validation and validation.get('quality_passed') is False:
            return '已生成，整臂姿态或画面检查仍有偏差，需复核'
        if closure and closure.get('review_signature')==signature(scene):
            if closure.get('continuous_motion'):
                return '已生成，仍有检查未通过，需复核' if not closure.get('quality_passed') else '已生成并通过当前整臂与画面检查'
            if closure.get('fold_recovery'):
                return '已修复画面对齐造成的折臂；其他画面偏差仍需复核' if not closure.get('quality_passed') else '已修复画面对齐造成的折臂'
            return '手臂已按当前镜头修复；近拳细节仍需复核' if not closure.get('quality_passed') else '手臂已按当前镜头修复'
        if saved != signature(scene):
            return '当前动作使用旧配置，请重新适配'
        validation=json.loads(action.get('sam_final_validation','null'))
        if validation and validation.get('quality_passed') is False:
            return '已适配，最终画面检查仍有偏差，需复核'
        return '当前动作已按此流程适配'
    except (ValueError, TypeError):
        return '当前动作的流程记录无效，请重新适配'

"""Local, asynchronous video-to-Blender SAM 3D Body preview panel."""
bl_info = {'name': 'SAM 3D Body - Ellen Preview', 'author': 'Local Tools',
           'version': (0, 42, 0), 'blender': (5, 2, 0), 'category': 'Animation'}

import json
from concurrent.futures import ThreadPoolExecutor
import subprocess
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import bpy
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from sam_ellen_gaze import SAMGazeRange, CLASSES as GAZE_CLASSES, draw_ranges, range_options
from sam_ellen_phone import CLASSES as PHONE_CLASSES
from sam_ellen_phone_ranges import SAMPhoneRange, CLASSES as PHONE_RANGE_CLASSES, changed as phone_changed, dirty as phone_dirty
from bpy_extras.io_utils import ImportHelper
from sam_ellen_retarget import bake_steps, bone_map, restore, snapshot, validate_take
from sam_ellen_face import combined_steps, restore_face
from sam_ellen_camera_fit import sync_camera_views

SERVICE_ROOT = Path.home() / 'Desktop/sam-3d-body-main'
_work = None
_poll_pool = None


class JobConnectionLost(RuntimeError):
    """Progress is unavailable; the server job must remain recoverable."""


def poll_job(state, base):
    """Only HTTP runs off-thread; scene access and baking stay on Blender's thread."""
    global _poll_pool
    now = time.monotonic()
    future = state.get('poll_future')
    if future is None:
        if now < state.get('poll_after', 0):
            return None
        address = urllib.parse.urlparse(base)
        if address.scheme != 'http' or address.hostname not in ('127.0.0.1', 'localhost'):
            raise ValueError('Only a local SAM service is allowed')
        url = base.rstrip('/')+'/sam3d/jobs/'+state['job']
        if _poll_pool is None:
            _poll_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='sam-progress')
        def read_status():
            with urllib.request.urlopen(url, timeout=5) as response:
                return json.load(response)
        state['poll_future'] = _poll_pool.submit(read_status)
        return None
    if not future.done():
        return None
    del state['poll_future']
    try:
        row = future.result()
    except urllib.error.HTTPError:
        raise
    except (OSError, urllib.error.URLError) as error:
        first = state.setdefault('poll_failed_since', now)
        state['poll_after'] = now+2
        state['poll_error'] = str(error)
        if now-first >= 120:
            raise JobConnectionLost('本地服务连续两分钟未响应，识别任务未取消；任务编号 '+state['job']) from error
        return None
    state.pop('poll_failed_since', None)
    state.pop('poll_error', None)
    state.pop('poll_after', None)
    return row


def request(scene, path, data=None, timeout=1):
    base = scene.sam_ellen.service_url.rstrip('/')
    address = urllib.parse.urlparse(base)
    if address.scheme != 'http' or address.hostname not in ('127.0.0.1', 'localhost'):
        raise ValueError('Only a local SAM service is allowed')
    body = json.dumps(data).encode('utf8') if data is not None else None
    req = urllib.request.Request(base+path, data=body,
        headers={'Content-Type': 'application/json', 'X-SAM3D-Client': 'blender'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode('utf8', errors='replace')[:1000]) from None


def redraw():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            area.tag_redraw()


def stop_work(cancel=True):
    global _work
    if not _work:
        return
    state, _work = _work, None
    future = state.get('poll_future')
    if future:
        future.cancel()
    process = state.get('alignment_upgrade') or state.get('entry_upgrade')
    if process:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        (state.get('alignment_log') or state.get('entry_log')).close()
    if state.get('bake'):
        state['bake'].close()
    if state.get('eyes'):
        state['eyes'].close()
    if cancel and state.get('job'):
        try:
            request(state['scene'], '/sam3d/jobs/'+state['job']+'/cancel', {})
        except Exception:
            pass


def video_changed(settings, context):
    current = bpy.path.abspath(settings.video) if settings.video else ''
    rig = settings.rig
    action = rig.animation_data.action if rig and rig.animation_data else None
    previous = settings.get('_timed_video', '') or (action.get('sam_video', '') if action else '')
    if previous and current and Path(previous).resolve() != Path(current).resolve():
        settings.mouth_enabled = False
        settings.punch_enabled = False
        settings.head_depth_enabled = False
        settings.gesture_mode = 'OFF'
    settings['_timed_video'] = current


def capture_mode_changed(settings, context):
    if settings.capture_mode != 'UPPER_BOTH':
        return
    settings.torso_enabled = True
    settings.torso_screen_strength = 0.
    # Share the camera/body fit, then solve both arm chains against that frame.
    settings.screen_enabled = True
    settings.screen_landmarks = True
    settings.screen_space = 'FRAME'
    settings.screen_body = True
    settings.screen_camera = settings.id_data.camera
    settings.mouth_enabled = False
    settings.punch_enabled = False
    settings.head_depth_enabled = False
    settings.gesture_mode = 'OFF'


def ensure_collar(scene):
    from fix_ellen_collar import ensure_scene
    result = ensure_scene(scene)
    if result['status'] in ('repaired', 'ready'):
        scene.sam_ellen.collar_status = '两侧衣领已恢复翘起，随肩部移动'
    elif result['status'] == 'protected':
        scene.sam_ellen.collar_status = '受保护目录：未自动修改衣领'
    else:
        scene.sam_ellen.collar_status = ''
    return result


def ensure_pipeline(scene):
    from sam_ellen_pipeline import apply
    result = apply(scene)
    scene.sam_ellen.pipeline_status = {
        'ready': '统一动捕配置已加载', 'custom': '此场景使用自定义配置',
        'protected': '受保护目录：未同步场景配置',
        'choose-rig': '请选择目标人物骨架',
        'choose-camera': '请选择场景摄像机',
    }[result['status']]
    return result


def shared_pipeline_changed(settings, context):
    if settings.use_shared_pipeline:
        ensure_pipeline(settings.id_data)


def profile_prop(layout, settings, name, **kwargs):
    from sam_ellen_pipeline import PROFILE
    row = layout.row()
    row.enabled = not settings.use_shared_pipeline or name not in PROFILE
    row.prop(settings, name, **kwargs)


def sync_capture_camera(scene):
    """Bind an explicit shared-profile import to the current scene's shot.

    Protected character files skip automatic load-time profile migration.
    That must not leave an explicitly requested import using another set's
    saved camera. Preserve independently selected gaze/punch targets.
    """
    settings = scene.sam_ellen
    camera = scene.camera
    if (not settings.use_shared_pipeline or camera is None
            or camera.type != 'CAMERA' or camera.name not in scene.objects):
        return
    previous = settings.screen_camera
    if previous == camera:
        return
    settings.screen_camera = camera
    if previous is not None:
        for name in ('gaze_camera', 'punch_camera'):
            if getattr(settings, name) == previous:
                setattr(settings, name, camera)


def begin_bake(scene, path, reapply=False, entry_completed=False):
    global _work
    sync_capture_camera(scene)
    ensure_pipeline(scene)
    settings = scene.sam_ellen
    rig = settings.rig
    if rig is None or rig.type != 'ARMATURE':
        raise ValueError('Choose the character armature')
    bone_map(rig)
    ensure_collar(scene)
    data = validate_take(json.loads(Path(path).read_text(encoding='utf8')))
    from sam_ellen_finger_repair import prepare_experiment, restore_experiment
    data = restore_experiment(data)
    if reapply and (not settings.video or Path(bpy.path.abspath(settings.video)).resolve() != Path(data['video']).resolve()):
        raise ValueError('所选视频与上次成功缓存不同，请点击“生成 / 重新识别”；旧缓存未应用')
    required_features = (5 if settings.punch_enabled and settings.punch_automatic else 4 if getattr(settings,'screen_chin_contact',True) and settings.screen_enabled
                         and settings.screen_landmarks and settings.screen_space=='FRAME'
                         else 3 if getattr(settings, 'head_pose_enabled', True) else 2)
    if settings.punch_enabled and settings.punch_automatic or (settings.screen_enabled
            and settings.screen_landmarks and settings.screen_space=='FRAME'):
        required_features=7
    if ((settings.punch_enabled and settings.punch_automatic or settings.screen_enabled and settings.screen_landmarks or settings.torso_enabled
            or getattr(settings, 'head_pose_enabled', True))
            and data.get('screen', {}).get('features_version', 1) < required_features):
        # Upgrade only the inexpensive image observations; keep the SAM/WiLoR cache.
        source = Path(path)
        upgraded = source.with_name(source.stem+'_camera_v3.json')
        stamp = [source.stat().st_mtime_ns, source.stat().st_size]
        if upgraded.is_file():
            cached = json.loads(upgraded.read_text(encoding='utf8'))
            if cached.get('_camera_input') == stamp and cached.get('screen', {}).get('features_version', 1) >= required_features:
                return begin_bake(scene, str(upgraded), reapply)
        if not Path(data['video']).is_file():
            raise ValueError('源视频已移动或删除，无法补充画面关键点；已保留现有动作')
        python = Path.home()/'Desktop/小小布朗尼/tools/motion_venv/Scripts/python.exe'
        worker = SERVICE_ROOT/'local_app/alignment_worker.py'
        output = source.with_name(source.stem+'_observations_v2.json')
        log = source.with_name(source.stem+'_camera_v3.log').open('w', encoding='utf8')
        try:
            process = subprocess.Popen([str(python), str(worker), '--body', str(source), '--output', str(output)],
                stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        except Exception:
            log.close()
            raise
        settings.status = '正在补充头部朝向与画面关键点（保留现有动作）'
        _work = {'scene': scene, 'alignment_upgrade': process, 'alignment_log': log,
                 'alignment_output': output, 'alignment_destination': upgraded,
                 'alignment_data': data, 'alignment_stamp': stamp, 'reapply': reapply,
                 'alignment_started': time.monotonic()}
        return
    explicit_punch=settings.punch_enabled and settings.punch_automatic
    image_fist_sides=[]
    if settings.screen_enabled and settings.screen_landmarks and settings.screen_space=='FRAME':
        from sam_punch_evidence import detect,approach_candidates
        image_fist_sides=[side for side in ('L','R') if detect(data['screen'],side)[0] or approach_candidates(data['screen'],side)]
    frame_palms=settings.screen_enabled and settings.screen_landmarks and settings.screen_space=='FRAME'
    if explicit_punch or frame_palms:
        from sam_hand_depth_worker import signature as hand_depth_signature
        depth=data.get('screen',{}).get('hand_pose_depth',{})
        depth_stamp=hand_depth_signature(data)
        if depth.get('version')!=1 or depth.get('input_signature')!=depth_stamp:
            source=Path(path);destination=source.with_name(source.stem+'_hand_depth_v1.json')
            if destination.is_file():
                cached=json.loads(destination.read_text(encoding='utf8'))
                if cached.get('screen',{}).get('hand_pose_depth',{}).get('input_signature')==depth_stamp:
                    return begin_bake(scene,str(destination),reapply)
            python=Path.home()/'Desktop/小小布朗尼/tools/motion_venv/Scripts/python.exe'
            worker=Path(__file__).with_name('sam_hand_depth_worker.py')
            output=source.with_name(source.stem+'_hand_depth_screen.json')
            log=source.with_name(source.stem+'_hand_depth.log').open('w',encoding='utf8')
            try:
                process=subprocess.Popen([str(python),str(worker),str(source),str(output)],
                    stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            except Exception:
                log.close();raise
            settings.status='正在补充视频手部深度，校验掌心、手背与手指方向'
            _work={'scene':scene,'alignment_upgrade':process,'alignment_log':log,
                'alignment_output':output,'alignment_destination':destination,'alignment_data':data,
                'alignment_stamp':data.get('_camera_input'),'reapply':reapply,'alignment_started':time.monotonic()}
            return
    if explicit_punch or image_fist_sides:
        from sam_punch_evidence import mask_signature, MASK_VERSION
        source = Path(path)
        candidates=([hand_side(settings) if hand_side(settings)!='BOTH' else settings.punch_side] if explicit_punch else [])+image_fist_sides
        candidates=list(dict.fromkeys(candidates))
        if not Path(data['video']).is_file():
            raise ValueError('源视频已移动或删除，无法校验拳头轮廓；已保留原动作')
        signature = mask_signature(data)
        tracking = data.get('screen', {}).get('punch_tracking', {})
        side=next((side for side in candidates if side not in tracking.get('completed_sides',[])),candidates[0])
        ready = (tracking.get('version') == 2 and tracking.get('mask_version') == MASK_VERSION
                 and side in tracking.get('completed_sides', [])
                 and tracking.get('input_signature') == signature)
        if not ready:
            destination = source.with_name(source.stem+f'_punch_video_v3_{side}.json')
            if destination.is_file():
                cached = json.loads(destination.read_text(encoding='utf8'))
                evidence = cached.get('screen', {}).get('punch_tracking', {})
                if (evidence.get('input_signature') == signature and evidence.get('mask_version') == MASK_VERSION
                        and side in evidence.get('completed_sides', [])):
                    return begin_bake(scene, str(destination), reapply)
            python = Path.home()/'Downloads/WiLoR-main/.venv/Scripts/python.exe'
            worker = Path.home()/'Desktop/小小布朗尼/tools/sam_punch_video_worker.py'
            if not python.is_file() or not worker.is_file():
                raise ValueError('缺少本地拳头视频轮廓识别环境')
            output = source.with_name(source.stem+f'_punch_masks_{side}.json')
            log = source.with_name(source.stem+f'_punch_masks_{side}.log').open('w',encoding='utf8')
            try:
                process = subprocess.Popen([str(python),str(worker),'--body',str(source),
                    '--output',str(output),'--side',side],stdout=log,stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW)
            except Exception:
                log.close()
                raise
            settings.status = '正在跟踪视频拳头轮廓，校验前伸、贴近镜头与收回'
            _work = {'scene':scene,'alignment_upgrade':process,'alignment_log':log,
                     'alignment_output':output,'alignment_destination':destination,
                     'alignment_data':data,'alignment_stamp':data.get('_camera_input'),
                     'reapply':reapply,'alignment_started':time.monotonic()}
            return
    if settings.hand_model == 'WILOR':
        report = data.get('wilor_refinement')
        required = ('L', 'R') if hand_side(settings) == 'BOTH' else (hand_side(settings),)
        if not report:
            raise ValueError('请先点击“补充 / 重新识别手指”，或将手指模型改为 SAM')
        temporal = getattr(settings, 'finger_temporal', True)
        needs_temporal = any(
            report['hands'].get(side, {}).get('temporal_filter', {}).get('version') != 2
            or report['hands'].get(side, {}).get('temporal_filter', {}).get('enabled') != temporal
            for side in required)
        if needs_temporal or report.get('finger_geometry_version') != 2 or (report.get('palm_orientation_version') != 1 and not report.get('fallback')):
            # Rebuild from original SAM + raw WiLoR; toggling never compounds edits.
            predictions = Path(path).with_name('wilor_predictions.json')
            worker = Path.home()/'Desktop/小小布朗尼/tools/enhance_sam_take_with_wilor.py'
            python = Path.home()/'Downloads/WiLoR-main/.venv/Scripts/python.exe'
            if not predictions.is_file() or not worker.is_file() or not python.is_file():
                raise ValueError('缺少 WiLoR 原始缓存，请点击“补充 / 重新识别手掌与手指”')
            upgrade_side = hand_side(settings)
            upgraded = Path(path).with_name(f'take_wilor_geometry_v2_{upgrade_side}_{int(temporal)}.json')
            command = [str(python), str(worker), '--take', str(path),
                '--predictions', str(predictions), '--output', str(upgraded),
                '--side', upgrade_side]
            if not temporal:
                command.append('--no-temporal-filter')
            job = subprocess.run(command, capture_output=True, timeout=300,
                creationflags=subprocess.CREATE_NO_WINDOW)
            if job.returncode:
                raise RuntimeError('手指缓存更新失败；原动作未更改，请重新识别手部')
            path = str(upgraded)
            data = validate_take(json.loads(upgraded.read_text(encoding='utf8')))
    elif data.get('wilor_original'):
        for row, original in zip(data['frames'], data['wilor_original']):
            for prop in ('positions', 'rotations'):
                for k, value in original[prop].items():
                    row[prop][int(k)] = value
        data.pop('wilor_refinement', None)
    if settings.hand_model == 'WILOR':
        from sam_ellen_arm_visibility import merge_cached_wilor
        merge_cached_wilor(data,path)
    else:
        data.get('screen',{}).pop('wilor_hands',None)
    if settings.screen_enabled and settings.screen_landmarks and settings.screen_space=='FRAME':
        from sam_ellen_entry import load_cache
        if not load_cache(data,path):
            if entry_completed:
                raise RuntimeError('入镜检查已结束，但结果与当前手部数据不匹配；已停止重复校验，请重新适配')
            worker=Path(__file__).with_name('sam_entry_video_worker.py')
            python=Path(__file__).parent/'motion_venv/Scripts/python.exe'
            output=Path(path).with_name('hand_entry_v1.json')
            if Path(data['video']).is_file() and python.is_file():
                log=output.with_suffix('.log').open('w',encoding='utf8')
                try:
                    command=[str(python),str(worker),str(path),str(output)]+(['--no-wilor'] if settings.hand_model!='WILOR' else [])
                    process=subprocess.Popen(command,
                        stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                except Exception:
                    log.close();raise
                settings.progress=0
                settings.status='正在校验手部首次露出时间与入镜方向（识别完成，动画尚未生成）'
                _work={'scene':scene,'entry_upgrade':process,'entry_log':log,'entry_path':str(path),
                       'entry_output':output,'reapply':reapply,'entry_started':time.monotonic()}
                return
    if settings.hand_model == 'WILOR' and getattr(settings, 'finger_reliability', False):
        settings.status = '正在进行实验性手指修复'
        data = prepare_experiment(data, path, enabled=True, side=hand_side(settings))
    from sam_ellen_gestures import apply_gesture
    data = apply_gesture(data, gesture_options(settings))
    import xr_ellen_bridge as xr
    if xr._runtime:
        if xr._runtime.recording:
            raise RuntimeError('Finish the XR recording before importing a SAM take')
        xr.stop()
    if not settings.previous_state:
        settings.previous_state = json.dumps(snapshot(rig, scene), ensure_ascii=False)
        settings.previous_rig = rig
    elif settings.previous_rig != rig:
        raise RuntimeError('Restore the previous character before changing the target')
    alignment = screen_options(settings)
    warnings = []
    if alignment:
        from sam_ellen_screen import validate_screen, MissingScreenObservations
        try:
            validate_screen(data, alignment['side'], alignment.get('space', 'FACE'),
                            alignment.get('coordinated_body') and alignment.get('landmarks'))
        except MissingScreenObservations:
            alignment = None
            warnings.append('手和脸的共同可见数据不足，本段已跳过画面对齐')
        else:
            sides = ('L', 'R') if alignment['side'] == 'BOTH' else (alignment['side'],)
            for side in sides:
                if not any(r.get('hands', {}).get(side) for r in data['screen']['frames']):
                    warnings.append(('左手' if side == 'L' else '右手')+'未出镜，保留该手原动作并继续其余画面对齐')
    settings.take_path = str(Path(path).resolve())
    settings.progress = 0
    settings.status = '正在生成艾莲动画'
    _work = {'scene': scene, 'rig': rig, 'warnings': warnings, 'bake': combined_steps(
        scene, rig, data, settings, contact_options(settings), gaze_options(settings), alignment,
        punch_options(settings), body_options(settings), reapply=reapply)}


def body_options(s):
    if not s.body_enabled:
        return None
    return {'strength': s.body_strength, 'amplitude': s.body_amplitude,
            'speed': s.body_speed, 'randomness': s.body_randomness, 'seed': s.body_seed,
            'model': 'additive-torso-v1'}


def head_pose_options(s):
    if not getattr(s, 'head_pose_enabled', True):
        return None
    return {'strength': 1., 'smoothing': .045, 'mirror': s.screen_mirror,
            'camera': s.screen_camera.name if s.screen_camera else '',
            'model': 'camera-relative-face-v1'}


def torso_options(s):
    if not getattr(s, 'torso_enabled', False):
        return None
    return {'strength': s.torso_strength, 'screen_strength': s.torso_screen_strength,
            'max_angle': s.torso_limit, 'smoothing': s.torso_smoothing,
            'camera': s.screen_camera.name if s.screen_camera else '',
            'mirror': s.screen_mirror, 'model': 'fixed-feet-captured-torso-v1'}


def body_settings_pending(s):
    action = s.rig.animation_data.action if s.rig and s.rig.animation_data else None
    if not action:
        return True
    try:
        old, new = json.loads(action.get('sam_body_support', 'null')), body_options(s)
        if (old is None) != (new is None):
            return True
        return bool(new and any(abs(old.get(k, -100)-v) > 1e-5 if isinstance(v, (int, float))
                               else old.get(k) != v for k, v in new.items()))
    except (ValueError, TypeError):
        return True


def screen_options(s):
    if not s.screen_enabled:
        return None
    from sam_ellen_pipeline import FRAMING_MODEL
    return {'side': 'BOTH' if s.capture_mode in ('UPPER_BOTH', 'PHONE_LEFT', 'PHONE_RIGHT') else s.screen_side,
            'framing_model': FRAMING_MODEL,
            'camera_mode': s.camera_mode, 'semantic_contacts': s.semantic_contacts,
            'shot_preset': s.shot_preset, 'shot_scale': s.shot_scale, 'shot_vertical': s.shot_vertical,
            'strength': s.screen_strength, 'depth': s.screen_depth,
            'chin_contact':getattr(s,'screen_chin_contact',True),
            'landmarks': s.screen_landmarks, 'space': s.screen_space,
            'coordinated_body': s.screen_body, 'body_limit': s.screen_body_limit,
            'calibrate_camera': s.screen_camera_calibrate,
            'model': ('image-landmarks-v3' if s.screen_space == 'FRAME' else 'native-hand-proportions-v2')
                     if s.screen_landmarks else 'legacy-palm',
            'height': s.id_data.render.resolution_y,
            'front': s.screen_front/100, 'mirror': s.screen_mirror,
            'start': s.screen_start, 'end': s.screen_end,
            'camera': s.screen_camera.name if s.screen_camera else '',
            'head_depth_enabled': s.head_depth_enabled,
            'head_depth_start': s.head_depth_start, 'head_depth_end': s.head_depth_end,
            'head_depth_offset': s.head_depth_offset/100}


def gesture_options(s):
    return {'mode': s.gesture_mode, 'side': hand_side(s) if hand_side(s) != 'BOTH' else s.screen_side,
            'start': s.gesture_start, 'end': s.gesture_end, 'strength': s.gesture_strength}


def hand_side(s):
    return {'SELFIE_LEFT': 'L', 'SELFIE_RIGHT': 'R'}.get(s.capture_mode, 'BOTH')


def punch_options(s):
    if not s.punch_enabled:
        return None
    side = hand_side(s) if s.punch_automatic and hand_side(s) != 'BOTH' else s.punch_side
    return {'side': side, 'automatic': s.punch_automatic, 'mirror': s.screen_mirror,
            'height': s.id_data.render.resolution_y,
            'start': s.punch_start, 'peak': s.punch_peak,
            'release': s.punch_release, 'end': s.punch_end, 'reference': s.punch_reference,
            'distance': s.punch_distance/100, 'x': s.punch_x, 'y': s.punch_y,
            'strength': s.punch_strength, 'stabilize': s.punch_stabilize,
            'camera': s.punch_camera.name if s.punch_camera else ''}


def contact_options(s):
    if not s.mouth_enabled:
        return None
    return {'side': s.mouth_side, 'start': s.mouth_start, 'end': s.mouth_end,
            'fade': s.mouth_fade, 'strength': s.mouth_strength,
            'x': s.mouth_x/100, 'front': s.mouth_front/100, 'z': s.mouth_z/100}


def gaze_options(s):
    if not s.gaze_enabled:
        return None
    return {'camera': s.gaze_camera.name if s.gaze_camera else '',
            'head_tracking': s.gaze_head_tracking,
            'motion': s.gaze_motion, 'eyes': s.gaze_eyes,
            'noise_enabled': s.gaze_noise_enabled,
            'noise_amplitude': s.gaze_noise_amplitude,
            'noise_speed': s.gaze_noise_speed,
            'noise_head': s.gaze_noise_head,
            'noise_eyes': s.gaze_noise_eyes,
            'noise_seed': s.gaze_noise_seed,
            'noise_randomness': s.gaze_noise_randomness,
            'noise_model': 'smooth-random-v1',
            'away_ranges': range_options(s), 'away_fade': s.gaze_away_fade}


def head_settings_pending(s):
    action = s.rig.animation_data.action if s.rig and s.rig.animation_data else None
    if not action:
        return True
    try:
        old = json.loads(action.get('sam_camera_gaze', 'null'))
        new = gaze_options(s)
        # Eye intervals/follow strength have a separate lightweight apply path.
        head_keys = ('camera', 'motion', 'noise_enabled', 'noise_amplitude', 'noise_speed',
                     'noise_head', 'noise_seed', 'noise_randomness')
        old = {k: old.get(k) for k in head_keys} if old and old.get('head_tracking') else None
        new = {k: new.get(k) for k in head_keys} if new and new.get('head_tracking') else None
        if (old is None) != (new is None):
            return True
        if new and any(abs(old.get(k, -100)-v) > 1e-5 if isinstance(v, (int, float))
                       else old.get(k) != v for k, v in new.items()):
            return True
        return (abs(action.get('sam_head_smoothing', 0)-s.head_smoothing) > 1e-5
                or json.loads(action.get('sam_head_pose', 'null')) != head_pose_options(s))
    except (ValueError, TypeError):
        return True


def tick():
    global _work
    if not _work:
        return None
    scene = _work['scene']
    settings = scene.sam_ellen
    try:
        if 'eyes' in _work:
            deadline = time.monotonic() + .04
            while time.monotonic() < deadline:
                try:
                    done, total = next(_work['eyes'])
                    settings.progress = min(.99, done / max(total, 1))
                    settings.status = f'仅更新眼睛视线 {done}/{total}'
                except StopIteration as result:
                    settings.progress = 1
                    settings.status = f'眼睛视线已更新：{result.value} 帧；身体、头部、口型和声音保留'
                    _work = None
                    if not bpy.app.background:
                        bpy.ops.ed.undo_push(message='仅应用眼睛视线')
                    redraw()
                    return None
            redraw()
            return .01
        if 'entry_upgrade' in _work:
            process=_work['entry_upgrade']
            if process.poll() is None:
                elapsed=time.monotonic()-_work['entry_started']
                if elapsed>120:
                    raise RuntimeError('入镜画面检查超时；已保留原动作')
                settings.status=f'正在校验手部入镜 / 已用 {elapsed:.0f} 秒（动画尚未生成）'
                redraw()
                return .2
            _work['entry_log'].close()
            if process.returncode:raise RuntimeError('入镜画面检查失败，请查看 '+str(_work['entry_log'].name))
            path,reapply=_work['entry_path'],_work['reapply'];_work=None
            begin_bake(scene,path,reapply,entry_completed=True)
            return .01
        if 'alignment_upgrade' in _work:
            if time.monotonic()-_work['alignment_started'] > 600:
                raise RuntimeError('画面关键点补充超时；已保留原动作')
            process = _work['alignment_upgrade']
            if process.poll() is None:
                return .2
            _work['alignment_log'].close()
            if process.returncode:
                raise RuntimeError('画面关键点识别失败，请查看 '+str(_work['alignment_log'].name))
            payload = _work['alignment_data']
            payload['screen'] = json.loads(_work['alignment_output'].read_text(encoding='utf8'))
            payload['_camera_input'] = _work['alignment_stamp']
            destination, reapply = _work['alignment_destination'], _work['reapply']
            destination.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding='utf8')
            _work = None
            begin_bake(scene, str(destination), reapply)
            return .01
        if 'bake' in _work:
            # Bounded work per UI tick; no bpy calls on a background thread.
            for _ in range(3):
                try:
                    done, total = next(_work['bake'])
                    settings.progress = min(.99, done / total)
                    settings.status = ('生成动画：最终修正 / 校验中' if done >= total
                                       else f'生成动画 {done}/{total}')
                except StopIteration as result:
                    settings.take = result.value
                    from sam_ellen_pipeline import stamp
                    stamp(scene, result.value)
                    settings.progress = 1
                    sequence = json.loads(result.value.get('sam_sequence', 'null'))
                    if sequence:
                        settings.status = f'已接入：第 {sequence["start"]} 帧起 / 时间轴至 {scene.frame_end} 帧'
                    else:
                        settings.status = f'已就绪：{scene.frame_end} 帧 / {result.value["sam_sample_count"]} 次采样'
                    hand_report = json.loads(result.value.get('sam_wilor', 'null'))
                    if hand_report and hand_report.get('warning'):
                        settings.status += '；'+hand_report['warning']
                    if hand_report:
                        count = sum(len(h.get('temporal_filter', {}).get('accepted', []))
                                    for h in hand_report.get('hands', {}).values())
                        if count:
                            settings.status += f'；已过滤 {count} 段手指短时误判'
                    experiment = json.loads(result.value.get('sam_finger_reliability', 'null'))
                    if experiment and experiment.get('enabled'):
                        settings.status += f'；实验修复 {len(experiment["accepted"])} 段手指分歧'
                        if experiment.get('warning'):
                            settings.status += '（证据不足，已保留原姿势）'
                    result.value['sam_import_warnings'] = json.dumps(_work.get('warnings', []), ensure_ascii=False)
                    for warning in _work.get('warnings', []):
                        settings.status += '；'+warning
                    screen_report = json.loads(result.value.get('sam_screen_report', '[]'))
                    hand_rows = [hand for row in screen_report
                                 for hand in (row['hands'].values() if 'hands' in row else (row,))]
                    limited = sum(any(h.get('limited') for h in row['hands'].values())
                                  if 'hands' in row else bool(row.get('limited')) for row in screen_report)
                    if limited:
                        settings.status += f'；{limited} 帧手部目标超出臂长，已限制'
                    punch_settings = json.loads(result.value.get('sam_punch', 'null'))
                    if punch_settings and punch_settings.get('automatic'):
                        count = len(punch_settings.get('segments', []))
                        settings.status += f'；自动找到 {count} 段出拳' if count else '；未找到证据充分的出拳，已保留原动作'
                        if any(r.get('incomplete') for r in punch_settings.get('segments', [])):
                            settings.status += '（部分拳头画面证据不足，未强行补全）'
                        punch_rows = json.loads(result.value.get('sam_punch_report', '[]'))
                        if any(r.get('partial') for r in punch_rows):
                            settings.status += '（拳头部分出框，大小仅作下界参考）'
                    validation = json.loads(result.value.get('sam_final_validation','null'))
                    if validation and not validation.get('safety_passed',True):
                        settings.status += '；安全检查未通过，请查看验证报告'
                    elif validation and not validation.get('quality_passed',True):
                        settings.status += '；安全检查通过，动作或画面对齐仍需复核'
                    visible = [r for r in hand_rows if r.get('model') == 'image-landmarks-v3'
                               and r.get('weight', 0) >= .99 and not r.get('held')]
                    if visible:
                        error = (validation['hand_mean_px'] if validation and validation.get('hand_mean_px') is not None
                                 else sum(r['after_error_pixels'] for r in visible)/len(visible))
                        settings.status += f'；最终手部关键点平均误差 {error:.0f} 像素'
                        if any(r.get('overrides') for r in visible):
                            settings.status += '（含捂嘴/出拳覆盖）'
                    if any('hands' in r for r in screen_report):
                        for side, label in (('L', '左手'), ('R', '右手')):
                            measured = [r['hands'][side] for r in screen_report
                                        if r['hands'][side].get('weight', 0) >= .99
                                        and not r['hands'][side].get('held')
                                        and 'after_center_error_pixels' in r['hands'][side]]
                            if measured:
                                error = sum(r['after_center_error_pixels'] for r in measured)/len(measured)
                                settings.status += f'；{label}中心偏差 {error:.0f} 像素'
                            else:
                                settings.status += f'；{label}画面证据不足，已保留捕捉动作'
                    calibration = json.loads(result.value.get('sam_camera_calibration', 'null'))
                    if calibration:
                        settings.status += '；沿用已有校准镜头' if calibration.get('reused_for_append') else '；已校准基础构图（原机位已保留）'
                    body_report = json.loads(result.value.get('sam_coordinated_body_report', '[]'))
                    visible_body = [r for r in body_report if r.get('active')]
                    if visible_body:
                        error = (validation['face_mean_px'] if validation and validation.get('face_mean_px') is not None
                                 else sum(r['after_center_px'] for r in visible_body)/len(visible_body))
                        settings.status += f'；脸部中心平均偏差 {error:.0f} 像素'
                    sync_camera_views(scene)
                    _work = None
                    redraw()
                    return None
            redraw()
            return .01
        if 'launcher' in _work:
            process = _work['launcher']
            if time.monotonic()-_work['started'] > 100:
                raise RuntimeError('Local service startup timed out')
            if process.poll() is None:
                return .5
            output = process.communicate()[0].decode('utf8', errors='replace')
            if process.returncode:
                raise RuntimeError('Cannot start SAM service: '+output[-1000:])
            urls = [row.strip() for row in output.splitlines() if row.strip().startswith('http://127.0.0.1:')]
            if not urls:
                raise RuntimeError('The launcher returned no local service address')
            settings.service_url = urls[-1]
            del _work['launcher']
        if not _work.get('job'):
            health = request(scene, '/sam3d-health')
            if health.get('blender_bridge') != 1:
                raise RuntimeError('Restart the SAM service to load the Blender bridge update')
            if health.get('sequential_video_decode') != 1:
                raise RuntimeError('本地服务仍为旧版，请重新点击“生成 / 重新识别”以连接修复版服务')
            if settings.face_mode != 'OFF' and health.get('face_capture') != 1:
                raise RuntimeError('请重启 SAM 本地服务以启用面部捕捉')
            if settings.screen_enabled and health.get('screen_alignment') != 1:
                raise RuntimeError('请重启 SAM 本地服务以启用手部画面对齐')
            if settings.hand_model == 'WILOR' and health.get('wilor_hands') != 1:
                raise RuntimeError('请重启 SAM 本地服务以启用 WiLoR 手指增强')
            if _work.get('face_only') or _work.get('hands_only'):
                body = json.loads(Path(settings.take_path).read_text(encoding='utf8'))
                if Path(bpy.path.abspath(settings.video)).resolve() != Path(body['video']).resolve():
                    raise ValueError('视频已更换，请点击“导入视频并生成动作”')
                if _work.get('hands_only'):
                    row = request(scene, '/sam3d/jobs/'+body['job_id']+'/hands', {'side': hand_side(settings)})
                else:
                    row = request(scene, '/sam3d/jobs/'+body['job_id']+'/face', {'face_rate': settings.face_fps})
            else:
                row = request(scene, '/sam3d/jobs', {'video': bpy.path.abspath(settings.video),
                              'rate': settings.sample_fps, 'start': settings.start_seconds, 'end': settings.end_seconds,
                              'face': settings.face_mode != 'OFF', 'face_rate': settings.face_fps,
                              'alignment': settings.screen_enabled or settings.torso_enabled or settings.punch_enabled,
                              'hands': hand_side(settings) if settings.hand_model == 'WILOR' else 'OFF'})
            _work['job'] = row['id']
            settings.job_id = row['id']
        row = poll_job(_work, settings.service_url)
        if row is None:
            if _work.get('poll_error'):
                settings.status = '本地服务响应暂时延迟，正在重新连接（识别任务继续）'
                redraw()
            return .2
        settings.progress = row['done']/max(1, row['total'])
        phase = {'face': '面部识别', 'alignment': '手部画面识别', 'hands': 'WiLoR 手指识别'}.get(row.get('phase'), '身体识别')
        settings.status = f'{phase} {row["done"]}/{row["total"]} / 已用 {row["elapsed_seconds"]:.0f} 秒'
        if row['state'] in ('error', 'cancelled'):
            raise RuntimeError(row['error'] or '已取消')
        if row['state'] == 'complete':
            begin_bake(scene, row['take_path'], reapply=_work.get('face_only', False) or _work.get('hands_only', False))
        redraw()
        return .5
    except JobConnectionLost as error:
        stop_work(cancel=False)
        settings.status = str(error)+'；已保留旧动作'
        redraw()
        return None
    except Exception as error:
        import traceback
        traceback.print_exc()
        eye_update = 'eyes' in (_work or {})
        stop_work(cancel=True)
        settings.status = ('眼睛视线未更改：' if eye_update else '新动作未导入，已保留旧动作：')+str(error)
        redraw()
        return None


def run_timer():
    if not bpy.app.timers.is_registered(tick):
        bpy.app.timers.register(tick, first_interval=.1)


class SAMSettings(bpy.types.PropertyGroup):
    use_shared_pipeline: BoolProperty(name='使用统一动捕配置', default=True, update=shared_pipeline_changed,
        description='所有场景共用插件中的画面对齐、双手、手指稳定和头部规则；关闭后可自行对比旧设置')
    pipeline_status: StringProperty(default='', options={'SKIP_SAVE'})
    import_mode: EnumProperty(name='动作导入方式', default='REPLACE', items=[
        ('REPLACE', '替换当前动作', '重新生成整段人物动作'),
        ('APPEND', '追加到末尾', '保留已有动作，将新视频和口型接在末尾')])
    transition_frames: IntProperty(name='衔接帧数', default=6, min=0, max=120,
                                  description='前一段结束到新动作开始的间隔；0 为下一帧直接切换')
    capture_mode: EnumProperty(name='拍摄预设', default='UPPER_BOTH', update=capture_mode_changed, items=[
        ('BOTH', '上半身双手（原模式 / 下半身固定）', '角色自身左右；保持原有双手和上半身驱动', 0),
        ('SELFIE_LEFT', '自拍左手（右手固定 / 下半身固定）', '只使用角色自身左手动捕，右手与腿部固定', 1),
        ('SELFIE_RIGHT', '自拍右手（左手固定 / 下半身固定）', '只使用角色自身右手动捕，左手与腿部固定', 2),
        ('UPPER_BOTH', '双手动捕（下半身固定）',
         '双臂、双手和上半身参与动捕，共用镜头对齐双手，保持下半身与站位', 3)])
    phone_enabled: BoolProperty(name='持手机（子选项）', default=False, update=phone_changed,
        description='只在指定帧区间显示手机并叠加握姿；其他帧完整使用主预设')
    phone_side: EnumProperty(name='持握手', default='R', update=phone_changed,
        items=[('L', '角色左手', ''), ('R', '角色右手', '')])
    phone_ranges: CollectionProperty(type=SAMPhoneRange)
    phone_range_index: IntProperty(default=0)
    phone_fade_frames: IntProperty(name='握姿渐入 / 渐出（帧）', default=6, min=0, max=120,
        update=phone_dirty, description='过渡发生在区间内；0 直接切换，区间外不改变主预设')
    phone_status: StringProperty(options={'SKIP_SAVE'})
    inactive_pose: EnumProperty(name='另一手', default='DOWN', items=[
        ('DOWN', '自然垂下', '未出镜手臂使用固定垂手姿势，不使用模型猜测'),
        ('CURRENT', '保持当前姿势', '固定为点击生成或重新适配时的手臂局部姿势，不是锁定世界坐标')])
    rig: PointerProperty(name='人物骨架', type=bpy.types.Object,
                          poll=lambda self, obj: obj.type == 'ARMATURE')
    collar_status: StringProperty(default='')
    video: StringProperty(name='视频', subtype='FILE_PATH', update=video_changed)
    service_url: StringProperty(name='本地服务', default='http://127.0.0.1:7872')
    sample_fps: IntProperty(name='每秒采样', min=1, max=30, default=30)
    start_seconds: FloatProperty(name='起点（秒）', min=0, default=0)
    end_seconds: FloatProperty(name='终点（0 为片尾）', min=0, default=0)
    smoothing: FloatProperty(name='轻微平滑', min=0, max=.8, default=.15)
    hand_model: EnumProperty(name='手部模型', default='WILOR', items=[
        ('SAM', 'SAM 原版', '使用原版手指，便于对比'),
        ('WILOR', 'WiLoR 手掌与手指', '识别掌心朝向与手指，保留手腕位置和身体；需要额外处理时间')])
    finger_temporal: BoolProperty(name='过滤手指短时误判', default=True,
        description='用前后可信帧、独立手部关键点和视频像素校验拇指及手指短时跳变；关闭后重新应用可对比原始 WiLoR')
    finger_reliability: BoolProperty(name='实验：修复手指识别分歧', default=False,
        description='生成动作时修复有前后姿势、SAM 和视频图像共同佐证的手指误判；不套用预设手势。关闭后重新应用可恢复')
    face_mode: EnumProperty(name='面部捕捉', default='FULL', items=[
        ('FULL', '口型 + 表情', '嘴部、眼睑、眉毛；视线仍由看向镜头控制'),
        ('LIPS', '仅口型', '保留已有眉毛和眼睑动画'),
        ('OFF', '关闭', '不识别面部，不改已有面部动画')])
    face_fps: IntProperty(name='面部每秒采样', min=10, max=60, default=30)
    face_mouth_gain: FloatProperty(name='口型幅度', min=0, max=2, default=1)
    face_expression_gain: FloatProperty(name='表情幅度', min=0, max=2, default=1)
    face_smoothing: FloatProperty(name='面部平滑（秒）', min=0, max=.3, default=.045)
    face_blink: BoolProperty(name='捕捉真实眨眼', default=True)
    face_take: PointerProperty(type=bpy.types.Action)
    face_status: StringProperty(default='待识别面部')
    previous_face_state: StringProperty()
    camera_mode: EnumProperty(name='拍摄方式', default='FIXED', items=[
        ('FIXED', '固定机位 · 人物移动', '摄像机一次标定；角色负责远近与侧移，画外下半身保守处理'),
        ('HANDHELD', '手持机位 · 尚未支持', '需要背景运动与人物运动分离，目前拒绝生成，避免把追脸误作手持重建'),
        ('LEGACY', '旧版追脸 · 兼容', '仅用于对照旧动作，不代表真实摄影机运动')])
    shot_preset: EnumProperty(name='取景构图', default='SOURCE', items=[
        ('SOURCE', '原视频脸部基准', '按原视频脸部位置与大小校准；不同头身比可能露出更多身体'),
        ('CHEST', '胸口近景 · 头顶优先', '收紧初始景别并允许脸部位置偏下；靠近镜头时仍允许自然出画')])
    shot_scale: FloatProperty(name='取景放大', default=1.25, min=.75, max=2.,
        description='相对脸部基准的固定取景倍率；不同视频可调整，不改变模型骨长')
    shot_vertical: FloatProperty(name='人物下移', default=.15, min=-.3, max=.3,
        description='按画面高度调整初始构图；正数将人物下移，为头顶留出空间')
    semantic_contacts: BoolProperty(name='按身体部位保留手势', default=True,
        description='连续近身手势按角色胸部、头部和嘴部比例重定向；二维重叠不直接认定三维接触')
    screen_enabled: BoolProperty(name='整体画面对齐', default=True)
    screen_body: BoolProperty(name='头、躯干与手共同对齐', default=True,
        description='保留自然骨长，先确定构图与人物位移，再统一求解手臂、手掌和近身手势')
    screen_body_limit: FloatProperty(name='对齐额外倾斜上限（度）', min=1, max=25, default=20)
    screen_camera_calibrate: BoolProperty(name='先校准摄像机基础构图', default=True,
        description='保留原摄像机；固定模式只校准一次，后续远近交给角色移动')
    screen_landmarks: BoolProperty(name='指尖 / 掌心联合对齐', default=True)
    screen_chin_contact: BoolProperty(name='保留手与下巴的位置关系',default=True,
        description='仅在连续图像关键点支持托下巴动作时，按角色下巴位置保持手指间隙；不套用手势')
    screen_space: EnumProperty(name='对齐基准', default='FRAME', items=[
        ('FRAME', '原视频画面位置', '匹配可见关键点的位置和手部画面大小；保留骨长，在手臂可达范围内调整深度'),
        ('FACE', '相对脸部的位置', '按人物脸部比例匹配手的位置，保留人物手部尺寸与动捕深度')])
    torso_enabled: BoolProperty(name='真实上半身运动', default=True)
    torso_strength: FloatProperty(name='躯干动作保留', min=0, max=1, default=.75, subtype='FACTOR')
    torso_screen_strength: FloatProperty(name='靠近 / 侧移跟随', min=0, max=1, default=0, subtype='FACTOR')
    torso_limit: FloatProperty(name='额外倾斜上限（度）', min=1, max=25, default=15)
    torso_smoothing: FloatProperty(name='躯干平滑（秒）', min=.01, max=.5, default=.12)
    screen_side: EnumProperty(name='对齐的手', default='L', items=[('L', '艾莲左手', ''), ('R', '艾莲右手', '')])
    screen_camera: PointerProperty(name='画面摄像机', type=bpy.types.Object,
                                  poll=lambda self, obj: obj.type == 'CAMERA')
    screen_strength: FloatProperty(name='画面对齐强度', min=0, max=1, default=1, subtype='FACTOR')
    screen_depth: FloatProperty(name='画面大小匹配余量', min=0, max=1, default=.7, subtype='FACTOR',
                               description='原视频模式中允许手臂前后移动以匹配画面大小；相对脸部模式仍只微调深度')
    screen_front: FloatProperty(name='额外前伸（厘米）', min=-20, max=20, default=0)
    screen_mirror: BoolProperty(name='反转视频左右', default=False)
    screen_start: FloatProperty(name='对齐开始（秒）', min=0, default=0)
    screen_end: FloatProperty(name='对齐结束（0 为片尾）', min=0, default=0)
    head_depth_enabled: BoolProperty(name='挠头深度稳定', default=False)
    head_depth_start: FloatProperty(name='挠头开始（秒）', min=0, default=0)
    head_depth_end: FloatProperty(name='挠头结束（秒）', min=.01, default=1)
    head_depth_offset: FloatProperty(name='掌心距眼睛平面（厘米）', min=-20, max=20, default=0)
    gesture_mode: EnumProperty(name='手势辅助', default='OFF', items=[
        ('OFF', '关闭', ''), ('MIDDLE_ONLY', '仅伸中指', '在指定时间段收拢其他手指')])
    gesture_start: FloatProperty(name='手势开始（秒）', min=0, default=0)
    gesture_end: FloatProperty(name='手势结束（秒）', min=.01, default=1)
    gesture_strength: FloatProperty(name='手势修正强度', min=0, max=1, default=1, subtype='FACTOR')
    gaze_enabled: BoolProperty(name='眼睛看向镜头', default=True,
                               description='只调整眼睛注视；头部默认保留视频动捕转动')
    gaze_away_ranges: CollectionProperty(type=SAMGazeRange)
    gaze_away_index: IntProperty(default=0)
    gaze_away_fade: IntProperty(name='范围内过渡（帧）', default=6, min=0, max=120,
                               description='在范围内平滑解除、恢复注视；0 为直接切换，短范围自动缩短过渡')
    gaze_head_tracking: BoolProperty(name='头部也转向镜头', default=False,
                                    description='可选表演效果：会改变视频中的头颈朝向；还原视频时请关闭')
    head_pose_enabled: BoolProperty(name='从人脸捕捉头部朝向', default=True,
                                    description='使用视频人脸姿态还原转头、点头和侧倾；短暂丢脸时回退身体动捕')
    head_smoothing: FloatProperty(name='身体模型头颈去抖（秒）', min=0, max=.5, default=.12,
                                  precision=3,
                                  description='按时间平滑头颈动捕；0 关闭，较大值会削弱快速点头；重新应用后生效')
    gaze_camera: PointerProperty(name='目标摄像机', type=bpy.types.Object,
                                 poll=lambda self, obj: obj.type == 'CAMERA',
                                 description='留空使用当前场景摄像机；移动或更换后重新应用')
    gaze_motion: FloatProperty(name='保留动捕晃动', min=0, max=1, default=.80, subtype='FACTOR',
                               description='仅在头部也转向镜头时使用：混入动捕转动，仍会改变原始头部朝向')
    gaze_eyes: FloatProperty(name='眼睛跟随', min=0, max=1, default=1, subtype='FACTOR')
    gaze_noise_enabled: BoolProperty(name='自然注视晃动', default=True,
                                      description='加入平滑随机的轻微左右晃动；同一种子可重复播放和渲染')
    gaze_noise_amplitude: FloatProperty(name='晃动幅度（度）', min=0, max=12, default=1.2)
    gaze_noise_speed: FloatProperty(name='随机晃动速度', min=.05, max=2, default=.12,
                                    description='控制随机变化快慢，没有固定往返周期；不控制动捕抖动')
    gaze_noise_seed: IntProperty(name='随机种子', min=0, max=1000000, default=17,
                                 description='更换数值可得到另一组平滑随机晃动，重新应用后生效')
    gaze_noise_randomness: FloatProperty(name='随机强度', min=0, max=1, default=.75,
                                        subtype='FACTOR',
                                        description='0 保留规律慢摆，1 完全使用平滑随机慢摆；幅度与速度分别由原参数控制')
    gaze_noise_head: FloatProperty(name='头部晃动比例', min=0, max=1, default=.35, subtype='FACTOR')
    gaze_noise_eyes: FloatProperty(name='眼睛晃动比例', min=0, max=1, default=.75, subtype='FACTOR')
    body_enabled: BoolProperty(name='身体辅助晃动', default=False,
                               description='腰部以上加入轻微慢摆；动捕动作明显时自动减弱，不移动根骨或腿部')
    body_strength: FloatProperty(name='辅助强度', min=0, max=1, default=.5, subtype='FACTOR',
                                 description='只调节额外晃动，不削弱原有动捕；0 等同关闭')
    body_amplitude: FloatProperty(name='身体幅度（度）', min=0, max=6, default=3)
    body_speed: FloatProperty(name='身体晃动速度', min=.05, max=.5, default=.12)
    body_randomness: FloatProperty(name='身体随机强度', min=0, max=1, default=.85, subtype='FACTOR')
    body_seed: IntProperty(name='身体随机种子', min=0, max=1000000, default=71)
    mouth_enabled: BoolProperty(name='捂嘴修正', default=False)
    punch_enabled: BoolProperty(name='捶向镜头修正', default=False)
    punch_automatic: BoolProperty(name='自动识别出拳与画面目标', default=True,
                                  description='从闭合拳头的持续变大识别前伸，按视频拳面位置与大小拟合；证据不足时保留原动作')
    punch_side: EnumProperty(name='出拳的手', default='L', items=[('L', '艾莲左手', ''), ('R', '艾莲右手', '')])
    punch_camera: PointerProperty(name='出拳目标摄像机', type=bpy.types.Object,
                                  poll=lambda self, obj: obj.type == 'CAMERA')
    punch_start: FloatProperty(name='出拳开始（秒）', min=0, default=2.10)
    punch_peak: FloatProperty(name='击中（秒）', min=0, default=2.27)
    punch_release: FloatProperty(name='停留结束（秒）', min=0, default=2.40)
    punch_end: FloatProperty(name='收回结束（秒）', min=0, default=2.80)
    punch_reference: FloatProperty(name='握拳参考（秒）', min=0, default=1.85)
    punch_distance: FloatProperty(name='拳面距镜头（厘米）', min=5, max=200, default=18,
                                  description='受真实臂长限制，无法到达时保持骨长并停在最远处')
    punch_x: FloatProperty(name='画面落点 X', min=0, max=1, default=.5)
    punch_y: FloatProperty(name='画面落点 Y', min=0, max=1, default=.48,
                           description='从画面顶端向下计算')
    punch_strength: FloatProperty(name='出拳修正强度', min=0, max=1, default=1, subtype='FACTOR')
    punch_stabilize: BoolProperty(name='平滑遮挡期间的头部（旧版）', default=False)
    mouth_side: EnumProperty(name='捂嘴的手', default='L', items=[
        ('L', '艾莲左手', ''), ('R', '艾莲右手', '')])
    mouth_start: FloatProperty(name='开始（秒）', min=0, default=0)
    mouth_end: FloatProperty(name='结束（秒）', min=0, default=3)
    mouth_fade: FloatProperty(name='过渡（秒）', min=0, max=5, default=.25)
    mouth_strength: FloatProperty(name='修正强度', min=0, max=1, default=1, subtype='FACTOR')
    mouth_x: FloatProperty(name='左右（厘米）', min=-10, max=10, default=0)
    mouth_front: FloatProperty(name='向外（厘米）', min=-1, max=10, default=0)
    mouth_z: FloatProperty(name='上下（厘米）', min=-10, max=10, default=0)
    progress: FloatProperty(min=0, max=1, default=0, options={'SKIP_SAVE'})
    status: StringProperty(default='待导入视频')
    job_id: StringProperty()
    take_path: StringProperty(subtype='FILE_PATH')
    take: PointerProperty(type=bpy.types.Action)
    previous_rig: PointerProperty(type=bpy.types.Object)
    previous_state: StringProperty()


class SAMImportVideo(bpy.types.Operator, ImportHelper):
    bl_idname = 'sam_ellen.import_video'
    bl_label = '导入视频并生成动作'
    filter_glob: StringProperty(default='*.mp4;*.mov;*.mkv;*.avi;*.webm;*.m4v', options={'HIDDEN'})

    def execute(self, context):
        context.scene.sam_ellen.video = self.filepath
        return bpy.ops.sam_ellen.process()


class SAMProcess(bpy.types.Operator):
    bl_idname = 'sam_ellen.process'
    bl_label = '生成 / 重新识别'
    face_only: BoolProperty(default=False, options={'SKIP_SAVE', 'HIDDEN'})
    hands_only: BoolProperty(default=False, options={'SKIP_SAVE', 'HIDDEN'})

    def execute(self, context):
        global _work
        if _work:
            self.report({'ERROR'}, 'A SAM job is already running')
            return {'CANCELLED'}
        scene = context.scene
        settings = scene.sam_ellen
        try:
            ensure_pipeline(scene)
            if settings.rig is None:
                raise ValueError('Choose the character armature')
            bone_map(settings.rig)
            if not Path(bpy.path.abspath(settings.video)).is_file():
                raise ValueError('Select a video first')
            import xr_ellen_bridge as xr
            if xr._runtime and xr._runtime.recording:
                raise RuntimeError('Finish XR recording first')
            if self.face_only and (not settings.take_path or settings.face_mode == 'OFF'):
                raise ValueError('请先启用面部捕捉，并生成身体动作')
            if self.hands_only and (not settings.take_path or settings.hand_model != 'WILOR'):
                raise ValueError('请先生成身体动作，并选择 WiLoR 手指增强')
            _work = {'scene': scene, 'started': time.monotonic(), 'face_only': self.face_only,
                     'hands_only': self.hands_only}
            try:
                health = request(scene, '/sam3d-health', timeout=.3)
                if health.get('sequential_video_decode') != 1:
                    raise OSError('Local service requires the video decoder update')
            except (OSError, urllib.error.URLError):
                launcher = SERVICE_ROOT / 'local_app/launch.ps1'
                if not launcher.is_file():
                    raise RuntimeError('SAM installation not found')
                _work['launcher'] = subprocess.Popen(
                    ['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(launcher), '-NoBrowser'],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW)
            settings.status = '正在连接本地 SAM 服务'
            settings.progress = 0
            run_timer()
            return {'FINISHED'}
        except Exception as error:
            stop_work()
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}


class SAMCancel(bpy.types.Operator):
    bl_idname = 'sam_ellen.cancel'
    bl_label = '取消'

    def execute(self, context):
        stop_work()
        context.scene.sam_ellen.status = '已取消'
        return {'FINISHED'}


class SAMReapply(bpy.types.Operator):
    bl_idname = 'sam_ellen.reapply'
    bl_label = '重新适配（不重跑识别）'

    def execute(self, context):
        try:
            if _work:
                raise RuntimeError('Wait for the current job first')
            begin_bake(context.scene, context.scene.sam_ellen.take_path, reapply=True)
            run_timer()
            return {'FINISHED'}
        except Exception as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}


class SAMCameraMatchPreset(bpy.types.Operator):
    bl_idname = 'sam_ellen.camera_match_preset'
    bl_label = '按统一流程重新适配'
    bl_description = '同步公共插件配置，用当前视频缓存重新适配画面对齐；不重跑身体和手指识别'

    def execute(self, context):
        s = context.scene.sam_ellen
        s.use_shared_pipeline = True
        try:
            result = ensure_pipeline(context.scene)
            ensure_collar(context.scene)
            if result['status'] != 'ready':
                self.report({'WARNING'}, s.pipeline_status)
                return {'CANCELLED'}
        except Exception as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        if s.take_path:
            return bpy.ops.sam_ellen.reapply()
        return {'FINISHED'}


class SAMRepairCollar(bpy.types.Operator):
    bl_idname = 'sam_ellen.repair_collar'
    bl_label = '恢复两侧衣领翘起'
    bl_description = '保留艾莲白色肩领向上展开的原造型，撤掉额外上臂跟随，不修改动作或表情'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            result = ensure_collar(context.scene)
            if result['status'] not in ('repaired', 'ready'):
                self.report({'WARNING'}, context.scene.sam_ellen.collar_status or '当前骨架不是已验证的艾莲衣领绑定')
                return {'CANCELLED'}
            self.report({'INFO'}, '两侧衣领已恢复翘起，无需重新识别动捕')
            return {'FINISHED'}
        except Exception as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}


class SAMRestoreCamera(bpy.types.Operator):
    bl_idname = 'sam_ellen.restore_camera'
    bl_label = '恢复原机位'

    def execute(self, context):
        if _work:
            self.report({'ERROR'}, '请先等待生成结束')
            return {'CANCELLED'}
        s = context.scene.sam_ellen
        current = s.screen_camera or context.scene.camera
        original = bpy.data.objects.get(current.get('sam_original_camera', '')) if current else None
        if original is None:
            self.report({'WARNING'}, '当前摄像机没有保存的原机位')
            return {'CANCELLED'}
        context.scene.camera = s.screen_camera = original
        sync_camera_views(context.scene)
        if s.gaze_camera == current:
            s.gaze_camera = original
        s.screen_camera_calibrate = False
        s.status = '已恢复原机位；重新应用画面对齐可按此机位适配动作'
        return {'FINISHED'}


class SAMRestore(bpy.types.Operator):
    bl_idname = 'sam_ellen.restore'
    bl_label = '恢复原动作'

    def execute(self, context):
        stop_work()
        s = context.scene.sam_ellen
        if s.previous_state and s.previous_rig:
            restore(s.previous_rig, context.scene, json.loads(s.previous_state))
            from sam_ellen_video_audio import activate
            activate(context.scene, s.previous_rig, s.previous_rig.animation_data.action)
            if s.previous_face_state:
                restore_face(json.loads(s.previous_face_state), context.scene)
                s.previous_face_state = ''
                context.scene.frame_set(context.scene.frame_current)
                s.face_status = '已恢复原面部动画'
            s.previous_state = ''
            s.previous_rig = None
            s.status = '已恢复原动作；SAM 动画仍保留在文件中'
        return {'FINISHED'}


class SAMPanel(bpy.types.Panel):
    bl_idname = 'SAM_PT_ellen_preview'
    bl_label = 'SAM 3D Body 视频动捕'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SAM 动捕'

    def draw(self, context):
        layout = self.layout
        s = context.scene.sam_ellen
        column = layout.column()
        column.enabled = not bool(_work)
        from sam_ellen_pipeline import REVISION, action_status
        box = column.box()
        box.label(text=f'公共插件 {".".join(map(str, bl_info["version"]))} / 流程 {REVISION}')
        box.prop(s, 'use_shared_pipeline')
        if s.pipeline_status:
            box.label(text=s.pipeline_status)
        box.label(text=action_status(context.scene))
        box.operator('sam_ellen.camera_match_preset', icon='FILE_REFRESH')
        column.prop(s, 'rig')
        from fix_ellen_collar import supported
        if supported(s.rig):
            box = column.box()
            box.operator('sam_ellen.repair_collar', icon='MOD_ARMATURE')
            if s.collar_status:
                box.label(text=s.collar_status)
        column.prop(s, 'capture_mode')
        from sam_ellen_phone_ranges import draw as draw_phone
        draw_phone(column, s)
        if hand_side(s) != 'BOTH':
            column.prop(s, 'inactive_pose')
        column.label(text='上半身 / 固定腿部与站位', icon='ARMATURE_DATA')
        box = column.box()
        profile_prop(box, s, 'torso_enabled')
        if s.torso_enabled:
            for name in ('torso_strength', 'torso_screen_strength', 'torso_limit', 'torso_smoothing'):
                profile_prop(box, s, name)
        if s.take_path:
            box.operator('sam_ellen.reapply', text='应用真实上半身运动', icon='CHECKMARK')
        profile_prop(column, s, 'sample_fps')
        row = column.row(align=True)
        row.prop(s, 'start_seconds')
        row.prop(s, 'end_seconds')
        profile_prop(column, s, 'smoothing')
        box = column.box()
        profile_prop(box, s, 'hand_model')
        if s.hand_model == 'WILOR':
            profile_prop(box, s, 'finger_temporal')
            box.prop(s, 'finger_reliability')
        if s.hand_model == 'WILOR' and s.take_path:
            box.operator('sam_ellen.process', text='补充 / 重新识别手掌与手指', icon='FILE_REFRESH').hands_only = True
        if s.take_path:
            box.operator('sam_ellen.reapply', text='应用手指模型 / 对比', icon='CHECKMARK')
        box = column.box()
        box.prop(s, 'gesture_mode')
        if s.gesture_mode != 'OFF':
            if hand_side(s) == 'BOTH':
                box.prop(s, 'screen_side', text='修正的手')
            box.prop(s, 'gesture_start')
            box.prop(s, 'gesture_end')
            box.prop(s, 'gesture_strength')
            if s.take_path:
                box.operator('sam_ellen.reapply', text='应用手势辅助', icon='CHECKMARK')
        box = column.box()
        profile_prop(box, s, 'screen_enabled')
        if s.screen_enabled:
            if s.capture_mode in ('UPPER_BOTH', 'PHONE_LEFT', 'PHONE_RIGHT'):
                box.label(text='双手共同对齐 / 共用镜头与躯干', icon='CON_TRACKTO')
            else:
                box.prop(s, 'screen_side')
            box.prop(s, 'screen_camera')
            profile_prop(box, s, 'screen_landmarks')
            if s.screen_landmarks:
                if s.screen_space=='FRAME':
                    profile_prop(box, s, 'screen_chin_contact')
                profile_prop(box, s, 'screen_space')
                if s.screen_space == 'FRAME':
                    box.prop(s, 'camera_mode')
                    if s.camera_mode=='FIXED':
                        box.prop(s, 'shot_preset')
                        if s.shot_preset=='CHEST':
                            box.prop(s, 'shot_scale')
                            box.prop(s, 'shot_vertical')
                    box.prop(s, 'semantic_contacts')
                    profile_prop(box, s, 'screen_body')
                    if s.screen_body:
                        profile_prop(box, s, 'screen_camera_calibrate')
                        if s.screen_camera and s.screen_camera.get('sam_original_camera'):
                            box.operator('sam_ellen.restore_camera', icon='LOOP_BACK')
                        if s.camera_mode!='FIXED':profile_prop(box, s, 'screen_body_limit')
            profile_prop(box, s, 'screen_strength')
            profile_prop(box, s, 'screen_depth')
            profile_prop(box, s, 'screen_front')
            box.prop(s, 'screen_mirror')
            profile_prop(box, s, 'screen_start')
            profile_prop(box, s, 'screen_end')
            box.prop(s, 'head_depth_enabled')
            if s.head_depth_enabled:
                box.prop(s, 'head_depth_start')
                box.prop(s, 'head_depth_end')
                box.prop(s, 'head_depth_offset')
            if s.take_path:
                box.operator('sam_ellen.reapply', text='校准镜头并应用对齐' if s.screen_body and s.screen_camera_calibrate and s.screen_space == 'FRAME' else '应用画面对齐', icon='CHECKMARK')
        box = column.box()
        box.prop(s, 'face_mode')
        if s.face_mode != 'OFF':
            box.prop(s, 'face_fps')
            box.prop(s, 'face_mouth_gain')
            if s.face_mode == 'FULL':
                box.prop(s, 'face_expression_gain')
                box.prop(s, 'face_blink')
            box.prop(s, 'face_smoothing')
            if s.take_path:
                box.operator('sam_ellen.process', text='补充 / 重新识别面部', icon='FILE_REFRESH').face_only = True
                box.operator('sam_ellen.reapply', text='应用面部设置', icon='CHECKMARK')
            box.label(text=s.face_status)
        box = column.box()
        profile_prop(box, s, 'head_pose_enabled')
        box.prop(s, 'head_smoothing')
        box.prop(s, 'gaze_enabled')
        if s.gaze_enabled:
            box.prop(s, 'gaze_camera')
            profile_prop(box, s, 'gaze_head_tracking')
            if s.gaze_head_tracking:
                box.prop(s, 'gaze_motion')
            else:
                box.label(text='保留视频头部转动', icon='CHECKMARK')
            box.prop(s, 'gaze_eyes')
            draw_ranges(box, s)
            box.prop(s, 'gaze_noise_enabled')
            if s.gaze_noise_enabled:
                box.prop(s, 'gaze_noise_amplitude')
                box.prop(s, 'gaze_noise_speed')
                box.prop(s, 'gaze_noise_randomness')
                if s.gaze_head_tracking:
                    box.prop(s, 'gaze_noise_head')
                box.prop(s, 'gaze_noise_eyes')
                box.prop(s, 'gaze_noise_seed')
        if s.take_path:
            if not s.gaze_enabled:
                box.operator('sam_ellen.apply_eye_contact', text='恢复原眼睛视线', icon='LOOP_BACK')
            from sam_ellen_gaze import eye_settings_pending
            eye_pending = eye_settings_pending(context.scene)
            box.label(text='眼睛视线待应用' if eye_pending else '眼睛视线已应用',
                      icon='TIME' if eye_pending else 'CHECKMARK')
            box.operator('sam_ellen.reapply', text='重新适配头部（整段动作）', icon='FILE_REFRESH')
            pending = head_settings_pending(s)
            box.label(text='头部设置待应用' if pending else '头部设置已应用',
                      icon='TIME' if pending else 'CHECKMARK')
        box = column.box()
        profile_prop(box, s, 'body_enabled')
        if s.body_enabled:
            for prop in ('body_strength', 'body_amplitude', 'body_speed', 'body_randomness', 'body_seed'):
                box.prop(s, prop)
        if s.take_path:
            box.operator('sam_ellen.reapply', text='应用身体辅助', icon='CHECKMARK')
            pending = body_settings_pending(s)
            box.label(text='身体设置待应用' if pending else '身体设置已应用',
                      icon='TIME' if pending else 'CHECKMARK')
        box = column.box()
        box.prop(s, 'mouth_enabled')
        if s.mouth_enabled:
            box.prop(s, 'mouth_side')
            box.prop(s, 'mouth_start')
            box.prop(s, 'mouth_end')
            box.prop(s, 'mouth_fade')
            box.prop(s, 'mouth_strength')
            for prop in ('mouth_x', 'mouth_front', 'mouth_z'):
                box.prop(s, prop)
            if s.take_path:
                box.operator('sam_ellen.reapply', text='应用捂嘴设置', icon='CHECKMARK')
        box = column.box()
        box.prop(s, 'punch_enabled')
        if s.punch_enabled:
            box.prop(s, 'punch_automatic')
            for prop in ('punch_side', 'punch_camera', 'punch_strength'):
                box.prop(s, prop)
            if not s.punch_automatic:
                for prop in ('punch_start', 'punch_peak', 'punch_release', 'punch_end',
                             'punch_reference', 'punch_distance', 'punch_x', 'punch_y'):
                    box.prop(s, prop)
            elif hand_side(s) != 'BOTH':
                box.label(text='出拳手自动跟随自拍左右手预设')
            if s.punch_automatic and s.take:
                try:
                    settings = json.loads(s.take.get('sam_punch', 'null'))
                    if settings and settings.get('automatic'):
                        box.label(text=f'识别到 {len(settings.get("segments", []))} 段出拳')
                        for segment in settings.get('segments', []):
                            box.label(text=f'{segment["start"]:.2f}–{segment["end"]:.2f} 秒')
                            if segment.get('incomplete'):
                                box.label(text='近镜头证据缺失：仅修正可见部分', icon='INFO')
                        if not settings.get('segments'):
                            box.label(text='未识别到出拳，当前保留原动作', icon='INFO')
                    elif not settings:
                        box.label(text='当前动作尚未应用出拳修正', icon='INFO')
                except (ValueError, TypeError):
                    pass
            if s.take and s.take.get('sam_punch_limited_samples', 0):
                box.label(text='部分落点超出臂长，已限制前伸', icon='INFO')
        if s.take_path:
            box.operator('sam_ellen.reapply', text='应用出拳修正', icon='CHECKMARK')
        column.prop(s, 'import_mode')
        from sam_ellen_editing import draw_video_audio_options
        draw_video_audio_options(column, context.scene)
        if s.import_mode == 'APPEND':
            column.prop(s, 'transition_frames')
        column.operator('sam_ellen.import_video', icon='FILE_MOVIE')
        column.prop(s, 'video', text='')
        if s.video:
            column.operator('sam_ellen.process', icon='FILE_REFRESH')
        if _work:
            layout.progress(factor=s.progress, type='BAR', text=f'{s.progress:.0%}')
            layout.operator('sam_ellen.cancel', icon='CANCEL')
        for line in textwrap.wrap(s.status, width=max(16, int(context.region.width/13))):
            layout.label(text=line)
        row = layout.row(align=True)
        row.enabled = not bool(_work)
        row.operator('screen.frame_jump', text='', icon='REW').end = False
        row.operator('screen.animation_play', text='播放 / 暂停', icon='PLAY')
        row.operator('xr_ellen.camera', text='', icon='VIEW_CAMERA')
        row = layout.row(align=True)
        row.operator('xr_ellen.viewport', text='快速预览', icon='SHADING_SOLID').mode = 'SOLID'
        row.operator('xr_ellen.viewport', text='材质预览', icon='SHADING_TEXTURE').mode = 'MATERIAL'
        if hasattr(context.scene, 'ellen_physics') and __import__('ellen_physics').manages(context.scene):
            import ellen_physics
            physics = layout.column()
            physics.enabled = not bool(_work)
            ellen_physics.draw(physics, context.scene)
        elif hasattr(context.scene, 'wiggle_enable'):
            layout.prop(context.scene, 'wiggle_enable', text='头发 / 衣带 / 尾巴物理')
        column = layout.column()
        column.enabled = not bool(_work)
        if s.take_path:
            column.operator('sam_ellen.reapply', icon='ACTION')
        if s.previous_state:
            column.operator('sam_ellen.restore', icon='LOOP_BACK')
        column.operator('wm.save_as_mainfile', text='保存 Blender 文件', icon='FILE_TICK')


@persistent
def before_load(_):
    stop_work()


@persistent
def after_load(_):
    for scene in bpy.data.scenes:
        from sam_ellen_phone_ranges import migrate
        migrate(scene)
        try:
            ensure_pipeline(scene)
        except Exception as error:
            scene.sam_ellen.pipeline_status = '统一配置未加载：'+str(error)
            print('SAM_PIPELINE_SYNC_FAILED', error)
        if scene.sam_ellen.rig:
            from sam_ellen_phone import sync
            try:
                sync(scene)
            except (ValueError, RuntimeError) as error:
                scene.sam_ellen.phone_status = str(error)
            sync_camera_views(scene)
            try:
                ensure_collar(scene)
            except Exception as error:
                scene.sam_ellen.collar_status = '衣领未自动修复：'+str(error)
                print('ELLEN_COLLAR_REPAIR_FAILED', error)


@persistent
def before_save(_):
    if _work and (_work.get('bake') or _work.get('eyes') or _work.get('alignment_upgrade') or _work.get('entry_upgrade')):
        scene = _work['scene']
        stop_work()
        scene.sam_ellen.status = '生成已取消，原动作已恢复'
    after_load(None)


CLASSES = (*GAZE_CLASSES, *PHONE_CLASSES, *PHONE_RANGE_CLASSES, SAMSettings, SAMImportVideo, SAMProcess, SAMCancel, SAMReapply, SAMCameraMatchPreset, SAMRepairCollar, SAMRestoreCamera, SAMRestore, SAMPanel)


def initial_pipeline_sync():
    # Blender restricts bpy.data during addon registration. Defer the initial
    # sync; subsequent file loads use the persistent handler directly.
    after_load(None)
    return None


def register():
    import ellen_physics
    ellen_physics.register('sam')
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.sam_ellen = PointerProperty(type=SAMSettings)
    bpy.app.handlers.load_pre.append(before_load)
    bpy.app.handlers.load_post.append(after_load)
    bpy.app.handlers.save_pre.append(before_save)
    import sam_ellen_editing
    sam_ellen_editing.register()
    bpy.app.timers.register(initial_pipeline_sync, first_interval=.1)


def unregister():
    global _poll_pool
    stop_work()
    if _poll_pool is not None:
        _poll_pool.shutdown(wait=False, cancel_futures=True)
        _poll_pool = None
    if bpy.app.timers.is_registered(initial_pipeline_sync):
        bpy.app.timers.unregister(initial_pipeline_sync)
    import sam_ellen_editing
    sam_ellen_editing.unregister()
    for handlers, function in ((bpy.app.handlers.load_pre, before_load),
                               (bpy.app.handlers.load_post, after_load),
                               (bpy.app.handlers.save_pre, before_save)):
        if function in handlers:
            handlers.remove(function)
    del bpy.types.Scene.sam_ellen
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    import ellen_physics
    ellen_physics.unregister('sam')

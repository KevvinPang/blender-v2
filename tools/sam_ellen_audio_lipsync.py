"""Bake a local voice file to editable mouth keys; keep source actions for restore."""
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path

import bpy
import numpy as np
from bpy.props import StringProperty
from bpy_extras.io_utils import ImportHelper

from sam_ellen_livliv import analyze, mouth_weights, VOWELS

CHECKPOINT = 'livliv_lipsync_checkpoint'
MOUTHS = (*VOWELS, '口横広げ', '口横狭め')


def decode(filepath):
    from sam_ellen_video_audio import executable
    path = Path(bpy.path.abspath(str(filepath))).resolve()
    if not path.is_file():
        raise ValueError('配音文件不存在')
    program = executable('ffmpeg')
    if not program:
        raise ValueError('未找到 FFmpeg')
    result = subprocess.run([program, '-v', 'error', '-i', str(path), '-vn',
        '-map', '0:a:0', '-ac', '1', '-ar', '48000', '-f', 'f32le', 'pipe:1'],
        capture_output=True, timeout=180,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode:
        raise ValueError('读取配音失败：'+result.stderr.decode('utf8', errors='replace')[-600:])
    return np.frombuffer(result.stdout, dtype='<f4').copy(), path


def bake(scene, filepath, start, gain=1, lead_ms=30, attach_audio=True,
         *, _capture=False, timeline=None, replace_end=None, audio_owner=None):
    import ellen_expression_timeline as expr
    from sam_ellen_editing import available, import_audio
    from sam_ellen_face import face_state, restore_face
    from sam_ellen_sequence import curves, validate_animation
    if not _capture:
        available()
    if replace_end is not None:
        if timeline or int(replace_end)<int(start):
            raise ValueError('覆盖口型的帧范围无效')
        replace_end=int(replace_end)
    mesh = expr.face_object(scene)
    keys = expr.edit_available(scene)
    if not keys.use_relative or not all(n in keys.key_blocks for n in VOWELS):
        raise ValueError('此角色需要先适配相对形态键「あ、い、う、え、お、ワ」')
    expr.intervals(scene)  # Reject invalid expression ranges before decoding or editing.
    ad = keys.animation_data_create()
    validate_animation(ad)
    paths = {keys.key_blocks[n].path_from_id('value') for n in MOUTHS if n in keys.key_blocks}
    if any(c.data_path in paths for c in ad.drivers):
        raise ValueError('嘴型存在驱动器，请先烘焙后生成')
    if not math.isfinite(gain) or not 0 <= gain <= 2:
        raise ValueError('口型幅度必须在 0 到 2 之间')
    samples, path = decode(filepath)
    track = analyze(samples, lead_ms=lead_ms)
    rate = scene.render.fps/scene.render.fps_base
    before = face_state(mesh, scene)
    checkpoint = mesh.get(CHECKPOINT)
    actions_before = {a.as_pointer() for a in bpy.data.actions}
    old = ad.action
    source, slot = old, ad.action_slot
    settings = mesh.ellen_expression_track
    if old and old == settings.generated_action and settings.base_action:
        source = settings.base_action
        slot = next((s for s in source.slots if s.handle == settings.base_slot), None)
    if source and any(c.modifiers or c.mute for c in curves(source, slot)):
        raise ValueError('原始面部动作有修改器或静音曲线，请先烘焙')
    sound = None
    muted_sounds = {}
    audio_actions = {}
    old_end = scene.frame_end
    audio_state = scene.sync_mode, scene.use_audio
    try:
        action = source.copy() if source else bpy.data.actions.new('Audio Lip Sync')
        action.name = 'Audio Lip Sync '+datetime.now().strftime('%Y%m%d_%H%M%S')
        action.use_fake_user = True
        for original in (source, old):
            if original:
                original.use_fake_user = True
        ad.action = action
        if slot:
            ad.action_slot = next(s for s in action.slots if s.identifier == slot.identifier)
        count = math.ceil(track['duration']*rate)
        origin = 1 if timeline else start
        frames = (np.arange(start,replace_end+1,dtype=float) if replace_end is not None
                  else np.arange(count+1, dtype=float)+origin)
        times = np.minimum((frames-origin)/rate, track['duration'])
        openings = np.interp(times, track['times'], track['open'])
        cues = np.interp(times, track['times'], track['vowel'])
        values = [mouth_weights(o, v, gain) for o, v in zip(openings, cues)]
        if replace_end is not None:
            values=[value if (frame-start)/rate<track['duration'] else {}
                    for frame,value in zip(frames,values)]
        # Manual generation replaces the whole mouth track. Capture append
        # merges this local track into the incoming clip after creating it.
        for name in MOUTHS:
            if name not in keys.key_blocks:
                continue
            curve = action.fcurve_ensure_for_datablock(keys, keys.key_blocks[name].path_from_id('value'))
            curve.keyframe_points.clear()
            boundary = [] if timeline or replace_end is not None else [start-1, 0]
            curve.keyframe_points.add(len(frames)+len(boundary)//2)
            curve.keyframe_points.foreach_set('co', boundary+[
                component for f, value in zip(frames, values)
                for component in (f, value.get(name, 0))])
            for point in curve.keyframe_points:
                point.interpolation = 'LINEAR'
            curve.extrapolation = 'CONSTANT'
            curve.update()
        action['livliv_source'] = str(path)
        action['livliv_revision'] = '83ecd9c48f33d9439411e596841193c24dc7b005'
        action['livliv_settings'] = json.dumps({'gain': gain, 'lead_ms': lead_ms,
            'start': start, 'duration': track['duration'], 'fps': rate})
        if timeline:
            from sam_ellen_sequence import merge_action
            temporary = action
            action = merge_action(keys, source, slot, temporary, rate, timeline, paths)
            bpy.data.actions.remove(temporary)
        if attach_audio:
            sound = import_audio(scene, str(path), start)
            if replace_end is not None:
                sound.frame_final_end=min(sound.frame_final_end,replace_end+1)
        if not _capture and replace_end is None:
            scene.frame_end = max(scene.frame_end, int(frames[-1]))
        if (any(i.enabled for i in settings.items) or settings.global_speech_gain != 1
                or settings.speech_speed != 1):
            settings.generated_action = None
            expr.rebuild(scene)
        scene.xr_ellen.face_mode = 'OFF'
        scene.xr_ellen.face_take = ad.action
        if sound and replace_end is not None:
            from sam_ellen_editing import sound_strips
            prior_audio=json.loads(checkpoint).get('audio') if checkpoint else None
            sound['v2_lipsync_owner']=mesh.name
            if audio_owner:sound['sam_audio_owner']=audio_owner.name
            for other in sound_strips(scene):
                if other==sound or other.frame_final_end<=start or other.frame_final_start>replace_end:continue
                owned=(other.get('v2_lipsync_owner')==mesh.name or other.name==prior_audio
                       or audio_owner and other.get('sam_audio_owner')==audio_owner.name)
                if owned:
                    muted_sounds[other.name]=other.mute;other.mute=True
            # Keep natural-assistance reapplication from reactivating the video sound.
            if audio_owner:
                actions=[audio_owner.animation_data.action if audio_owner.animation_data else None,
                         audio_owner.get('mocap_v2_base_action'),audio_owner.get('mocap_v2_assisted_action')]
                for body_action in actions:
                    if not body_action or body_action.name in audio_actions:continue
                    audio_actions[body_action.name]=body_action.get('sam_audio_tracks')
                    tracks=json.loads(body_action.get('sam_audio_tracks','[]'))
                    tracks=[name for name in tracks if name not in muted_sounds and name!=sound.name]
                    body_action['sam_audio_tracks']=json.dumps(tracks+[sound.name],ensure_ascii=False)
        saved = {'face': before, 'audio': sound.name if sound else None,
                 'frame_end': old_end, 'generated_end': scene.frame_end,
                 'sync_mode': audio_state[0], 'use_audio': audio_state[1],
                 'audio_mutes':muted_sounds,'audio_actions':audio_actions,
                 'audio_owner':audio_owner.name if audio_owner else None}
        if not _capture:
            mesh[CHECKPOINT] = json.dumps(saved, ensure_ascii=False)
        elif CHECKPOINT in mesh:
            # Capture's normal restore / per-clip checkpoints own this take.
            del mesh[CHECKPOINT]
        scene.frame_set(scene.frame_current)
        return {'duration': track['duration'], 'frames': len(frames), 'mesh': mesh.name,
                'audio': sound.name if sound else None, 'action': ad.action.name}
    except Exception:
        restore_audio_state(scene,muted_sounds,audio_actions)
        if sound:
            scene.sequence_editor.strips.remove(sound)
        scene.frame_end = old_end
        scene.sync_mode, scene.use_audio = audio_state
        restore_face(before, scene)
        if checkpoint is None:
            if CHECKPOINT in mesh:
                del mesh[CHECKPOINT]
        else:
            mesh[CHECKPOINT] = checkpoint
        for created in list(bpy.data.actions):
            if created.as_pointer() not in actions_before and created.users <= int(created.use_fake_user):
                bpy.data.actions.remove(created)
        raise


def bake_capture(scene, rig, filepath, timeline=None):
    """Use the capture rig, bypassing only the manual operator's busy guard."""
    from sam_ellen_face import face_object
    mesh = face_object(scene, rig)
    if mesh is None:
        raise ValueError('找不到动捕人物的面部网格')
    selected = scene.ellen_expression_object
    scene.ellen_expression_object = mesh
    try:
        settings = scene.sam_ellen_editing
        return bake(scene, filepath, int(timeline['start']) if timeline else 1,
                    settings.lipsync_gain, settings.lipsync_lead, attach_audio=False,
                    _capture=True, timeline=timeline,
                    replace_end=int(rig['mocap_v2_frame_count']) if not timeline and rig.get('mocap_v2_frame_count') else None)
    except Exception:
        scene.ellen_expression_object = selected
        raise


def restore_audio_state(scene,mutes,actions):
    from sam_ellen_editing import sound_strips
    for strip in sound_strips(scene):
        if strip.name in mutes:strip.mute=mutes[strip.name]
    for name,value in actions.items():
        action=bpy.data.actions.get(name)
        if not action:continue
        if value is None:
            if 'sam_audio_tracks' in action:del action['sam_audio_tracks']
        else:action['sam_audio_tracks']=value


def restore_previous(scene):
    import ellen_expression_timeline as expr
    from sam_ellen_editing import available, sound_strips
    from sam_ellen_face import restore_face
    available()
    mesh = expr.face_object(scene)
    if not mesh or CHECKPOINT not in mesh:
        raise ValueError('此角色没有音频口型备份')
    saved = json.loads(mesh[CHECKPOINT])
    state = saved['face']
    refs = [state['action'], *[state.get('track', {}).get(n)
            for n in ('base_action', 'generated_action')]]
    if any(name and name not in bpy.data.actions for name in refs):
        raise ValueError('原始表情动作备份缺失，未执行恢复')
    restore_face(state, scene)
    restore_audio_state(scene,saved.get('audio_mutes',{}),saved.get('audio_actions',{}))
    owner=bpy.data.objects.get(saved.get('audio_owner') or '')
    audio_actions=saved.get('audio_actions',{})
    if owner and audio_actions and owner.animation_data and owner.animation_data.action:
        # Natural assistance can copy/replace the body action after dubbing.
        # Restore its ownership too, not just the action names in the checkpoint.
        value=next(iter(audio_actions.values()))
        current=owner.animation_data.action
        if value is None:
            if 'sam_audio_tracks' in current:del current['sam_audio_tracks']
        else:current['sam_audio_tracks']=value
    for strip in sound_strips(scene):
        if strip.name == saved['audio']:
            scene.sequence_editor.strips.remove(strip)
    if scene.frame_end == saved['generated_end']:
        scene.frame_end = saved['frame_end']
    scene.sync_mode, scene.use_audio = saved['sync_mode'], saved['use_audio']
    del mesh[CHECKPOINT]
    scene.frame_set(scene.frame_current)


class SAMAudioLipSync(bpy.types.Operator, ImportHelper):
    bl_idname = 'sam_ellen.audio_lipsync'
    bl_label = '选择配音并生成口型'
    bl_description = '替换当前表情角色的整条说话口型；保留身体动作与眉眼，可恢复生成前口型'
    bl_options = {'REGISTER', 'UNDO'}
    filter_glob: StringProperty(default='*.wav;*.mp3;*.flac;*.ogg;*.m4a;*.aac;*.mp4', options={'HIDDEN'})

    def execute(self, context):
        try:
            s = context.scene.sam_ellen_editing
            result = bake(context.scene, self.filepath, s.audio_start, s.lipsync_gain,
                          s.lipsync_lead, s.lipsync_import_audio)
            s.status = f"已生成 {result['duration']:.2f} 秒口型：{result['mesh']}"
            self.report({'INFO'}, s.status)
            return {'FINISHED'}
        except Exception as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}


class SAMRestoreAudioLipSync(bpy.types.Operator):
    bl_idname = 'sam_ellen.restore_audio_lipsync'
    bl_label = '恢复生成前口型'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            restore_previous(context.scene)
            context.scene.sam_ellen_editing.status = '已恢复生成前口型'
            return {'FINISHED'}
        except Exception as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}


CLASSES = (SAMAudioLipSync, SAMRestoreAudioLipSync)

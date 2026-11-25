"""Independent voice tracks and explicit removal of the latest appended take."""
import json
import math
from pathlib import Path

import bpy
from bpy.props import BoolProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from bpy_extras.io_utils import ImportHelper

from sam_ellen_sequence import curves, validate_animation


def available():
    import sam_ellen_bridge as bridge
    import xr_ellen_bridge as xr
    if bridge._work or xr._runtime:
        raise ValueError('请先完成动捕任务并断开 XR 连接')


def sound_strips(scene):
    editor = scene.sequence_editor
    return [s for s in editor.strips if s.type == 'SOUND'] if editor else []


def import_audio(scene, filepath, start, volume=1, extend=False):
    path = Path(bpy.path.abspath(filepath)).resolve()
    if not path.is_file():
        raise ValueError('音频文件不存在')
    editor = scene.sequence_editor_create()
    # Use a free channel so existing edits never get moved or overwritten.
    used = {s.channel for s in editor.strips}
    channel = next((i for i in range(1, 129) if i not in used), None)
    if channel is None:
        raise ValueError('音轨通道已满，请先清理不需要的片段')
    strip = None
    try:
        strip = editor.strips.new_sound('配音 - '+path.stem, str(path), channel, int(start))
        if strip.frame_final_duration <= 0:
            raise ValueError('无法读取此音频，请先转换为 WAV')
        strip.volume = volume
        strip.show_waveform = True
        scene.sync_mode = 'AUDIO_SYNC'
        scene.use_audio = True
        if extend:
            scene.frame_end = max(scene.frame_end, math.ceil(strip.frame_final_end)-1)
        return strip
    except Exception:
        if strip:
            editor.strips.remove(strip)
        raise


def last_plan(scene):
    rig = scene.sam_ellen.rig
    action = rig.animation_data.action if rig and rig.animation_data else None
    plan = json.loads(action.get('sam_sequence', 'null')) if action else None
    if not plan or plan['start'] <= plan['prefix_end']:
        raise ValueError('当前没有可删除的最后追加段；整轮撤回请用“恢复导入前动作”')
    return rig, action, plan


def trim_copy(owner, source, end, created):
    if not source:
        return None
    result = source.copy()
    created.append(result)
    source.use_fake_user = result.use_fake_user = True
    result.name = source.name+' 截至 '+str(end)
    # Preserve preceding bezier handles; removing a key can recalculate AUTO handles.
    for curve in curves(result):
        retained = [(p.co.x, tuple(p.handle_left), tuple(p.handle_right), p.interpolation)
                    for p in curve.keyframe_points if p.co.x <= end]
        for index in range(len(curve.keyframe_points)-1, -1, -1):
            if curve.keyframe_points[index].co.x > end:
                curve.keyframe_points.remove(curve.keyframe_points[index], fast=True)
        curve.update()
        for point, (_, left, right, interpolation) in zip(curve.keyframe_points, retained):
            point.handle_left_type = point.handle_right_type = 'FREE'
            point.handle_left, point.handle_right = left, right
            point.interpolation = interpolation
    return result


def delete_last(scene):
    available()
    from sam_ellen_face import face_object, face_state, restore_face
    from sam_ellen_retarget import snapshot, restore
    settings = scene.sam_ellen
    rig, action, plan = last_plan(scene)
    mesh = face_object(scene, rig)
    validate_animation(rig.animation_data)
    if mesh:
        validate_animation(mesh.data.shape_keys.animation_data)
    state = snapshot(rig, scene)
    facial = face_state(mesh, scene) if mesh else None
    saved = {name: getattr(settings, name) for name in
             ('take', 'face_take', 'take_path', 'video', 'job_id', 'status', 'face_status')}
    checkpoint = json.loads(action.get('sam_previous_clip', 'null'))
    created = []
    try:
        if checkpoint:
            previous = checkpoint['body']['action']
            if previous and previous not in bpy.data.actions:
                raise ValueError('前一段动作备份缺失，未执行删除')
            face = checkpoint.get('face')
            if face:
                references = [face.get('action')]+[
                    face.get('track', {}).get(k) for k in ('base_action', 'generated_action')]
                if any(name and name not in bpy.data.actions for name in references):
                    raise ValueError('前一段面部备份缺失，未执行删除')
            restore(rig, scene, checkpoint['body'])
            if face:
                restore_face(face, scene)
            for name, value in checkpoint['settings'].items():
                setattr(settings, name, bpy.data.actions.get(value) if name in ('take', 'face_take') and value else value)
        else:
            # Older files have a boundary but no per-import checkpoint.
            # Trim copies of all facial sources as well as the displayed result.
            end = plan['prefix_end']
            body = trim_copy(rig, action, end, created)
            for name in list(body.keys()):
                if name.startswith('sam_'):
                    del body[name]
            body['sam_job_id'] = 'trimmed-prefix'
            body['sam_removed_tail'] = action.name
            ad = rig.animation_data
            slot = ad.action_slot.identifier
            ad.action = body
            ad.action_slot = next(s for s in body.slots if s.identifier == slot)
            if mesh:
                keys = mesh.data.shape_keys
                ad = keys.animation_data_create()
                old, slot = ad.action, ad.action_slot.identifier if ad.action_slot else None
                clipped = trim_copy(keys, old, end, created)
                ad.action = clipped
                if clipped and slot:
                    ad.action_slot = next(s for s in clipped.slots if s.identifier == slot)
                track = mesh.ellen_expression_track
                if track.base_action:
                    track.base_action = trim_copy(keys, track.base_action, end, created)
                if track.generated_action == old:
                    track.generated_action = clipped
                scene.xr_ellen.face_take = clipped
                settings.face_take = clipped
            settings.take = body
            settings.take_path = settings.job_id = ''
            scene.frame_end = max(scene.frame_start, end)
        scene.frame_set(min(scene.frame_current, scene.frame_end))
        from sam_ellen_video_audio import activate
        activate(scene, rig, rig.animation_data.action)
        settings.status = f'已删除最后追加段；当前结束于第 {scene.frame_end} 帧，独立配音保留'
        settings.face_status = '口型已随动作一同撤回'
    except Exception:
        restore(rig, scene, state)
        if facial:
            restore_face(facial, scene)
        for name, value in saved.items():
            setattr(settings, name, value)
        for item in reversed(created):
            if item.users <= int(item.use_fake_user):
                bpy.data.actions.remove(item)
        raise


class SAMEditingSettings(bpy.types.PropertyGroup):
    video_audio: BoolProperty(name='随动捕导入视频声音', default=True,
                             description='成功生成动作后导入视频音轨；追加时接在对应动作之后')
    video_audio_lipsync: BoolProperty(name='同时根据视频声音生成口型', default=False,
        description='用导入的视频音轨生成本段说话口型，替代视频嘴部识别；保留眉眼表情，追加时保留前段口型')
    movie_audio: BoolProperty(name='自动设置有声 MP4 输出', default=True,
                             description='有视频音轨时设置 H.264 / AAC；保留输出路径')
    audio_start: IntProperty(name='配音起始帧', default=1)
    audio_volume: FloatProperty(name='配音音量', default=1, min=0, max=2, subtype='FACTOR')
    extend_range: BoolProperty(name='播放范围延长至配音结束', default=False)
    lipsync_gain: FloatProperty(name='生成基础幅度', default=1, min=0, max=2,
        description='生成时使用的基础幅度；生成后请用“已生成口型幅度”即时调整')
    lipsync_lead: FloatProperty(name='口型提前（毫秒）', default=30, min=-200, max=200)
    lipsync_import_audio: BoolProperty(name='同时导入这段配音', default=True)
    status: StringProperty()


class SAMImportAudio(bpy.types.Operator, ImportHelper):
    bl_idname = 'sam_ellen.import_audio'
    bl_label = '导入独立配音'
    bl_options = {'REGISTER', 'UNDO'}
    filter_glob: StringProperty(default='*.wav;*.mp3;*.flac;*.ogg;*.m4a;*.aac;*.wma', options={'HIDDEN'})

    def execute(self, context):
        try:
            available()
            s = context.scene.sam_ellen_editing
            strip = import_audio(context.scene, self.filepath, s.audio_start, s.audio_volume, s.extend_range)
            s.status = f'已导入：{Path(self.filepath).name}，第 {int(strip.frame_start)} 帧开始'
            self.report({'INFO'}, '配音已导入，按空格与动作同步播放')
            return {'FINISHED'}
        except Exception as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}


class SAMAudioCurrentFrame(bpy.types.Operator):
    bl_idname = 'sam_ellen.audio_current_frame'
    bl_label = '使用当前帧'
    bl_options = {'UNDO'}

    def execute(self, context):
        context.scene.sam_ellen_editing.audio_start = context.scene.frame_current
        return {'FINISHED'}


class SAMPrepareMovie(bpy.types.Operator):
    bl_idname = 'sam_ellen.prepare_movie_audio'
    bl_label = '设置有声 MP4 输出'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from sam_ellen_video_audio import prepare_movie
        prepare_movie(context.scene)
        self.report({'INFO'}, '已设置 MP4 / H.264 / AAC，渲染动画将包含未静音的音轨')
        return {'FINISHED'}


class SAMRemoveAudio(bpy.types.Operator):
    bl_idname = 'sam_ellen.remove_audio'
    bl_label = '移除这条配音'
    bl_options = {'UNDO'}
    strip_name: StringProperty()

    def execute(self, context):
        try:
            available()
            strip = next((s for s in sound_strips(context.scene) if s.name == self.strip_name), None)
            if not strip:
                raise ValueError('找不到这条配音')
            context.scene.sequence_editor.strips.remove(strip)
            return {'FINISHED'}
        except Exception as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}


class SAMDeleteLast(bpy.types.Operator):
    bl_idname = 'sam_ellen.delete_last'
    bl_label = '删除最后追加的一段'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            available()
            last_plan(context.scene)
            return True
        except Exception:
            return False

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=430)

    def draw(self, context):
        _, _, plan = last_plan(context.scene)
        self.layout.label(text=f'删除第 {plan["start"]} 帧起的最后一段及衔接帧')
        self.layout.label(text=f'身体与口型一起撤回，保留至第 {plan["prefix_end"]} 帧')
        self.layout.label(text='独立配音保留；可按 Ctrl+Z 撤销', icon='INFO')

    def execute(self, context):
        try:
            delete_last(context.scene)
            self.report({'INFO'}, context.scene.sam_ellen.status)
            return {'FINISHED'}
        except Exception as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}


class SAMApplySpeechGain(bpy.types.Operator):
    bl_idname = 'sam_ellen.apply_speech_gain'
    bl_label = '仅应用口型幅度'
    bl_description = '应用此角色已生成口型的幅度，无需重新识别视频、分析声音或生成身体动作'
    bl_options = {'REGISTER', 'UNDO'}
    mesh_name: StringProperty(options={'HIDDEN'})

    def execute(self, context):
        try:
            import ellen_expression_timeline as expr
            mesh = context.scene.objects.get(self.mesh_name) if self.mesh_name else expr.face_object(context.scene)
            expr.apply_speech_gain(context.scene, mesh)
            message = f'已仅应用口型幅度：{mesh.ellen_expression_track.global_speech_gain:.2f}'
            context.scene.sam_ellen_editing.status = message
            self.report({'INFO'}, message)
            return {'FINISHED'}
        except Exception as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}


class SAMApplySpeechSpeed(bpy.types.Operator):
    bl_idname = 'sam_ellen.apply_speech_speed'
    bl_label = '仅应用口型速度'
    bl_description = '从原始口型调整变化速度；不重新识别视频或分析音频，也不改变台词时长'
    bl_options = {'REGISTER', 'UNDO'}
    mesh_name: StringProperty(options={'HIDDEN'})

    def execute(self, context):
        try:
            import ellen_expression_timeline as expr
            mesh = context.scene.objects.get(self.mesh_name) if self.mesh_name else expr.face_object(context.scene)
            expr.apply_speech_gain(context.scene, mesh)
            message = f'已应用口型变化速度：{mesh.ellen_expression_track.speech_speed:.2f}'
            context.scene.sam_ellen_editing.status = message
            self.report({'INFO'}, message)
            return {'FINISHED'}
        except Exception as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}


def draw_video_audio_options(layout, scene):
    settings = scene.sam_ellen_editing
    layout.prop(settings, 'video_audio')
    column = layout.column()
    column.enabled = settings.video_audio
    column.prop(settings, 'video_audio_lipsync')
    if settings.video_audio and settings.video_audio_lipsync:
        column.prop(settings, 'lipsync_gain')
        column.prop(settings, 'lipsync_lead')


class SAMEditingPanel(bpy.types.Panel):
    bl_idname = 'SAM_PT_ellen_editing'
    bl_label = '配音与动作管理'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SAM 动捕'
    bl_order = -1

    def draw(self, context):
        layout = self.layout
        s = context.scene.sam_ellen_editing
        import ellen_expression_timeline as expr
        box = layout.box()
        box.label(text='口型幅度与速度')
        if hasattr(context.scene, 'ellen_expression_object'):
            box.prop(context.scene, 'ellen_expression_object', text='说话角色')
        expr.draw_live_speech_gain(box, context.scene)
        layout.prop(s, 'movie_audio')
        layout.operator('sam_ellen.prepare_movie_audio', icon='FILE_MOVIE')
        if s.status:
            import textwrap
            for line in textwrap.wrap(s.status, width=max(16, int(context.region.width/13))):
                layout.label(text=line)
        layout.separator()
        row = layout.row(align=True)
        row.prop(s, 'audio_start')
        row.operator('sam_ellen.audio_current_frame', text='', icon='TIME')
        layout.prop(s, 'audio_volume')
        layout.prop(s, 'extend_range')
        layout.operator('sam_ellen.import_audio', icon='SOUND')
        box = layout.box()
        box.label(text='音频驱动口型')
        box.label(text='替换所选角色整条说话口型')
        box.prop(s, 'lipsync_gain')
        box.prop(s, 'lipsync_lead')
        box.prop(s, 'lipsync_import_audio')
        box.operator('sam_ellen.audio_lipsync', icon='SOUND')
        box.operator('sam_ellen.restore_audio_lipsync', icon='LOOP_BACK')
        for strip in sound_strips(context.scene):
            box = layout.box()
            row = box.row(align=True)
            row.prop(strip, 'name', text='')
            row.prop(strip, 'mute', text='', icon='HIDE_ON' if strip.mute else 'HIDE_OFF')
            row.operator('sam_ellen.remove_audio', text='', icon='X').strip_name = strip.name
            box.prop(strip, 'frame_start', text='起始帧')
            box.prop(strip, 'volume', text='音量')
        row = layout.row(align=True)
        row.operator('screen.animation_play', text='播放 / 暂停', icon='PLAY')
        layout.separator()
        layout.operator('sam_ellen.delete_last', icon='TRASH')
        if context.scene.sam_ellen.previous_state:
            layout.operator('sam_ellen.restore', text='恢复导入前动作（整轮撤回）', icon='LOOP_BACK')


CLASSES = (SAMEditingSettings, SAMImportAudio, SAMAudioCurrentFrame, SAMRemoveAudio,
           SAMPrepareMovie, SAMDeleteLast, SAMApplySpeechGain, SAMApplySpeechSpeed, SAMEditingPanel)

from sam_ellen_audio_lipsync import CLASSES as LIPSYNC_CLASSES
CLASSES += LIPSYNC_CLASSES


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.sam_ellen_editing = PointerProperty(type=SAMEditingSettings)


def unregister():
    del bpy.types.Scene.sam_ellen_editing
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)

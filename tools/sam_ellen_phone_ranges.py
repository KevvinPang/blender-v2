"""Frame intervals for a reversible phone overlay over the main capture preset."""
import json
from pathlib import Path

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty

WEIGHT = 'sam_phone_grip_weight'
VISIBLE = 'sam_phone_visible'
_busy = False


def ranges(settings):
    from sam_ellen_gaze import merged_ranges
    values = [(r.start_frame, r.end_frame) for r in settings.phone_ranges if r.enabled]
    if any(end < start for start, end in values):
        raise ValueError('持手机的结束帧不能早于开始帧')
    return merged_ranges(values)


def configured_side(scene):
    s = scene.sam_ellen
    return s.phone_side if s.phone_enabled else None


def dirty(self, context):
    if hasattr(self.id_data, 'sam_ellen'):
        self.id_data.sam_ellen.phone_status = '帧区间有修改，待应用'


def changed(settings, context):
    global _busy
    if _busy or not settings.rig:
        return
    try:
        # Disabling an old full-clip preset must also release baked finger keys.
        restore_legacy_fingers(settings.id_data)
        apply(settings.id_data)
    except (ValueError, RuntimeError, OSError) as error:
        settings.phone_status = str(error)


def migrate(scene):
    global _busy
    s = scene.sam_ellen
    raw = s.get('capture_mode')
    if raw not in (4, 5):
        return False
    _busy = True
    try:
        s['capture_mode'] = 3
        s.phone_enabled = True
        s.phone_side = 'L' if raw == 4 else 'R'
        if not s.phone_ranges:
            r = s.phone_ranges.add()
            r.start_frame, r.end_frame = scene.frame_start, scene.frame_end
        s.phone_fade_frames = 0
    finally:
        _busy = False
    return True


def weight_points(intervals, fade):
    """Integer-frame fade, entirely inside inclusive ranges; zero elsewhere."""
    from sam_ellen_gaze import contact_weight
    points = {}
    for start, end in intervals:
        points[start - 1] = 0.
        points[end + 1] = 0.
        span = min(fade, (end - start) / 2 + 1)
        if span <= 1:
            points[start] = points[end] = 1.
        else:
            steps = min(end-start+1, int(span) + 2)
            for frame in {start+i for i in range(steps)} | {end-i for i in range(steps)}:
                points[frame] = 1 - contact_weight(frame, [(start, end)], fade)
    return sorted(points.items()) or [(1, 0.)]


def set_curve(scene, name, points, interpolation):
    from sam_ellen_sequence import curves
    scene[name] = 0.
    path = '["' + name + '"]'
    ad = scene.animation_data_create()
    if ad.action is None:
        ad.action = bpy.data.actions.new('SAM Phone Frame Intervals')
    elif ad.action.users > 1:
        ad.action = ad.action.copy()
    existing = next((c for c in curves(ad.action, ad.action_slot) if c.data_path == path), None)
    curve = existing or ad.action.fcurve_ensure_for_datablock(scene, path)
    curve.keyframe_points.clear()
    curve.keyframe_points.add(len(points))
    curve.keyframe_points.foreach_set('co', [v for frame, value in points for v in (frame, value)])
    for key in curve.keyframe_points:
        key.interpolation = interpolation
    curve.extrapolation = 'CONSTANT'
    curve.update()


def drive(owner, path, scene, name):
    curve = owner.driver_add(path)
    driver = curve.driver
    driver.type = 'SCRIPTED'
    while driver.variables:
        driver.variables.remove(driver.variables[0])
    var = driver.variables.new()
    var.name = 'phone'
    var.type = 'SINGLE_PROP'
    var.targets[0].id_type = 'SCENE'
    var.targets[0].id = scene
    var.targets[0].data_path = '["' + name + '"]'
    driver.expression = 'phone'


def install_timing(scene, rig, display, enabled):
    from sam_ellen_phone import TAG, apply_grip, phone_profile
    s = scene.sam_ellen
    intervals = ranges(s) if enabled else []
    # Validate before changing any evaluated animation.
    grip_intervals = intervals
    if enabled and phone_profile(rig)[1].get('grip_transition_outside_visibility'):
        # Complete the calibrated contact pose before the solid prop appears.
        # Release after it disappears, so the case never slices through fingers
        # while they blend between the captured hand and the fitted grip.
        from sam_ellen_gaze import merged_ranges
        padding = max(0, s.phone_fade_frames)
        grip_intervals = merged_ranges([(a-padding, b+padding) for a, b in intervals])
    weights = weight_points(grip_intervals, s.phone_fade_frames)
    visibility = {}
    for start, end in intervals:
        visibility[start-1] = 0.
        visibility[start] = 1.
        visibility[end+1] = 0.
    if rig == s.rig:
        set_curve(scene, WEIGHT, weights, 'LINEAR')
        set_curve(scene, VISIBLE, sorted(visibility.items()) or [(1, 0.)], 'CONSTANT')
    if enabled:
        apply_grip(rig, s.phone_side)
    for bone in rig.pose.bones:
        constraint = bone.constraints.get(TAG)
        if constraint:
            if enabled and bone.mmd_bone.name_j.startswith('左' if s.phone_side == 'L' else '右'):
                drive(constraint, 'influence', scene, WEIGHT)
                constraint.mute = False
            else:
                try:
                    constraint.driver_remove('influence')
                except (TypeError, RuntimeError):
                    pass
                constraint.influence = 0.
    group = display.node_group
    socket = next(s for s in group.interface.items_tree if s.name == '按帧显示手机')
    value = getattr(display.properties.inputs, socket.identifier)
    drive(value, 'value', scene, VISIBLE)
    scene.frame_set(scene.frame_current, subframe=scene.frame_subframe)


def restore_legacy_fingers(scene):
    """Recover only old frozen finger curves; preserve all other authored keys."""
    from sam_ellen_phone import finger_rotations
    from sam_ellen_retarget import bake_steps, snapshot, restore
    from sam_ellen_sequence import curves
    s = scene.sam_ellen
    rig = s.rig
    old = rig.animation_data.action if rig and rig.animation_data else None
    if old is None or old.get('sam_capture_mode') not in ('PHONE_LEFT', 'PHONE_RIGHT'):
        return False
    if old.get('sam_sequence'):
        raise ValueError('旧版多段持手机动作请先按主预设重新适配，再应用手机帧区间')
    source = old.get('sam_take_path') or s.take_path
    if not source or not Path(bpy.path.abspath(source)).is_file():
        raise ValueError('旧版握姿缺少动捕缓存，请先用主预设重新适配以恢复手指动作')
    data = json.loads(Path(bpy.path.abspath(source)).read_text(encoding='utf8'))
    saved = snapshot(rig, scene)
    saved_camera = scene.camera
    names = finger_rotations(rig, 'L' if old['sam_capture_mode'] == 'PHONE_LEFT' else 'R')
    paths = {rig.pose.bones[n].path_from_id('rotation_quaternion') for n in names}
    temporary = None
    result = None
    try:
        iterator = bake_steps(scene, rig, data, s.smoothing, s.capture_mode,
                              s.inactive_pose, head_smoothing=s.head_smoothing)
        while True:
            try:
                next(iterator)
            except StopIteration as finished:
                temporary = finished.value
                break
        incoming = {(c.data_path, c.array_index): c for c in curves(temporary, rig.animation_data.action_slot)
                    if c.data_path in paths}
        restore(rig, scene, saved)
        result = old.copy()
        result.name = old.name + ' / 主预设手指'
        rig.animation_data.action = result
        if saved['slot']:
            rig.animation_data.action_slot = next(slot for slot in result.slots if slot.identifier == saved['slot'])
        for key, src in incoming.items():
            dst = result.fcurve_ensure_for_datablock(rig, key[0], index=key[1])
            dst.keyframe_points.clear()
            dst.keyframe_points.add(len(src.keyframe_points))
            dst.keyframe_points.foreach_set('co', [v for p in src.keyframe_points for v in p.co])
            for point in dst.keyframe_points:
                point.interpolation = 'LINEAR'
            dst.update()
        result['sam_capture_mode'] = s.capture_mode
        result['sam_phone_legacy_backup'] = old.name
        old.use_fake_user = True
        if s.take == old:
            s.take = result
    except BaseException:
        restore(rig, scene, saved)
        if result:
            bpy.data.actions.remove(result)
        raise
    finally:
        if temporary:
            temporary.use_fake_user = False
            bpy.data.actions.remove(temporary)
        scene.camera = saved_camera
        scene.render.fps, scene.render.fps_base = saved['fps']
        scene.frame_start, scene.frame_end, frame = saved['range']
        scene.frame_set(frame)
    return True


def apply(scene):
    from sam_ellen_phone import ensure, sync
    s = scene.sam_ellen
    if s.phone_enabled:
        ranges(s)
        ensure(scene)
    else:
        sync(scene)
    s.phone_status = ('已应用 %d 个持手机帧区间；区间外使用主预设' % len(ranges(s))
                      if s.phone_enabled else '未启用持手机，使用主预设')


class SAMPhoneRange(bpy.types.PropertyGroup):
    enabled: BoolProperty(name='启用', default=True, update=dirty)
    start_frame: IntProperty(name='开始帧', default=1, min=-1048574, max=1048574, update=dirty)
    end_frame: IntProperty(name='结束帧', default=61, min=-1048574, max=1048574, update=dirty,
                           description='包含结束帧；区间外隐藏手机并使用主预设手指动作')


class SAM_UL_phone_ranges(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, 'enabled', text='')
        row.label(text=f'{item.start_frame}–{item.end_frame} 帧')


class SAMPhoneRangeEdit(bpy.types.Operator):
    bl_idname = 'sam_ellen.phone_range'
    bl_label = '持手机帧区间'
    bl_options = {'REGISTER', 'UNDO'}
    operation: EnumProperty(items=[(v, v, '') for v in ('ADD','REMOVE','START','END','JUMP','PREVIEW','APPLY')])

    def execute(self, context):
        import sam_ellen_bridge as bridge
        if bridge._work:
            self.report({'ERROR'}, '请等待当前动捕任务完成')
            return {'CANCELLED'}
        scene, s = context.scene, context.scene.sam_ellen
        try:
            if self.operation == 'ADD':
                start = max((r.end_frame for r in s.phone_ranges), default=scene.frame_current-1) + 1
                item = s.phone_ranges.add()
                item.start_frame = start
                item.end_frame = start + round(2 * scene.render.fps / scene.render.fps_base)
                s.phone_range_index = len(s.phone_ranges)-1
                dirty(s, context)
            elif self.operation == 'APPLY':
                if s.phone_enabled:
                    ranges(s)
                restore_legacy_fingers(scene)
                apply(scene)
            else:
                if not 0 <= s.phone_range_index < len(s.phone_ranges):
                    raise ValueError('请先添加或选中一个持手机时间段')
                item = s.phone_ranges[s.phone_range_index]
                if self.operation == 'REMOVE':
                    s.phone_ranges.remove(s.phone_range_index)
                    s.phone_range_index = max(0, min(s.phone_range_index, len(s.phone_ranges)-1))
                    restore_legacy_fingers(scene)
                    apply(scene)
                elif self.operation == 'START':
                    item.start_frame = scene.frame_current
                elif self.operation == 'END':
                    item.end_frame = scene.frame_current
                elif self.operation == 'JUMP':
                    scene.frame_set(item.start_frame)
                elif self.operation == 'PREVIEW':
                    start, end = item.start_frame, item.end_frame
                    ranges(s)
                    restore_legacy_fingers(scene)
                    apply(scene)
                    scene.frame_set((start+end)//2)
            return {'FINISHED'}
        except (ValueError, RuntimeError, OSError) as error:
            s.phone_status = str(error)
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}


def draw(layout, s):
    box = layout.box()
    box.prop(s, 'phone_enabled')
    if not s.phone_enabled:
        return
    if s.rig:
        from sam_ellen_phone import phone_profile
        try:
            box.label(text='手机：' + phone_profile(s.rig)[1]['label'], icon='OUTLINER_OB_MESH')
        except ValueError as error:
            box.label(text=str(error), icon='ERROR')
    box.prop(s, 'phone_side')
    row = box.row()
    row.template_list('SAM_UL_phone_ranges', '', s, 'phone_ranges', s, 'phone_range_index', rows=2)
    buttons = row.column(align=True)
    buttons.operator('sam_ellen.phone_range', text='', icon='ADD').operation = 'ADD'
    buttons.operator('sam_ellen.phone_range', text='', icon='REMOVE').operation = 'REMOVE'
    if 0 <= s.phone_range_index < len(s.phone_ranges):
        item = s.phone_ranges[s.phone_range_index]
        for prop, op in (('start_frame','START'), ('end_frame','END')):
            row = box.row(align=True)
            row.prop(item, prop)
            row.operator('sam_ellen.phone_range', text='', icon='TIME').operation = op
        row = box.row(align=True)
        row.operator('sam_ellen.phone_range', text='跳到开始帧').operation = 'JUMP'
        row.operator('sam_ellen.phone_range', text='预览选中段').operation = 'PREVIEW'
    box.prop(s, 'phone_fade_frames')
    box.operator('sam_ellen.phone_range', text='应用持手机时间段', icon='CHECKMARK').operation = 'APPLY'
    if not s.phone_ranges:
        box.label(text='未设置区间：全程使用主预设', icon='INFO')
    box.label(text='区间外使用主预设，手机隐藏')
    box.operator('sam_ellen.phone_fit', icon='OUTLINER_OB_MESH')
    box.operator('sam_ellen.select_phone_character', icon='OBJECT_ORIGIN')
    if s.phone_status:
        import textwrap
        for line in textwrap.wrap(s.phone_status, width=24):
            box.label(text=line)
    try:
        ranges(s)
    except ValueError as error:
        box.label(text=str(error), icon='ERROR')


CLASSES = (SAMPhoneRange, SAM_UL_phone_ranges, SAMPhoneRangeEdit)

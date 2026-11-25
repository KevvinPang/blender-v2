"""Frame-based exceptions to camera eye contact; drawing is strictly read-only."""
import math
import json
import sys

import bpy
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, IntProperty
from mathutils import Matrix, Quaternion

EYE_SOURCE = 'sam_eye_contact_source'
EYE_SETTINGS = 'sam_eye_contact_settings'


def eye_options(settings):
    from sam_ellen_bridge import gaze_options
    options = gaze_options(settings)
    if options is not None:
        options = {k: v for k, v in options.items()
                   if k not in ('head_tracking', 'motion', 'noise_head')}
        options['head_tracking'] = False
        options['motion'] = 1.
    return options


def eye_settings_pending(scene):
    s = scene.sam_ellen
    action = s.rig.animation_data.action if s.rig and s.rig.animation_data else None
    if not action:
        return True
    try:
        applied = json.loads(action.get(EYE_SETTINGS, action.get('sam_camera_gaze', 'null')))
        if applied is not None:
            applied = {k: v for k, v in applied.items()
                       if k not in ('head_tracking', 'motion', 'noise_head', 'center_motion', 'eye_limit_degrees')}
            applied.update(head_tracking=False, motion=1.)
        return applied != eye_options(s)
    except (ValueError, TypeError):
        return True


def _eye_curves(action, slot, paths):
    from sam_ellen_sequence import curves
    return [c for c in curves(action, slot) if c.data_path in paths]


def _validate_eye_animation(rig, action, slot, paths):
    ad = rig.animation_data
    if ad.use_tweak_mode or any(not t.mute for t in ad.nla_tracks):
        raise ValueError('请先退出 NLA 编辑并将混合轨道烘焙或静音，再应用眼睛视线')
    if any(c.data_path in paths for c in ad.drivers):
        raise ValueError('眼睛旋转存在驱动器，请先烘焙后应用视线')
    seen = set()
    for c in _eye_curves(action, slot, paths):
        key = c.data_path, c.array_index
        if c.modifiers or c.mute or key in seen:
            raise ValueError('眼睛旋转含修改器、静音或多层混合，请先烘焙后应用视线')
        seen.add(key)


def _copy_curve(source, action, owner):
    """Copy original eye curves exactly, including authored Bezier handles."""
    if not action.slots:
        slot = action.slots.new('OBJECT', owner.name)
        layer = action.layers.new('Original eyes')
        layer.strips.new(type='KEYFRAME').channelbags.new(slot)
    bag = action.layers[0].strips[0].channelbags[0]
    dest = bag.fcurves.new(source.data_path, index=source.array_index)
    dest.extrapolation = source.extrapolation
    dest.keyframe_points.add(len(source.keyframe_points))
    for a, b in zip(source.keyframe_points, dest.keyframe_points):
        b.co = a.co
        b.interpolation, b.easing = a.interpolation, a.easing
        b.amplitude, b.back, b.period = a.amplitude, a.back, a.period
        b.type = a.type
        b.handle_left_type = b.handle_right_type = 'FREE'
        b.handle_left, b.handle_right = a.handle_left, a.handle_right
        b.handle_left_type, b.handle_right_type = a.handle_left_type, a.handle_right_type
    dest.update()
    return dest


def _write_eye_keys(curve, samples, index):
    curve.keyframe_points.clear()
    curve.keyframe_points.add(len(samples))
    curve.keyframe_points.foreach_set('co', [x for f, q in samples for x in (f, q[index])])
    for key in curve.keyframe_points:
        key.interpolation = 'LINEAR'
    curve.update()


def _eye_source(rig, action, slot, eyes):
    paths = {b.path_from_id('rotation_quaternion') for b in eyes}
    source = action.get(EYE_SOURCE)
    if isinstance(source, bpy.types.Action):
        return source, False
    source = bpy.data.actions.new('SAM · 原始眼睛姿态（勿编辑）')
    try:
        baseline = json.loads(action.get('sam_gaze_eye_base', '{}'))
        source['eye_baseline'] = json.dumps({b.name: list(
            Matrix(baseline[b.name]['basis']).to_quaternion()
            if b.name in baseline else b.rotation_quaternion) for b in eyes})
        # Old SAM takes already contain camera tracking. Their pre-gaze pose
        # is stored explicitly; never treat generated tracking as the baseline.
        if not baseline:
            for c in _eye_curves(action, slot, paths):
                _copy_curve(c, source, rig)
        return source, True
    except BaseException:
        bpy.data.actions.remove(source)
        raise


def _dependencies(obj):
    if obj.parent:
        yield obj.parent
    constraints = list(obj.constraints)
    if obj.type == 'ARMATURE':
        constraints += [c for b in obj.pose.bones for c in b.constraints if not c.mute and c.influence]
    for c in constraints:
        for prop in ('target', 'pole_target'):
            other = getattr(c, prop, None)
            if isinstance(other, bpy.types.Object):
                yield other
    if obj.animation_data:
        for fc in obj.animation_data.drivers:
            for var in fc.driver.variables:
                for target in var.targets:
                    if isinstance(target.id, bpy.types.Object):
                        yield target.id


def _copy_dependencies(scratch, roots, clones):
    """Evaluate only the rig, camera and their transform dependencies."""
    pending = list(roots)
    while pending:
        obj = pending.pop()
        if obj in clones:
            continue
        clone = obj.copy()
        clones[obj] = clone
        scratch.collection.objects.link(clone)
        pending.extend(_dependencies(obj))
    for obj, clone in clones.items():
        if obj.parent:
            clone.parent = clones[obj.parent]
        constraints = list(clone.constraints)
        if clone.type == 'ARMATURE':
            for b in clone.pose.bones:
                if hasattr(b, 'wiggle_enable'):
                    b.wiggle_enable = False
                constraints += list(b.constraints)
        for c in constraints:
            for prop in ('target', 'pole_target'):
                target = getattr(c, prop, None)
                if target in clones:
                    setattr(c, prop, clones[target])
        if clone.animation_data:
            for fc in clone.animation_data.drivers:
                for var in fc.driver.variables:
                    for target in var.targets:
                        if target.id in clones:
                            target.id = clones[target.id]


def _pose_dependencies(rig, names):
    """Bones needed by the head/eyes, including self-targeting constraints."""
    needed = set()
    pending = list(names)
    drivers = list(rig.animation_data.drivers) if rig.animation_data else []
    # Global custom-property drivers may feed the head's constraints as well.
    for fc in drivers:
        if not fc.data_path.startswith('pose.bones['):
            pending.extend(target.bone_target for var in fc.driver.variables for target in var.targets
                           if target.id == rig and target.bone_target)
    while pending:
        name = pending.pop()
        if name in needed or name not in rig.pose.bones:
            continue
        needed.add(name)
        bone = rig.pose.bones[name]
        if bone.parent:
            pending.append(bone.parent.name)
        for c in bone.constraints:
            if c.mute or not c.influence:
                continue
            for target_prop, bone_prop in (('target', 'subtarget'), ('pole_target', 'pole_subtarget')):
                if getattr(c, target_prop, None) == rig:
                    pending.append(getattr(c, bone_prop, ''))
        for fc in drivers:
            if fc.data_path.startswith(bone.path_from_id()+'.'):
                pending.extend(target.bone_target for var in fc.driver.variables for target in var.targets
                               if target.id == rig and target.bone_target)
    return needed


def apply_eye_steps(scene):
    """Sample current head/camera motion and commit only eye quaternion keys.

    No video/cache, face bake, retarget, camera fit or source-action replacement.
    The private scene contains no character meshes, so scrubbing it does not
    evaluate the heavy render scene. Closing the generator leaves actions intact.
    """
    import sam_ellen_retarget as rt
    from sam_ellen_sequence import curves
    s = scene.sam_ellen
    rig = s.rig
    if not rig or rig.type != 'ARMATURE' or not rig.animation_data or not rig.animation_data.action:
        raise ValueError('请先选择已有动作的角色骨架')
    ad = rig.animation_data
    action, slot = ad.action, ad.action_slot
    if not slot:
        raise ValueError('当前动作未绑定角色插槽')
    tweak = sys.modules.get('motion_tweak')
    if tweak and tweak._drag:
        raise ValueError('请先确认或取消当前动作微调')
    bones = rt.gaze_bones(rig)
    eyes = [bones[k] for k in ('両目', '左目', '右目')]
    if any(b.rotation_mode != 'QUATERNION' for b in eyes):
        raise ValueError('请先将眼睛旋转烘焙为四元数')
    paths = {b.path_from_id('rotation_quaternion') for b in eyes}
    targets = [(action, slot)]
    p = getattr(scene, 'motion_tweak', None)
    if p and p.rig == rig and action in (p.source, p.result):
        for target in (p.source, p.result):
            if target and target != action:
                target_slot = next((x for x in target.slots if x.identifier == p.slot_id), None)
                if not target_slot:
                    raise ValueError('微调原动作的眼睛插槽不匹配')
                targets.append((target, target_slot))
    for target, target_slot in targets:
        _validate_eye_animation(rig, target, target_slot, paths)
    options = eye_options(s)
    camera = s.gaze_camera or scene.camera
    if options is not None and (not camera or camera.type != 'CAMERA' or camera.name not in scene.objects):
        raise ValueError('请选择当前场景的目标摄像机')
    rate = scene.render.fps / scene.render.fps_base
    first, last = action.frame_range
    frames = sorted({float(first), float(last), *map(float, range(math.ceil(first), math.floor(last)+1)),
                     *(float(pt.co.x) for c in _eye_curves(action, slot, paths) for pt in c.keyframe_points)})
    if not frames or not all(math.isfinite(f) for f in frames) or rate <= 0:
        raise ValueError('当前动作的帧范围或帧率无效')
    source, created = _eye_source(rig, action, slot, eyes)
    scratch = None
    solver_rig = None
    sample_action = None
    clones = {}
    committed = False
    try:
        raw = {(c.data_path, c.array_index): c for c in curves(source)}
        baseline = json.loads(source['eye_baseline'])
        channels = {b.name: [(1+f-first, tuple(raw[b.path_from_id('rotation_quaternion'), i].evaluate(f)
                          if (b.path_from_id('rotation_quaternion'), i) in raw else baseline[b.name][i]
                          for i in range(4))) for f in frames] for b in eyes}
        if options is not None:
            scratch = bpy.data.scenes.new('SAM · 仅眼睛视线采样')
            scratch.render.fps, scratch.render.fps_base = scene.render.fps, scene.render.fps_base
            if hasattr(scratch, 'wiggle_enable'):
                scratch.wiggle_enable = False
            _copy_dependencies(scratch, (rig, camera), clones)
            copy_rig, copy_camera = clones[rig], clones[camera]
            needed = _pose_dependencies(rig, [b.name for b in bones.values()])
            needed_paths = tuple(rig.pose.bones[name].path_from_id()+'.' for name in needed)
            sample_action = action.copy()
            sample_action.name = 'SAM · 头眼采样临时动作'
            sample_action.use_fake_user = False
            for layer in sample_action.layers:
                for strip in layer.strips:
                    for bag in strip.channelbags:
                        for curve in list(bag.fcurves):
                            if curve.data_path.startswith('pose.bones[') and not curve.data_path.startswith(needed_paths):
                                bag.fcurves.remove(curve)
            copy_rig.animation_data.action = sample_action
            copy_rig.animation_data.action_slot = next(x for x in sample_action.slots if x.identifier == slot.identifier)
            for b in copy_rig.pose.bones:
                if b.name not in needed:
                    for c in b.constraints:
                        c.mute = True
            for driver in list(copy_rig.animation_data.drivers):
                if driver.data_path.startswith('pose.bones[') and not driver.data_path.startswith(needed_paths):
                    copy_rig.animation_data.drivers.remove(driver)
            solver_rig = copy_rig.copy()
            solver_rig.animation_data_clear()
            solver_rig.parent = None
            scratch.collection.objects.link(solver_rig)
            for b in solver_rig.pose.bones:
                for c in b.constraints:
                    if b.name not in needed:
                        c.mute = True
                    for prop in ('target', 'pole_target'):
                        if getattr(c, prop, None) == copy_rig:
                            setattr(c, prop, solver_rig)
            # Keep the animated evaluator separate from the unanimated solver.
            # Only head/eye dependencies need to be copied for each frame.
            cameras = [None] * len(frames)
            def sample(i):
                f = frames[i]
                scratch.frame_set(math.floor(f), subframe=f-math.floor(f))
                graph = scratch.view_layers[0].depsgraph
                evaluated = copy_rig.evaluated_get(graph)
                cameras[i] = (evaluated.matrix_world.inverted()
                              @ copy_camera.evaluated_get(graph).matrix_world.translation)
                solver_rig.matrix_world = evaluated.matrix_world
                for name in needed:
                    bone = evaluated.pose.bones[name]
                    dest = solver_rig.pose.bones[name]
                    dest.matrix_basis = bone.matrix_basis
                    for src_constraint, dest_constraint in zip(bone.constraints, dest.constraints):
                        dest_constraint.influence = src_constraint.influence
            data = {'fps': rate, 'frames': [{'time': (f-first)/rate} for f in frames],
                    '_sam_timeline': {'start': first, 'fps': rate}}
            solver_options = dict(options, eye_limit_degrees=5 if scene.get('CharacterStudio') else 20)
            eye_names = {bones[k].name for k in ('左目', '右目')}
            independent = not any(
                b.name in needed and (any(p.name in eye_names for p in b.parent_recursive)
                                      or any(getattr(c, 'target', None) == rig
                                             and getattr(c, 'subtarget', '') in eye_names
                                             for c in b.constraints))
                for b in rig.pose.bones if b.name not in eye_names)
            independent = independent and not any(
                p.name in eye_names for name in eye_names for p in rig.pose.bones[name].parent_recursive)
            yield from rt.correct_gaze(scratch, solver_rig, channels, data, solver_options, cameras,
                                       sample_pose=sample, independent_eyes=independent)
        else:
            yield len(frames), len(frames)
        if ad.action != action or ad.action_slot != slot or tweak and tweak._drag:
            raise ValueError('角色动作已在采样期间更换，眼睛结果未应用，请重试')
        # In-place writes preserve motion-tweak bindings, action metadata and
        # every non-eye key/handle. Also update its source so a later arm edit
        # cannot erase the eye settings.
        rollback = []
        try:
            for target, target_slot in targets:
                old = []
                for c in _eye_curves(target, target_slot, paths):
                    # Keep a snapshot of existing point handles for rollback.
                    old.append((c, [(tuple(k.co), tuple(k.handle_left), tuple(k.handle_right),
                                     k.handle_left_type, k.handle_right_type, k.interpolation,
                                     k.easing, k.amplitude, k.back, k.period, k.type)
                                    for k in c.keyframe_points]))
                before = {(c.data_path, c.array_index) for c, _ in old}
                props = {name: target.get(name) for name in (EYE_SOURCE, EYE_SETTINGS)}
                rollback.append((target, target_slot, old, before, props))
                for bone in eyes:
                    path = bone.path_from_id('rotation_quaternion')
                    samples = channels[bone.name]
                    previous = None
                    aligned = []
                    for frame, value in samples:
                        q = Quaternion(value).normalized()
                        if previous is not None and previous.dot(q) < 0:
                            q.negate()
                        aligned.append((first+frame-1, tuple(q)))
                        previous = q
                    for i in range(4):
                        # Target slots share the rig identifier. Existing curves
                        # are resolved explicitly to avoid touching other slots.
                        c = next((c for c in _eye_curves(target, target_slot, paths)
                                  if c.data_path == path and c.array_index == i), None)
                        if c is None:
                            bag = next(b for l in target.layers for strip in l.strips
                                       for b in strip.channelbags if b.slot_handle == target_slot.handle)
                            c = bag.fcurves.new(path, index=i)
                        _write_eye_keys(c, aligned, i)
                target[EYE_SOURCE] = source
                target[EYE_SETTINGS] = json.dumps(options, ensure_ascii=False)
            committed = True
        except BaseException:
            for target, target_slot, old, before, props in reversed(rollback):
                for c in _eye_curves(target, target_slot, paths):
                    if (c.data_path, c.array_index) not in before:
                        bag = next(b for l in target.layers for strip in l.strips
                                   for b in strip.channelbags if b.slot_handle == target_slot.handle)
                        bag.fcurves.remove(c)
                for c, points in old:
                    c.keyframe_points.clear()
                    c.keyframe_points.add(len(points))
                    for k, row in zip(c.keyframe_points, points):
                        co, left, right, lt, rt, interpolation, easing, amplitude, back, period, kind = row
                        k.co = co
                        k.interpolation, k.easing = interpolation, easing
                        k.amplitude, k.back, k.period, k.type = amplitude, back, period, kind
                        k.handle_left_type = k.handle_right_type = 'FREE'
                        k.handle_left, k.handle_right = left, right
                        k.handle_left_type, k.handle_right_type = lt, rt
                    c.update()
                for name, value in props.items():
                    if value is None:
                        if name in target: del target[name]
                    else:
                        target[name] = value
            raise
        scene.frame_set(scene.frame_current, subframe=scene.frame_subframe)
        return len(frames)
    finally:
        if solver_rig:
            bpy.data.objects.remove(solver_rig, do_unlink=True)
        for clone in clones.values():
            bpy.data.objects.remove(clone, do_unlink=True)
        if sample_action:
            bpy.data.actions.remove(sample_action)
        if scratch:
            bpy.data.scenes.remove(scratch)
        if created and not committed and source.users == 0:
            bpy.data.actions.remove(source)


class SAMApplyEyeContact(bpy.types.Operator):
    bl_idname = 'sam_ellen.apply_eye_contact'
    bl_label = '仅应用眼睛视线'
    bl_description = '只更新当前动作的眼睛关键帧；不重新识别视频，不生成身体、头部或口型动画；当前动作微调保留'

    def execute(self, context):
        import sam_ellen_bridge as bridge
        try:
            from sam_ellen_editing import available
            available()
            if context.screen and context.screen.is_animation_playing:
                raise ValueError('请先暂停播放再应用眼睛视线')
            s = context.scene.sam_ellen
            if not bpy.app.background:
                bpy.ops.ed.undo_push(message='眼睛视线更新前')
            bridge._work = {'scene': context.scene, 'eyes': apply_eye_steps(context.scene)}
            s.status, s.progress = '正在单独更新眼睛视线', 0
            bridge.run_timer()
            return {'FINISHED'}
        except Exception as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}


def merged_ranges(ranges):
    ordered = []
    for start, end in ranges:
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (start, end)):
            raise ValueError('不看镜头的帧范围必须是有限数值')
        if end < start:
            raise ValueError('不看镜头的结束帧不能早于开始帧')
        ordered.append((start, end))
    result = []
    for start, end in sorted(ordered):
        if result and start <= result[-1][1] + 1:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def contact_weight(frame, ranges, fade):
    """One outside exceptions; relax and return within their inclusive bounds."""
    for start, end in ranges:
        # Timestamp conversion can land a few ulps past an integer boundary.
        if start - 1e-6 <= frame <= end + 1e-6:
            frame = max(start, min(end, frame))
            if fade <= 0:
                return 0.
            span = min(fade, (end - start) / 2 + 1)
            t = min(1., (frame - start + 1) / span, (end - frame + 1) / span)
            return 1 - t*t*(3 - 2*t)
    return 1.


def range_options(settings):
    return merged_ranges((r.start_frame, r.end_frame) for r in settings.gaze_away_ranges if r.enabled)


class SAMGazeRange(bpy.types.PropertyGroup):
    enabled: BoolProperty(name='启用', default=True)
    start_frame: IntProperty(name='开始帧', default=1, min=-1048574, max=1048574)
    end_frame: IntProperty(name='结束帧', default=61, min=-1048574, max=1048574,
                           description='包含结束帧；范围内解除镜头注视，恢复原始眼睛姿态')


class SAM_UL_gaze_ranges(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, 'enabled', text='')
        row.label(text=f'{item.start_frame}–{item.end_frame} 帧')


class SAMGazeRangeEdit(bpy.types.Operator):
    bl_idname = 'sam_ellen.gaze_range'
    bl_label = '编辑不看镜头时间段'
    bl_options = {'REGISTER', 'UNDO'}
    operation: EnumProperty(items=[(v, v, '') for v in ('ADD', 'REMOVE', 'START', 'END', 'JUMP')])

    def execute(self, context):
        import sam_ellen_bridge as bridge
        if bridge._work:
            self.report({'ERROR'}, '请等待当前动捕任务完成')
            return {'CANCELLED'}
        scene = context.scene
        s = scene.sam_ellen
        if self.operation == 'ADD':
            item = s.gaze_away_ranges.add()
            item.start_frame = scene.frame_current
            item.end_frame = scene.frame_current + max(1, round(scene.render.fps / scene.render.fps_base))
            s.gaze_away_index = len(s.gaze_away_ranges) - 1
        else:
            if not 0 <= s.gaze_away_index < len(s.gaze_away_ranges):
                self.report({'ERROR'}, '请先添加或选中一个时间段')
                return {'CANCELLED'}
            item = s.gaze_away_ranges[s.gaze_away_index]
            if self.operation == 'REMOVE':
                s.gaze_away_ranges.remove(s.gaze_away_index)
                s.gaze_away_index = max(0, min(s.gaze_away_index, len(s.gaze_away_ranges) - 1))
            elif self.operation == 'START':
                item.start_frame = scene.frame_current
            elif self.operation == 'END':
                item.end_frame = scene.frame_current
            elif self.operation == 'JUMP':
                scene.frame_set(item.start_frame)
        return {'FINISHED'}


def draw_ranges(layout, settings):
    box = layout.box()
    box.label(text='不看镜头的时间段（帧）')
    if not any(r.enabled for r in settings.gaze_away_ranges):
        box.label(text='未设置：全程看向镜头', icon='INFO')
    row = box.row()
    row.template_list('SAM_UL_gaze_ranges', '', settings, 'gaze_away_ranges', settings, 'gaze_away_index', rows=2)
    buttons = row.column(align=True)
    buttons.operator('sam_ellen.gaze_range', text='', icon='ADD').operation = 'ADD'
    buttons.operator('sam_ellen.gaze_range', text='', icon='REMOVE').operation = 'REMOVE'
    if 0 <= settings.gaze_away_index < len(settings.gaze_away_ranges):
        item = settings.gaze_away_ranges[settings.gaze_away_index]
        row = box.row(align=True)
        row.prop(item, 'start_frame')
        row.operator('sam_ellen.gaze_range', text='', icon='TIME').operation = 'START'
        row = box.row(align=True)
        row.prop(item, 'end_frame')
        row.operator('sam_ellen.gaze_range', text='', icon='TIME').operation = 'END'
        box.operator('sam_ellen.gaze_range', text='跳到开始帧', icon='REW').operation = 'JUMP'
    box.prop(settings, 'gaze_away_fade')
    box.label(text='范围内恢复原眼姿，随头部转动')
    box.operator('sam_ellen.apply_eye_contact', icon='CHECKMARK')
    box.label(text='仅更新眼睛，保留当前头部和身体动作')
    try:
        range_options(settings)
    except ValueError as error:
        box.label(text=str(error), icon='ERROR')


CLASSES = (SAMGazeRange, SAM_UL_gaze_ranges, SAMGazeRangeEdit, SAMApplyEyeContact)

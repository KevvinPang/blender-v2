"""Independent, non-destructive motion editing; no capture-plugin dependencies.

Edits are sampled into an ordinary Action on confirmation. Playback and native
rendering need neither Python pose callbacks nor a running modal editor.
"""
bl_info = {'name': '动作微调 / Motion Tweak', 'author': 'Local Tools',
           'version': (0, 4, 0), 'blender': (5, 2, 0), 'category': 'Animation',
           'location': '3D View > Sidebar > 动作微调'}

import math
import sys
from pathlib import Path
if str(Path(__file__).resolve().parent) not in sys.path:sys.path.insert(0,str(Path(__file__).resolve().parent))
import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatVectorProperty,
                       IntProperty, PointerProperty, StringProperty)
from mathutils import Matrix, Quaternion, Vector
from bpy_extras import view3d_utils

PREFIX = 'MT · '
_keymaps = []
_draw = None
_drag = None


def editable(name):
    return name in {'Arm_L', 'Arm_R', 'Elbow_L', 'Elbow_R', 'Wrist_L', 'Wrist_R'} or (
        name.endswith(('_L', '_R')) and name.startswith(
            ('Thumb', 'IndexFinger', 'MiddleFinger', 'RingFinger', 'LittleFinger'))
        and not name.startswith('_') and 'Tip' not in name)


def part_label(name):
    side = '左' if name.endswith('_L') else '右'
    for prefix, label in [('Arm_', '上臂'), ('Elbow_', '前臂'), ('Wrist_', '手掌'),
                          ('Thumb', '拇指'), ('IndexFinger', '食指'),
                          ('MiddleFinger', '中指'), ('RingFinger', '无名指'),
                          ('LittleFinger', '小指')]:
        if name.startswith(prefix):
            return side + label + (' · ' + name.split('_')[0][-1] if 'Finger' in name or name.startswith('Thumb') else '')
    return name


def curves(action, slot_handle):
    return [fc for layer in action.layers for strip in layer.strips
            for bag in strip.channelbags if bag.slot_handle == slot_handle for fc in bag.fcurves]


def assign_action(rig, action, slot_id):
    slot = next((s for s in action.slots if s.identifier == slot_id), None)
    if slot is None:
        raise ValueError('动作插槽不匹配，未覆盖原动作')
    rig.animation_data_create().action = action
    rig.animation_data.action_slot = slot


def current_clip(scene):
    p = scene.motion_tweak
    return p.clips[p.active_clip] if 0 <= p.active_clip < len(p.clips) else None


def redraw():
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def refresh(self, context):
    if context:
        redraw()


def section_selected(self,context):
    if context and not self.advanced_ui:
        from motion_tweak_sections import select
        select(context.scene)
    refresh(self,context)


def part_selected(self,context):
    if context and not self.advanced_ui:
        from motion_tweak_sections import select
        select(context.scene,jump=False)
    refresh(self,context)


def section_boundary_changed(self,context):
    if self.simple_section:self.auto_section=False
    refresh(self,context)


class MTKey(bpy.types.PropertyGroup):
    bone: StringProperty()
    frame: IntProperty()
    offset: FloatVectorProperty(size=3, subtype='TRANSLATION')
    rotation: FloatVectorProperty(size=4, default=(1, 0, 0, 0), subtype='QUATERNION')
    radius: IntProperty(name='影响前后帧数', default=8, min=1, max=240)
    edge_start: BoolProperty(default=False)
    edge_end: BoolProperty(default=False)


class MTPart(bpy.types.PropertyGroup):
    bone: StringProperty()
    offset: FloatVectorProperty(size=3,name='位置偏移（厘米）',min=-25,max=25)
    rotation: FloatVectorProperty(size=3,name='朝向偏移',subtype='EULER',min=-1.0472,max=1.0472)


class MTClip(bpy.types.PropertyGroup):
    name: StringProperty(default='局部微调')
    start: IntProperty(name='起始帧', default=1, min=0,update=section_boundary_changed)
    end: IntProperty(name='结束帧', default=100, min=1,update=section_boundary_changed)
    enabled: BoolProperty(name='启用', default=True)
    side: EnumProperty(name='调整哪只手',items=[('L','左手','角色左手'),('R','右手','角色右手')],default='L')
    auto_section: BoolProperty(default=False)
    simple_section: BoolProperty(default=False)
    segment_basis: FloatVectorProperty(size=4,default=(1,0,0,0),subtype='QUATERNION')
    parts: CollectionProperty(type=MTPart)
    keys: CollectionProperty(type=MTKey)
    holds: CollectionProperty(type=MTKey)
    fade_in: IntProperty(name='渐入帧数', default=6, min=0,
                         description='整段调整在范围内从原动作渐入；0 为在边界后一帧切入')
    fade_out: IntProperty(name='渐出帧数', default=6, min=0,
                          description='整段调整在范围内渐回原动作；0 为在边界切回')


class MTLayerBackup(bpy.types.PropertyGroup):
    """Persistent editable session, including its original sampling rig."""
    rig: PointerProperty(type=bpy.types.Object)
    source: PointerProperty(type=bpy.types.Action)
    result: PointerProperty(type=bpy.types.Action)
    evaluator: PointerProperty(type=bpy.types.Object)
    slot_id: StringProperty()
    clips: CollectionProperty(type=MTClip)
    active_clip: IntProperty(default=-1)
    near_keys: CollectionProperty(type=MTKey)
    near_radius: IntProperty(default=8)
    edit_mode: StringProperty(default='NEAR')
    selected_bone: StringProperty()
    editing: BoolProperty()
    show_original: BoolProperty()
    status: StringProperty()


class MTSettings(bpy.types.PropertyGroup):
    rig: PointerProperty(type=bpy.types.Object, poll=lambda s, o: o.type == 'ARMATURE')
    source: PointerProperty(type=bpy.types.Action)
    result: PointerProperty(type=bpy.types.Action)
    evaluator: PointerProperty(type=bpy.types.Object)
    slot_id: StringProperty()
    clips: CollectionProperty(type=MTClip)
    active_clip: IntProperty(default=-1,update=section_selected)
    edit_mode: EnumProperty(name='调整方式', items=[
        ('NEAR', '修附近', '直接修改当前帧，自动平滑影响前后附近的动作'),
        ('RANGE', '修整段', '设置明确的起止帧，应用同样的修正量')], default='NEAR', update=refresh)
    near_keys: CollectionProperty(type=MTKey)
    near_radius: IntProperty(name='影响前后帧数', default=8, min=1, max=240,
                            description='下一次 G / R 确认时，以当前帧为中心向前后平滑影响的帧数')
    selected_bone: StringProperty(update=refresh)
    editing: BoolProperty(name='点击模型进行微调', default=False, update=refresh)
    show_original: BoolProperty(default=False)
    status: StringProperty(default='选择角色，点击“建立独立微调层”')
    saved_layers: CollectionProperty(type=MTLayerBackup)
    advanced_ui: BoolProperty(name='旧版高级面板',default=False)
    segment_part: EnumProperty(name='调整部位',items=[
        ('Wrist','整只手','移动整只手，手臂跟随'),('Thumb0','拇指','调整拇指的位置与朝向'),
        ('IndexFinger1','食指','调整食指的位置与朝向'),('MiddleFinger1','中指','调整中指的位置与朝向'),
        ('RingFinger1','无名指','调整无名指的位置与朝向'),('LittleFinger1','小指','调整小指的位置与朝向')],default='Wrist',update=part_selected)


LAYER_FIELDS = ('rig', 'source', 'result', 'evaluator', 'slot_id', 'active_clip',
                'near_radius', 'edit_mode', 'selected_bone', 'editing', 'show_original', 'status')
KEY_FIELDS = ('bone', 'frame', 'offset', 'rotation', 'radius', 'edge_start', 'edge_end')


def copy_layer(source, target):
    for field in LAYER_FIELDS:
        setattr(target, field, getattr(source, field))
    def copy_keys(keys, destination):
        destination.clear()
        for key in keys:
            item = destination.add()
            for field in KEY_FIELDS:
                setattr(item, field, getattr(key, field))
    copy_keys(source.near_keys, target.near_keys)
    target.clips.clear()
    for clip in source.clips:
        item = target.clips.add()
        for field in ('name', 'start', 'end', 'enabled', 'fade_in', 'fade_out','side','auto_section','simple_section','segment_basis'):
            setattr(item, field, getattr(clip, field))
        for part in clip.parts:
            dst=item.parts.add();dst.bone=part.bone;dst.offset=part.offset;dst.rotation=part.rotation
        copy_keys(clip.keys, item.keys)
        copy_keys(clip.holds, item.holds)


def current_action(p):
    return p.rig.animation_data.action if p.rig and p.rig.animation_data else None


def binding_changed(p):
    return bool(p.source) and current_action(p) not in (p.source, p.result)


def validate(p):
    rig = p.rig
    if not rig or rig.type != 'ARMATURE':
        raise ValueError('请先选择角色骨架')
    if not p.source or not p.evaluator:
        raise ValueError('请先建立独立微调层')
    if binding_changed(p):
        raise ValueError('当前动作已更换：请在“动作微调”面板点击“为当前动作新建微调层”；旧调整会保留')
    return rig


def setup(scene, rig):
    p = scene.motion_tweak
    if p.source:
        raise ValueError('已有微调层，请先清除后重建')
    if not rig or rig.type != 'ARMATURE' or not rig.animation_data or not rig.animation_data.action:
        raise ValueError('请选择已有动画的角色骨架')
    if not all(n in rig.pose.bones for n in ('Arm_L', 'Elbow_L', 'Wrist_L', 'Arm_R', 'Elbow_R', 'Wrist_R')):
        raise ValueError('第一版支持艾莲、玲使用的 Arm / Elbow / Wrist 骨架命名')
    if any(not t.mute for t in rig.animation_data.nla_tracks):
        raise ValueError('请先将 NLA 混合合成为独立动作，再建立微调层')
    if rig.animation_data.use_tweak_mode:
        raise ValueError('请先退出 NLA 编辑模式')
    if not rig.animation_data.action_slot:
        raise ValueError('当前动作没有绑定角色插槽，请先在动作编辑器中选择插槽')
    source = rig.animation_data.action
    slot_id = rig.animation_data.action_slot.identifier
    helper = rig.copy()
    helper_data = collection = result = None
    had_fake_user = source.use_fake_user
    try:
        helper.data = helper_data = rig.data.copy()
        helper.name = PREFIX + '原动作采样（勿编辑）'
        helper['motion_tweak_helper'] = True
        collection = bpy.data.collections.new(PREFIX + '内部采样')
        scene.collection.children.link(collection)
        collection.objects.link(helper)
        helper.hide_render = True
        helper.hide_set(True)
        helper.hide_select = True
        for b in helper.pose.bones:
            if hasattr(b, 'wiggle_enable'):
                b.wiggle_enable = False
            for c in b.constraints:
                if hasattr(c, 'target') and c.target == rig:
                    c.target = helper
        if helper.animation_data:
            for fc in helper.animation_data.drivers:
                for var in fc.driver.variables:
                    for target in var.targets:
                        if target.id == rig:
                            target.id = helper
        source.use_fake_user = True
        result = source.copy()
        result.use_fake_user = False
        result.name = source.name + ' · 微调'
        result['motion_tweak'] = True
        assign_action(rig, result, slot_id)
        p.rig, p.source, p.result, p.evaluator = rig, source, result, helper
        p.slot_id = slot_id
        p.show_original = False
        p.status = '点击部位，直接按 G / R；修附近无需先设置帧段'
    except Exception:
        # Restore directly, including when assign_action itself was the failure.
        rig.animation_data.action = source
        rig.animation_data.action_slot = next(s for s in source.slots if s.identifier == slot_id)
        source.use_fake_user = had_fake_user
        p.source = p.result = p.evaluator = None
        bpy.data.objects.remove(helper, do_unlink=True)
        if helper_data and helper_data.users == 0:
            bpy.data.armatures.remove(helper_data)
        if collection:
            bpy.data.collections.remove(collection)
        if result:
            bpy.data.actions.remove(result)
        raise
    return result


def use_current_action(scene):
    """Start a fresh session on the new take; archive old edits without rebaking."""
    p = scene.motion_tweak
    if _drag:
        raise ValueError('请先确认或取消当前 G / R 调整')
    if not binding_changed(p):
        raise ValueError('当前已绑定微调层，无需重新建立')
    rig = p.rig
    if not current_action(p):
        raise ValueError('角色当前没有动作，请先导入或选择动作')
    backup = p.saved_layers.add()
    copy_layer(p, backup)
    p.source = p.result = p.evaluator = None
    p.clips.clear(); p.near_keys.clear(); p.active_clip = -1
    p.editing = False; p.selected_bone = ''; p.show_original = False
    try:
        result = setup(scene, rig)
    except Exception:
        copy_layer(backup, p)
        p.saved_layers.remove(len(p.saved_layers)-1)
        raise
    p.status = '已为当前动作新建微调层；旧动作和调整记录已保留'
    scene.frame_set(scene.frame_current)
    redraw()
    return result


def resume_layer(scene):
    """Explicitly return to the stored result, leaving a foreign take untouched."""
    p = scene.motion_tweak
    if _drag:
        raise ValueError('请先确认或取消当前 G / R 调整')
    if not p.rig or not p.result or not p.evaluator:
        raise ValueError('旧微调结果不完整，请为当前动作新建微调层')
    foreign = current_action(p) if binding_changed(p) else None
    # A manually selected take may have no SAM pointer or other user. Keep it
    # through save/reopen when this explicit operation unlinks it from the rig.
    if not any(s.identifier == p.slot_id for s in p.result.slots):
        raise ValueError('旧微调动作插槽不匹配')
    if foreign:
        foreign.use_fake_user = True
    assign_action(p.rig, p.result, p.slot_id)
    p.show_original = False
    scene.frame_set(scene.frame_current)
    redraw()


def restore_previous_layer(scene, index=-1):
    p = scene.motion_tweak
    if p.source:
        validate(p)
    if _drag:
        raise ValueError('请先确认或取消当前 G / R 调整')
    if not p.saved_layers:
        raise ValueError('没有已保留的微调层')
    if index == -1:
        index = len(p.saved_layers)-1
    if not 0 <= index < len(p.saved_layers):
        raise ValueError('所选微调层已不存在')
    previous = p.saved_layers[index]
    if not previous.rig or not previous.result or not previous.evaluator:
        raise ValueError('已保留的微调层不完整')
    # Validate its slot before modifying any session metadata.
    if not any(s.identifier == previous.slot_id for s in previous.result.slots):
        raise ValueError('旧微调动作插槽不匹配')
    if p.source:
        saved = p.saved_layers.add()
        copy_layer(p, saved)
    copy_layer(previous, p)
    resume_layer(scene)
    p.saved_layers.remove(index)
    p.status = '已切回保留的微调层；刚才的微调层也已保留'


class MT_OT_UseCurrent(bpy.types.Operator):
    bl_idname = 'motion_tweak.use_current'
    bl_label = '为当前动作新建微调层'
    bl_description = '保留旧动作、微调结果和全部调整记录，为当前播放的动作建立空白微调层'
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        try:
            use_current_action(context.scene)
            self.report({'INFO'}, context.scene.motion_tweak.status)
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, str(e)); return {'CANCELLED'}


class MT_OT_Resume(bpy.types.Operator):
    bl_idname = 'motion_tweak.resume'
    bl_label = '切回旧微调结果'
    bl_description = '切回此微调层保存的旧动作；当前新动作的数据保留'
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        try:
            resume_layer(context.scene)
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, str(e)); return {'CANCELLED'}


class MT_OT_Previous(bpy.types.Operator):
    bl_idname = 'motion_tweak.previous'
    bl_label = '切回上一次微调层'
    bl_description = '恢复上一次的原动作、采样骨架和全部调整记录；当前微调层同时保留，可再次切回'
    bl_options = {'REGISTER', 'UNDO'}
    index: IntProperty(default=-1, options={'HIDDEN'})
    def execute(self, context):
        try:
            restore_previous_layer(context.scene, self.index)
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, str(e)); return {'CANCELLED'}


def smooth(t):
    return t*t*(3-2*t)


def hold_for(clip, bone):
    return next((k for k in clip.holds if k.bone == bone), None)


def hold_weight(clip, frame, hold=None):
    if frame < clip.start or frame > clip.end:
        return 0.0
    if frame==clip.start and not (hold and hold.edge_start):return 0.0
    if frame==clip.end and not (hold and hold.edge_end):return 0.0
    # Keep a full-strength interior even when the requested fades exceed the clip.
    available = max(0, clip.end-clip.start-2)
    fade_in = 0 if hold and hold.edge_start else clip.fade_in
    fade_out = 0 if hold and hold.edge_end else clip.fade_out
    total = fade_in+fade_out
    scale = min(1.0, available/total) if total else 1.0
    enter, leave = fade_in*scale, fade_out*scale
    a = smooth(min(1.0, (frame-clip.start)/enter)) if enter else 1.0
    b = smooth(min(1.0, (clip.end-frame)/leave)) if leave else 1.0
    return min(a, b)


def value_at(clip, bone, frame):
    """Endpoints stay neutral unless explicitly edited; never discard an anchor."""
    if not clip.enabled or frame < clip.start or frame > clip.end:
        return Vector((0, 0, 0)), Quaternion()
    hold = hold_for(clip, bone)
    if hold:
        w = hold_weight(clip, frame, hold)
        return Vector(hold.offset)*w, Quaternion().slerp(Quaternion(hold.rotation).normalized(), w)
    points = [(k.frame, Vector(k.offset), Quaternion(k.rotation).normalized())
              for k in clip.keys if k.bone == bone and clip.start <= k.frame <= clip.end]
    if not any(f==clip.start for f,_,_ in points):
        points.append((clip.start, Vector(), Quaternion()))
    if not any(f==clip.end for f,_,_ in points):
        points.append((clip.end, Vector(), Quaternion()))
    points.sort(key=lambda a: a[0])
    for a, b in zip(points, points[1:]):
        if a[0] <= frame <= b[0]:
            t = smooth((frame-a[0])/max(1, b[0]-a[0]))
            return a[1].lerp(b[1], t), a[2].slerp(b[2], t)
    return Vector((0, 0, 0)), Quaternion()


def near_key_at(settings, bone, frame):
    return next((k for k in settings.near_keys if k.bone == bone and k.frame == frame), None)


def near_value(settings, bone, frame):
    """Interpolate nearby edit targets, keeping each anchor exact.

    Overlapping neighborhoods share a single interpolation; they never sum
    offsets, nor multiply one another's strength. Disjoint edits fade to zero.
    Returns a target and opacity over any existing range correction.
    """
    points = sorted((k for k in settings.near_keys if k.bone == bone), key=lambda k:k.frame)
    left = next((k for k in reversed(points) if k.frame <= frame), None)
    right = next((k for k in points if k.frame >= frame), None)
    if left and left.frame == frame:
        return Vector(left.offset), Quaternion(left.rotation).normalized(), 1.0
    if left and right and left.frame+left.radius >= right.frame-right.radius:
        t = smooth((frame-left.frame)/(right.frame-left.frame))
        return (Vector(left.offset).lerp(Vector(right.offset), t),
                Quaternion(left.rotation).normalized().slerp(Quaternion(right.rotation).normalized(), t), 1.0)
    if left and frame < left.frame+left.radius:
        return Vector(left.offset), Quaternion(left.rotation).normalized(), smooth(1-(frame-left.frame)/left.radius)
    if right and frame > right.frame-right.radius:
        return Vector(right.offset), Quaternion(right.rotation).normalized(), smooth(1-(right.frame-frame)/right.radius)
    return Vector(), Quaternion(), 0.0


def put_near_key(settings, bone, frame, offset, rotation):
    if not editable(bone):
        raise ValueError('请选择双臂、手掌或手指')
    key = near_key_at(settings, bone, frame)
    if key is None:
        key = settings.near_keys.add()
    key.bone, key.frame = bone, frame
    key.offset, key.rotation = offset, rotation
    key.radius = settings.near_radius
    return key


def commit_near(scene, bone, frame, offset, rotation):
    """Record only on confirmation, rolling metadata back on bake failure."""
    p = scene.motion_tweak
    validate(p)
    key = near_key_at(p, bone, frame)
    saved = (tuple(key.offset), tuple(key.rotation), key.radius) if key else None
    key = put_near_key(p, bone, frame, offset, rotation)
    try:
        bake(scene)
    except Exception:
        if saved:
            key.offset, key.rotation, key.radius = saved
        else:
            p.near_keys.remove(len(p.near_keys)-1)
        raise
    p.status = part_label(bone)+'：修正 '+str(frame)+' 帧，自动影响前后各 '+str(key.radius)+' 帧'


def put_key(clip, bone, frame, offset, rotation):
    if not clip.start <= frame <= clip.end:
        raise ValueError('请在起止帧范围内调整，起始帧和结束帧均可编辑')
    hold = hold_for(clip, bone)
    if hold:
        # Dragging in a fade edits the full-strength value, so the visible pose
        # at the edit frame does not change on confirmation.
        if frame==clip.start:hold.edge_start=True
        if frame==clip.end:hold.edge_end=True
        w = hold_weight(clip, frame, hold)
        q = rotation.normalized()
        if q.w < 0:
            q = -q
        hold.frame = frame
        hold.offset = Vector(offset)/w
        hold.rotation = Quaternion(q.axis, min(math.pi, q.angle/w)) if q.angle > 1e-8 else Quaternion()
        return hold
    key = next((k for k in clip.keys if k.bone == bone and k.frame == frame), None)
    if key is None:
        key = clip.keys.add()
    key.bone, key.frame = bone, frame
    key.offset, key.rotation = offset, rotation
    return key


def commit_range(scene, clip, bone, frame, offset, rotation):
    hold=hold_for(clip,bone)
    collection=clip.holds if hold else clip.keys
    old=hold or next((k for k in clip.keys if k.bone==bone and k.frame==frame),None)
    saved=(old.frame,tuple(old.offset),tuple(old.rotation),old.edge_start,old.edge_end) if old else None
    key=put_key(clip,bone,frame,offset,rotation)
    try:bake(scene)
    except Exception:
        if saved:key.frame,key.offset,key.rotation,key.edge_start,key.edge_end=saved
        else:collection.remove(len(collection)-1)
        raise


def source_pose(scene):
    p = scene.motion_tweak
    dg = bpy.context.evaluated_depsgraph_get()
    ev = p.evaluator.evaluated_get(dg)
    return {b.name: b.matrix_basis.copy() for b in ev.pose.bones}, {b.name: b.matrix.copy() for b in ev.pose.bones}


def limit_rotation(q, degrees):
    q = q.normalized()
    if q.w < 0:
        q = -q
    return Quaternion(q.axis, min(q.angle, math.radians(degrees))) if q.angle > 1e-8 else Quaternion()


def solve_pose(rig, bases, source, edits):
    """Rigid two-segment IK using source bend plane and immutable local capture.

    Twist intermediates are included through convert_local_to_pose. Only local
    rotation is written: no joint translation, no scale, no geometry stretching.
    """
    bases = {n: m.copy() for n, m in bases.items()}
    poses = {n: m.copy() for n, m in source.items()}
    changed = set()
    warnings = []

    def propagate(name):
        for child in rig.data.bones[name].children:
            # The editable chains have no pose constraints on their ancestors.
            # Leave driven twist-distribution branches for Blender to evaluate.
            if rig.pose.bones[child.name].constraints:
                continue
            poses[child.name] = child.convert_local_to_pose(
                bases[child.name], child.matrix_local,
                parent_matrix=poses[name], parent_matrix_local=child.parent.matrix_local)
            propagate(child.name)

    def orient(name, q):
        bone = rig.data.bones[name]
        loc = poses[name].translation.copy()
        desired = Matrix.LocRotScale(loc, q.normalized(), poses[name].to_scale())
        kwargs = {'parent_matrix': poses[bone.parent.name],
                  'parent_matrix_local': bone.parent.matrix_local} if bone.parent else {}
        basis = bone.convert_local_to_pose(desired, bone.matrix_local, invert=True, **kwargs)
        # Keep exact source joint origins/scales, replacing only local rotation.
        bases[name] = Matrix.LocRotScale(bases[name].translation, basis.to_quaternion(), bases[name].to_scale())
        poses[name] = bone.convert_local_to_pose(bases[name], bone.matrix_local, **kwargs)
        changed.add(name)
        propagate(name)

    def swing(name, start, end, target):
        old, new = end-start, target-start
        if old.length > 1e-7 and new.length > 1e-7:
            orient(name, old.rotation_difference(new) @ poses[name].to_quaternion())

    def hand_move(side, delta):
        upper, fore, wrist = 'Arm_'+side, 'Elbow_'+side, 'Wrist_'+side
        a, b, c = [poses[n].translation.copy() for n in (upper, fore, wrist)]
        l1, l2 = (b-a).length, (c-b).length
        original_hand = poses[wrist].to_quaternion()
        direction = c + delta - a
        if direction.length < 1e-7:
            direction = c-a
        # Elbow flexion stays between 6 and 150 degrees; shoulder stays fixed.
        lo = math.sqrt(l1*l1+l2*l2+2*l1*l2*math.cos(math.radians(150)))
        hi = math.sqrt(l1*l1+l2*l2+2*l1*l2*math.cos(math.radians(6)))
        d = max(lo, min(hi, direction.length))
        if abs(d-direction.length) > 0.002:
            warnings.append('目标超过手臂可达范围，已保持骨长并限制伸展')
        axis = direction.normalized()
        old_axis = (c-a).normalized()
        transported = old_axis.rotation_difference(axis) @ (b-a)
        pole = transported-axis*transported.dot(axis)
        if pole.length < 1e-5:
            hint = Vector((0, -1, 0))
            pole = hint-axis*hint.dot(axis)
            if pole.length < 1e-5:
                pole = Vector((1 if side == 'L' else -1, 0, 0))
                pole -= axis*pole.dot(axis)
        pole.normalize()
        x = (l1*l1-l2*l2+d*d)/(2*d)
        elbow = a+axis*x+pole*math.sqrt(max(0, l1*l1-x*x))
        target = a+axis*d
        swing(upper, a, b, elbow)
        b2, c2 = poses[fore].translation.copy(), poses[wrist].translation.copy()
        swing(fore, b2, c2, target)
        orient(wrist, original_hand)

    # Parents before children, irrespective of the order in which edits were made.
    order = sorted(edits, key=lambda n: len(rig.data.bones[n].parent_recursive))
    for name in order:
        offset, rotation = edits[name]
        if offset.length > 1e-8:
            if name.startswith(('Wrist_', 'Elbow_')):
                hand_move(name[-1], offset)
            else:
                m = poses[name]
                start = m.translation.copy()
                end = m @ Vector((0, rig.data.bones[name].length, 0))
                swing(name, start, end, end+offset)
        if rotation.angle > 1e-7:
            limit = 60 if name.startswith('Wrist_') else 85
            rotation = limit_rotation(rotation, limit)
            desired = rotation @ poses[name].to_quaternion()
            if name.startswith('Elbow_'):
                # R may change elbow flexion, but must not turn it into a ball
                # joint or reverse the capture's anatomical bend direction.
                side=name[-1]
                a,b,c=[poses[n].translation.copy() for n in ('Arm_'+side,name,'Wrist_'+side)]
                axis=(b-a).normalized()
                bend=(c-b)-axis*(c-b).dot(axis)
                if bend.length<1e-5:
                    bend=Vector((0,-1,0));bend-=axis*bend.dot(axis)
                if bend.length<1e-5:
                    bend=Vector((1,0,0));bend-=axis*bend.dot(axis)
                bend.normalize()
                forward=desired@Vector((0,1,0))
                flex=math.acos(max(-1,min(1,forward.dot(axis))))
                flex=max(math.radians(6),min(math.radians(150),flex))
                safe=axis*math.cos(flex)+bend*math.sin(flex)
                desired=forward.rotation_difference(safe)@desired
            orient(name, desired)
            if name.startswith('Wrist_') or 'Finger' in name or name.startswith('Thumb'):
                # Also bound the resulting joint, rather than just the edit.
                local=bases[name].to_quaternion()
                capped=limit_rotation(local,90 if name.startswith('Wrist_') else 115)
                if abs(local.dot(capped))<1-1e-7:
                    bone=rig.data.bones[name]
                    basis=Matrix.LocRotScale(bases[name].translation,capped,bases[name].to_scale())
                    q=bone.convert_local_to_pose(basis,bone.matrix_local,
                        parent_matrix=poses[bone.parent.name],parent_matrix_local=bone.parent.matrix_local).to_quaternion()
                    orient(name,q)
                    warnings.append('已限制手腕或手指的过度弯曲')
    return bases, poses, changed, warnings


def edits_at(scene, frame, override=None, range_override=None):
    edits = {}
    for clip in scene.motion_tweak.clips:
        if not clip.enabled or not clip.start <= frame <= clip.end:
            continue
        for name in {k.bone for k in clip.keys} | {k.bone for k in clip.holds}:
            value = value_at(clip, name, frame)
            if value[0].length > 1e-10 or value[1].angle > 1e-7:
                edits[name] = value
    if range_override:
        edits[range_override[0]] = range_override[1:]
    for name in {k.bone for k in scene.motion_tweak.near_keys}:
        target, rotation, weight = near_value(scene.motion_tweak, name, frame)
        if weight > 0:
            base, q = edits.get(name, (Vector(), Quaternion()))
            edits[name] = base.lerp(target, weight), q.slerp(rotation, weight)
    if override:
        edits[override[0]] = override[1:]
    return edits


def curve_bag(action, slot_id):
    slot = next(s for s in action.slots if s.identifier == slot_id)
    return next(b for l in action.layers for s in l.strips for b in s.channelbags if b.slot_handle == slot.handle)


def bake(scene):
    """Build transactionally from source, replacing only adjusted rotations."""
    p = scene.motion_tweak
    rig = validate(p)
    intervals = sorted((c.start, c.end) for c in p.clips if c.enabled and (c.keys or c.holds))
    for c in p.clips:
        if c.end-c.start < 2:
            raise ValueError('帧段至少需要 3 帧')
        if any(not c.start <= k.frame <= c.end for k in c.keys):
            raise ValueError('有微调关键帧落在帧段边界外，请先调整帧段或删除该关键帧')
    # Shared neutral endpoints remain compatible; conflicting boundary targets
    # must be explicit errors rather than depend on collection order.
    enabled = [c for c in p.clips if c.enabled and (c.keys or c.holds)]
    for i,a in enumerate(enabled):
        for b in enabled[i+1:]:
            sides_a={k.bone[-1] for k in list(a.keys)+list(a.holds)}
            sides_b={k.bone[-1] for k in list(b.keys)+list(b.holds)}
            if max(a.start,b.start)<min(a.end,b.end) and sides_a&sides_b:
                raise ValueError('同一只手的调整区间重叠，请修改起止帧；左右手可以同时调整')
            for frame in {a.start,a.end} & {b.start,b.end}:
                names=({k.bone for k in a.keys}|{k.bone for k in a.holds}) & ({k.bone for k in b.keys}|{k.bone for k in b.holds})
                for name in names:
                    av,aq=value_at(a,name,frame);bv,bq=value_at(b,name,frame)
                    if (av.length>1e-10 or aq.angle>1e-7) and (bv.length>1e-10 or bq.angle>1e-7):
                        raise ValueError('相邻片段在同一边界帧修改了同一部位，请合并片段')
    for c in enabled:
        if any(k.frame==c.start for k in c.keys) or any(k.edge_start for k in c.holds):
            intervals.append((c.start-1,c.start-1))
        if any(k.frame==c.end for k in c.keys) or any(k.edge_end for k in c.holds):
            intervals.append((c.end+1,c.end+1))
    intervals += [(k.frame-k.radius, k.frame+k.radius) for k in p.near_keys]
    old = rig.animation_data.action
    original_frame = scene.frame_current
    physics = getattr(scene, 'wiggle_enable', None)
    result = p.source.copy()
    result.use_fake_user = False
    result.name = p.source.name + ' · 微调'
    result['motion_tweak'] = True
    try:
        if physics is not None:
            scene.wiggle_enable = False
        samples = {}
        touched = set()
        frames = sorted({f for a,b in intervals for f in range(a,b+1)})
        for frame in frames:
            scene.frame_set(frame)
            bases, source = source_pose(scene)
            solved, poses, changed, warnings = solve_pose(rig, bases, source, edits_at(scene, frame))
            touched.update(changed)
            samples[frame] = solved
        bag = curve_bag(result, p.slot_id)
        for name in touched:
            path = rig.pose.bones[name].path_from_id()+'.rotation_quaternion'
            existing = [fc for fc in bag.fcurves if fc.data_path == path]
            if rig.pose.bones[name].rotation_mode != 'QUATERNION':
                raise ValueError('第一版要求可编辑骨骼使用四元数旋转：'+name)
            if any(fc.modifiers for fc in existing):
                raise ValueError('请先烘焙带 FCurve 修改器的骨骼：'+name)
            channel = {fc.array_index: fc for fc in existing}
            source_fc = {fc.array_index:fc for fc in curves(p.source, p.evaluator.animation_data.action_slot_handle) if fc.data_path==path}
            retained = {}
            for i in range(4):
                fc = channel.get(i)
                if fc is None:
                    fc = bag.fcurves.new(path, index=i, group_name=PREFIX+part_label(name))
                retained[i] = [(pt.co.x, tuple(pt.handle_left), tuple(pt.handle_right),
                                 pt.handle_left_type, pt.handle_right_type) for pt in fc.keyframe_points
                               if pt.co.x not in samples]
                for pt in list(fc.keyframe_points)[::-1]:
                    if pt.co.x in samples:
                        fc.keyframe_points.remove(pt, fast=True)
                channel[i] = fc
            previous = None
            for frame in frames:
                q = samples[frame][name].to_quaternion().normalized()
                # Match source hemisphere at each interval boundary, then keep continuity.
                if frame-1 not in samples:
                    previous = None
                ref = previous if previous else Quaternion([source_fc[i].evaluate(frame) for i in range(4)]) if len(source_fc)==4 else q
                if q.dot(ref) < 0:
                    q = -q
                previous = q
                for i in range(4):
                    pt = channel[i].keyframe_points.insert(frame, q[i], options={'FAST'})
                    pt.interpolation = 'LINEAR'
            for i,fc in channel.items():
                fc.update()
                points = {pt.co.x:pt for pt in fc.keyframe_points}
                for frame,left,right,lt,rt in retained[i]:
                    pt = points[frame]
                    pt.handle_left_type,pt.handle_right_type = 'FREE','FREE'
                    pt.handle_left,pt.handle_right = left,right
        assign_action(rig, result, p.slot_id)
        p.result = result
        p.show_original = False
        p.status = '已生成独立微调动作；Ctrl+F12 使用同一动作，无需运行编辑器'
    except Exception:
        assign_action(rig, old, p.slot_id)
        bpy.data.actions.remove(result)
        raise
    finally:
        if physics is not None:
            scene.wiggle_enable = physics
        scene.frame_set(original_frame)
    if old != p.source and old.get('motion_tweak') and old.users == 0:
        bpy.data.actions.remove(old)
    redraw()
    return result


class MT_OT_Setup(bpy.types.Operator):
    bl_idname = 'motion_tweak.setup'
    bl_label = '建立独立微调层'
    bl_options = {'REGISTER','UNDO'}
    def execute(self, context):
        try:
            rig = context.scene.motion_tweak.rig
            if rig is None:
                ob = context.object
                rig = ob if ob and ob.type=='ARMATURE' else ob.find_armature() if ob and ob.type=='MESH' else None
            setup(context.scene,rig)
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'},str(e));return {'CANCELLED'}


class MT_OT_AddClip(bpy.types.Operator):
    bl_idname = 'motion_tweak.add_clip'
    bl_label = '添加调整帧段'
    bl_options = {'REGISTER','UNDO'}
    def execute(self,context):
        p=context.scene.motion_tweak
        c=p.clips.add()
        c.name='微调 '+str(len(p.clips))
        s=context.scene
        c.start=s.frame_preview_start if s.use_preview_range else s.frame_start
        c.end=s.frame_preview_end if s.use_preview_range else s.frame_end
        p.active_clip=len(p.clips)-1
        return {'FINISHED'}


class MT_OT_Bake(bpy.types.Operator):
    bl_idname='motion_tweak.apply'
    bl_label='更新微调结果'
    bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        try:bake(context.scene);return {'FINISHED'}
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}


def apply_whole_range(scene):
    p = scene.motion_tweak
    validate(p)
    clip = current_clip(scene)
    bone, frame = p.selected_bone, scene.frame_current
    if p.show_original:
        raise ValueError('请先切回微调结果')
    if not clip or not clip.enabled:
        raise ValueError('请先添加并启用调整帧段')
    if not editable(bone) or bone not in p.rig.pose.bones:
        raise ValueError('请先点击要应用到整段的部位')
    if not clip.start <= frame <= clip.end:
        raise ValueError('请先将时间轴移到起止帧范围内（含起止帧）')
    existing = hold_for(clip, bone)
    if existing:
        # Clicking again after changing fades must not attenuate the stored edit.
        bake(scene)
        return
    if not any(k.bone == bone for k in clip.keys):
        raise ValueError('请先对所选部位用 G / R 调整并确认，再应用到整段')
    offset, rotation = value_at(clip, bone, frame)
    if offset.length < 1e-8 and rotation.angle < 1e-7:
        raise ValueError('当前帧没有有效调整量，请移到已调整的帧')
    hold = clip.holds.add()
    hold.bone, hold.frame = bone, frame
    hold.offset, hold.rotation = offset, rotation
    hold.edge_start = frame==clip.start or any(k.bone==bone and k.frame==clip.start for k in clip.keys)
    hold.edge_end = frame==clip.end or any(k.bone==bone and k.frame==clip.end for k in clip.keys)
    try:
        bake(scene)
    except Exception:
        clip.holds.remove(len(clip.holds)-1)
        raise
    p.status = part_label(bone)+'：已应用到 '+str(clip.start)+'—'+str(clip.end)+' 帧，两端平滑衔接'


class MT_OT_WholeRange(bpy.types.Operator):
    bl_idname='motion_tweak.whole_range'
    bl_label='应用到整段'
    bl_description='将当前所选部位的当前帧调整量应用到起止帧范围；原动作继续播放，两端按渐入渐出衔接'
    bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        try:
            apply_whole_range(context.scene)
            self.report({'INFO'},context.scene.motion_tweak.status)
            return {'FINISHED'}
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}


class MT_OT_LocalRange(bpy.types.Operator):
    bl_idname='motion_tweak.local_range'
    bl_label='取消整段应用'
    bl_description='仅取消所选部位的整段应用，恢复它之前的微调关键帧'
    bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        p=context.scene.motion_tweak;c=current_clip(context.scene)
        if not c:return {'CANCELLED'}
        for i,hold in enumerate(c.holds):
            if hold.bone!=p.selected_bone:continue
            saved=(hold.bone,hold.frame,tuple(hold.offset),tuple(hold.rotation),hold.edge_start,hold.edge_end)
            c.holds.remove(i)
            try:
                bake(context.scene)
            except Exception as e:
                hold=c.holds.add()
                hold.bone,hold.frame,hold.offset,hold.rotation,hold.edge_start,hold.edge_end=saved
                self.report({'ERROR'},str(e));return {'CANCELLED'}
            p.status=part_label(p.selected_bone)+'：已取消整段应用，保留原有调整记录'
            return {'FINISHED'}
        return {'CANCELLED'}


class MT_OT_Compare(bpy.types.Operator):
    bl_idname='motion_tweak.compare'
    bl_label='原动作 / 微调结果'
    bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        p=context.scene.motion_tweak
        try:
            rig=validate(p)
            p.show_original=not p.show_original
            assign_action(rig,p.source if p.show_original else p.result,p.slot_id)
            context.scene.frame_set(context.scene.frame_current)
            return {'FINISHED'}
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}


class MT_OT_Remove(bpy.types.Operator):
    bl_idname='motion_tweak.remove'
    bl_label='删除当前调整片段'
    bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        p=context.scene.motion_tweak
        try:validate(p)
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
        if current_clip(context.scene):
            p.clips.remove(p.active_clip)
            p.active_clip=min(p.active_clip,len(p.clips)-1)
            try:bake(context.scene)
            except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
        return {'FINISHED'}


class MT_OT_DeleteKey(bpy.types.Operator):
    bl_idname='motion_tweak.delete_key'
    bl_label='删除此微调关键帧'
    bl_options={'REGISTER','UNDO'}
    index:IntProperty()
    def execute(self,context):
        try:validate(context.scene.motion_tweak)
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
        c=current_clip(context.scene)
        if c and 0<=self.index<len(c.keys):
            c.keys.remove(self.index)
            try:bake(context.scene)
            except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
        return {'FINISHED'}


class MT_OT_Jump(bpy.types.Operator):
    bl_idname='motion_tweak.jump'
    bl_label='跳到微调关键帧'
    frame:IntProperty()
    def execute(self,context):
        context.scene.frame_set(self.frame);return {'FINISHED'}


class MT_OT_DeleteNear(bpy.types.Operator):
    bl_idname='motion_tweak.delete_near'
    bl_label='删除这次附近修正'
    bl_options={'REGISTER','UNDO'}
    index:IntProperty()
    def execute(self,context):
        p=context.scene.motion_tweak
        if not 0<=self.index<len(p.near_keys):return {'CANCELLED'}
        k=p.near_keys[self.index]
        saved=(k.bone,k.frame,tuple(k.offset),tuple(k.rotation),k.radius)
        p.near_keys.remove(self.index)
        try:bake(context.scene)
        except Exception as e:
            k=p.near_keys.add()
            k.bone,k.frame,k.offset,k.rotation,k.radius=saved
            p.near_keys.move(len(p.near_keys)-1,self.index)
            self.report({'ERROR'},str(e));return {'CANCELLED'}
        p.status='已删除这次附近修正，其他调整保留'
        return {'FINISHED'}


class MT_OT_Preview(bpy.types.Operator):
    bl_idname='motion_tweak.preview'
    bl_label='循环预览此帧段'
    def execute(self,context):
        p=context.scene.motion_tweak;s=context.scene
        if p.edit_mode=='NEAR':
            key=near_key_at(p,p.selected_bone,s.frame_current)
            radius=key.radius if key else p.near_radius
            s.use_preview_range=True
            s.frame_preview_start=max(s.frame_start,s.frame_current-radius-4)
            s.frame_preview_end=min(s.frame_end,s.frame_current+radius+4)
            s.frame_set(s.frame_preview_start)
            bpy.ops.screen.animation_play()
            return {'FINISHED'}
        c=current_clip(context.scene)
        if c:
            s=context.scene;s.use_preview_range=True
            s.frame_preview_start=max(s.frame_start,c.start-8)
            s.frame_preview_end=min(s.frame_end,c.end+8)
            s.frame_set(c.start)
            bpy.ops.screen.animation_play()
        return {'FINISHED'}


class MT_OT_Clear(bpy.types.Operator):
    bl_idname='motion_tweak.clear'
    bl_label='恢复原动作并移除微调层'
    bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        p=context.scene.motion_tweak
        try:
            rig=validate(p)
            assign_action(rig,p.source,p.slot_id)
            helper=p.evaluator
            p.evaluator=None
            if helper:
                collections=list(helper.users_collection);data=helper.data
                bpy.data.objects.remove(helper,do_unlink=True)
                if data.users==0:bpy.data.armatures.remove(data)
                for collection in collections:
                    if not collection.objects:bpy.data.collections.remove(collection)
            p.source=None;p.result=None;p.clips.clear();p.near_keys.clear();p.active_clip=-1
            p.editing=False;p.selected_bone='';p.show_original=False
            p.status='已恢复原动捕'
            context.scene.frame_set(context.scene.frame_current)
            return {'FINISHED'}
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}


def picking_enabled(context):
    p=context.scene.motion_tweak
    return context.area and context.area.type=='VIEW_3D' and context.mode=='OBJECT' and p.editing and p.source and not binding_changed(p) and not _drag


class MT_OT_Pick(bpy.types.Operator):
    bl_idname='motion_tweak.pick'
    bl_label='点击部位'
    @classmethod
    def poll(cls,context):return picking_enabled(context)
    def invoke(self,context,event):
        if context.screen.is_animation_playing:
            return {'PASS_THROUGH'}
        p=context.scene.motion_tweak
        mouse=(event.mouse_region_x,event.mouse_region_y)
        origin=view3d_utils.region_2d_to_origin_3d(context.region,context.region_data,mouse)
        direction=view3d_utils.region_2d_to_vector_3d(context.region,context.region_data,mouse)
        hit,point,normal,index,ob,matrix=context.scene.ray_cast(context.evaluated_depsgraph_get(),origin,direction)
        if not hit or ob.type!='MESH' or ob.original.find_armature()!=p.rig:
            return {'PASS_THROUGH'}
        evaluated=ob.evaluated_get(context.evaluated_depsgraph_get())
        mesh=evaluated.to_mesh()
        try:
            if index<0 or index>=len(mesh.polygons):return {'PASS_THROUGH'}
            polygon=mesh.polygons[index]
            local=matrix.inverted()@point
            weights={}
            groups={g.index:g.name for g in ob.original.vertex_groups}
            for i in polygon.vertices:
                v=mesh.vertices[i]
                proximity=1/max(0.0001,(v.co-local).length)**2
                for g in v.groups:
                    name=groups.get(g.group,'')
                    # Twist skin weights belong to their upper / lower arm.
                    if name.startswith(('ArmTwist','_dummy_ArmTwist','_shadow_ArmTwist')):name='Arm_'+name[-1]
                    if name.startswith(('HandTwist','_dummy_HandTwist','_shadow_HandTwist')):name='Elbow_'+name[-1]
                    if editable(name):weights[name]=weights.get(name,0)+g.weight*proximity
            if not weights:
                self.report({'INFO'},'第一版支持双臂、手掌和手指，请点击这些部位')
                return {'FINISHED'}
            p.selected_bone=max(weights,key=weights.get)
            if not p.advanced_ui:
                from motion_tweak_sections import ensure_part
                clip=current_clip(context.scene)
                if clip and clip.simple_section and p.selected_bone.endswith('_'+clip.side):ensure_part(clip,p.selected_bone)
            p.status='已选 '+part_label(p.selected_bone)+'；G 移动，R 旋转'
            redraw()
            return {'FINISHED'}
        finally:evaluated.to_mesh_clear()


class MT_OT_Transform(bpy.types.Operator):
    bl_idname='motion_tweak.transform'
    bl_label='微调部位'
    bl_options={'REGISTER','UNDO','BLOCKING','GRAB_CURSOR'}
    rotate:BoolProperty(default=False)
    @classmethod
    def poll(cls,context):return picking_enabled(context) and bool(context.scene.motion_tweak.selected_bone)
    def invoke(self,context,event):
        global _drag
        p=context.scene.motion_tweak
        try:
            validate(p)
            if p.show_original:raise ValueError('请先切回微调结果再编辑')
            if context.screen.is_animation_playing:raise ValueError('请先暂停播放再调整')
            self.mode=p.edit_mode
            self.clip=current_clip(context.scene)
            if not p.advanced_ui:
                self.mode='RANGE'
                if not self.clip or not self.clip.simple_section:raise ValueError('请先选择一个动作区间')
                if not p.selected_bone.endswith('_'+self.clip.side):raise ValueError('所选部位与动作段的左右手不一致，请选择对应动作段')
                if context.scene.frame_current in (self.clip.start,self.clip.end):raise ValueError('边界用于平滑过渡，请在动作段中间调整')
            if self.mode=='RANGE':
                if not self.clip:raise ValueError('修整段：请先添加调整帧段')
                if not self.clip.enabled:raise ValueError('请先启用当前片段')
                if not self.clip.start<=context.scene.frame_current<=self.clip.end:raise ValueError('请将时间轴移到起止帧范围内（含起止帧）')
            self.bone=p.selected_bone;self.rig=p.rig
            self.region=next(region for region in context.area.regions if region.type=='WINDOW')
            self.rv3d=context.space_data.region_3d
            self.frame=context.scene.frame_current
            self.previous_action=self.rig.animation_data.action
            self.bases,self.source=source_pose(context.scene)
            self.offset,self.rotation=(edits_at(context.scene,self.frame).get(self.bone,(Vector(),Quaternion()))
                                       if self.mode=='NEAR' else value_at(self.clip,self.bone,self.frame))
            self.mouse=Vector((event.mouse_x-self.region.x,event.mouse_y-self.region.y))
            m=self.rig.matrix_world@self.source[self.bone]
            self.pivot=m.translation if self.bone.startswith('Wrist_') else m@Vector((0,self.rig.data.bones[self.bone].length,0))
            self.screen_pivot=view3d_utils.location_3d_to_region_2d(self.region,self.rv3d,self.pivot)
            if self.screen_pivot is None:raise ValueError('所选部位不在当前视图中')
            self.axis=None;self.latest=(self.offset.copy(),self.rotation.copy())
            self.preview_action=p.result.copy()
            self.preview_action.name=PREFIX+'临时预览'
            # Mute this frame's animated rotations while dragging; original remains intact.
            for fc in curves(self.preview_action,self.rig.animation_data.action_slot_handle):
                if '.rotation_quaternion' in fc.data_path and any('"'+n+'"' in fc.data_path for n in self.rig.pose.bones.keys() if editable(n)):
                    fc.mute=True
            assign_action(self.rig,self.preview_action,p.slot_id)
            _drag=self
            context.window_manager.modal_handler_add(self)
            self.update_preview(context,self.mouse,False,False)
            return {'RUNNING_MODAL'}
        except Exception as e:
            self.report({'ERROR'},str(e));return {'CANCELLED'}
    def restore_preview(self,context):
        p=context.scene.motion_tweak
        assign_action(self.rig,self.previous_action,p.slot_id)
        if self.preview_action.users==0:bpy.data.actions.remove(self.preview_action)
        context.scene.frame_set(self.frame)
    def update_preview(self,context,mouse,fine,snap):
        p=context.scene.motion_tweak
        inv=self.rig.matrix_world.inverted()
        if self.rotate:
            a=self.mouse-self.screen_pivot;b=mouse-self.screen_pivot
            angle=math.atan2(a.x*b.y-a.y*b.x,a.dot(b)) if a.length>3 and b.length>3 else (mouse.x-self.mouse.x)*0.01
            if fine:angle*=0.1
            if snap:angle=round(angle/math.radians(5))*math.radians(5)
            axis=Vector(self.axis) if self.axis else view3d_utils.region_2d_to_vector_3d(self.region,self.rv3d,self.screen_pivot)
            local=inv.to_quaternion()@axis
            rotation=limit_rotation(Quaternion(local,angle)@self.rotation,60 if self.bone.startswith('Wrist_') else 85)
            offset=self.offset
        else:
            initial=view3d_utils.region_2d_to_location_3d(self.region,self.rv3d,self.mouse,self.pivot)
            now=view3d_utils.region_2d_to_location_3d(self.region,self.rv3d,mouse,self.pivot)
            delta=now-initial
            if self.axis:
                axis=Vector(self.axis)
                if abs(axis.dot(view3d_utils.region_2d_to_vector_3d(self.region,self.rv3d,self.mouse)))>.95:
                    delta=axis*((mouse.x-self.mouse.x)*0.001)
                else:delta=axis*delta.dot(axis)
            if fine:delta*=0.1
            if snap:delta=Vector(round(v/.01)*.01 for v in delta)
            offset=self.offset+inv.to_3x3()@delta
            rotation=self.rotation
        self.latest=(offset.copy(),rotation.copy())
        edit=(self.bone,offset,rotation)
        edits=edits_at(context.scene,self.frame,edit) if self.mode=='NEAR' else edits_at(context.scene,self.frame,range_override=edit)
        solved,poses,changed,warnings=solve_pose(self.rig,self.bases,self.source,edits)
        for name in self.rig.pose.bones.keys():
            if editable(name):self.rig.pose.bones[name].matrix_basis=solved[name]
        context.view_layer.update()
        p.status=warnings[0] if warnings else part_label(self.bone)+' · '+('旋转' if self.rotate else '移动')
        axis='XYZ'[self.axis.index(1)] if self.axis else '视图平面'
        context.area.header_text_set(('R 旋转' if self.rotate else 'G 移动')+' · '+axis+' | X/Y/Z 限轴 · Shift 精细 · Ctrl 吸附 · 左键确认 · Esc 取消')
        redraw()
    def modal(self,context,event):
        global _drag
        if event.type in {'ESC','RIGHTMOUSE'} and event.value=='PRESS':
            self.restore_preview(context);_drag=None;context.area.header_text_set(None);redraw();return {'CANCELLED'}
        if event.type in {'LEFTMOUSE','RET','NUMPAD_ENTER'} and event.value=='PRESS':
            self.restore_preview(context);_drag=None;context.area.header_text_set(None)
            try:
                if self.mode=='NEAR':
                    commit_near(context.scene,self.bone,self.frame,*self.latest)
                elif not context.scene.motion_tweak.advanced_ui:
                    from motion_tweak_sections import commit_drag
                    commit_drag(context.scene,self.clip,self.bone,self.frame,*self.latest)
                else:
                    commit_range(context.scene,self.clip,self.bone,self.frame,*self.latest)
                return {'FINISHED'}
            except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
        if event.type in {'X','Y','Z'} and event.value=='PRESS':
            axes={'X':(1,0,0),'Y':(0,1,0),'Z':(0,0,1)}
            self.axis=None if self.axis==axes[event.type] else axes[event.type]
        if event.type in {'MOUSEMOVE','X','Y','Z','LEFT_SHIFT','RIGHT_SHIFT','LEFT_CTRL','RIGHT_CTRL'}:
            self.update_preview(context,Vector((event.mouse_x-self.region.x,event.mouse_y-self.region.y)),event.shift,event.ctrl)
        return {'RUNNING_MODAL'}


class MT_UL_Clips(bpy.types.UIList):
    def draw_item(self,context,layout,data,item,icon,active_data,active_propname,index):
        layout.prop(item,'enabled',text='')
        layout.prop(item,'name',text='',emboss=False)
        layout.label(text=f'{item.start}—{item.end}')


class MT_PT_Main(bpy.types.Panel):
    bl_label='动作微调 · 按动作段调整'
    bl_idname='MT_PT_Main'
    bl_space_type='VIEW_3D';bl_region_type='UI';bl_category='动作微调'
    def draw(self,context):
        l=self.layout;p=context.scene.motion_tweak
        if not p.advanced_ui:
            from motion_tweak_sections import draw
            draw(l,context)
            return
        l.prop(p,'advanced_ui')
        l.use_property_split=False
        if p.saved_layers and not binding_changed(p):
            box=l.box()
            box.label(text='已保留 '+str(len(p.saved_layers))+' 个旧微调层')
            for i,layer in enumerate(p.saved_layers):
                name=layer.source.name if layer.source else '动作已删除'
                box.operator('motion_tweak.previous',text='切回：'+name,icon='LOOP_BACK').index=i
        if not p.source:
            l.prop(p,'rig',text='角色骨架')
            l.operator('motion_tweak.setup',icon='DUPLICATE')
            l.label(text='保留原动捕，另建调整动作',icon='INFO')
            return
        l.label(text='角色：'+(p.rig.name if p.rig else '骨架已删除'),icon='ARMATURE_DATA')
        l.label(text='原动作：'+p.source.name)
        if binding_changed(p):
            box=l.box();box.alert=True
            box.label(text='已导入或切换到另一段动作',icon='INFO')
            action=current_action(p)
            box.label(text='当前：'+(action.name if action else '无动作'))
            box.label(text='旧微调记录仍然保留')
            box.operator('motion_tweak.use_current',icon='DUPLICATE')
            box.operator('motion_tweak.resume',icon='LOOP_BACK')
            return
        r=l.row();r.operator('motion_tweak.compare',text='正在看原动作 · 切回微调' if p.show_original else '对照原动作',icon='ARROW_LEFTRIGHT')
        l.prop(p,'editing',toggle=True,icon='RESTRICT_SELECT_OFF')
        l.row().prop(p,'edit_mode',expand=True)
        if p.edit_mode=='NEAR':
            l.label(text='停在要修的帧 → 点部位 → G / R')
            l.prop(p,'near_radius',text='前后各影响')
            l.label(text='自动平滑过渡，无需设置起止帧')
        else:
            l.label(text='设置范围 → G / R → 应用到整段')
        l.label(text='所选：'+(part_label(p.selected_bone) if p.selected_bone else '未选部位'))
        r=l.row(align=True)
        r.operator('motion_tweak.transform',text='G 移动',icon='EVENT_G').rotate=False
        r.operator('motion_tweak.transform',text='R 旋转',icon='EVENT_R').rotate=True
        l.separator()
        if p.edit_mode=='NEAR':
            box=l.box();box.label(text='附近修正记录')
            if not p.near_keys:box.label(text='G / R 确认后自动记录')
            for i,k in sorted(enumerate(p.near_keys),key=lambda item:(item[1].frame,item[1].bone)):
                r=box.row(align=True)
                r.operator('motion_tweak.jump',text=f'{k.frame} 帧 · {part_label(k.bone)} · ±{k.radius}').frame=k.frame
                r.operator('motion_tweak.delete_near',text='',icon='X').index=i
            l.operator('motion_tweak.preview',text='循环预览附近',icon='PLAY')
            if any(c.keys or c.holds for c in p.clips):
                l.label(text='已有帧段调整保留，可切到“修整段”查看')
            l.separator();l.label(text=p.status[:48],icon='INFO')
            l.operator('motion_tweak.clear',icon='LOOP_BACK')
            return
        l.template_list('MT_UL_Clips','',p,'clips',p,'active_clip',rows=2)
        r=l.row(align=True);r.operator('motion_tweak.add_clip',icon='ADD');r.operator('motion_tweak.remove',text='',icon='REMOVE')
        c=current_clip(context.scene)
        if c:
            r=l.row(align=True);r.prop(c,'start');r.prop(c,'end')
            box=l.box()
            box.label(text='想让这段都保持同样的修正？')
            box.label(text='先 G / R 调好并确认，再应用到整段')
            r=box.row();r.scale_y=1.3
            r.operator('motion_tweak.whole_range',icon='DUPLICATE')
            r=box.row(align=True);r.prop(c,'fade_in');r.prop(c,'fade_out')
            box.label(text='渐入渐出包含在起止帧内；中间保持调整量')
            hold=hold_for(c,p.selected_bone)
            if hold:
                box.label(text=part_label(p.selected_bone)+'：整段应用中',icon='CHECKMARK')
                if hold.edge_start or hold.edge_end:
                    box.label(text='手动调整的边界保留修正，不强制归零')
                box.operator('motion_tweak.local_range',icon='LOOP_BACK')
            else:
                box.label(text='当前部位：尚未应用到整段')
            for h in c.holds:
                if h.bone!=p.selected_bone:
                    box.label(text=part_label(h.bone)+'：整段应用中')
            if c.end<=c.start+1:l.label(text='帧段至少需要 3 帧',icon='ERROR')
            box=l.box();box.label(text='段内调整记录（整段应用时保留备份）')
            if not c.keys:box.label(text='暂无调整；起始帧和结束帧也可编辑')
            for i,k in sorted(enumerate(c.keys),key=lambda item:(item[1].frame,item[1].bone)):
                r=box.row(align=True)
                r.enabled=not bool(hold_for(c,k.bone))
                r.operator('motion_tweak.jump',text=f'{k.frame} 帧 · {part_label(k.bone)}').frame=k.frame
                r.operator('motion_tweak.delete_key',text='',icon='X').index=i
            l.operator('motion_tweak.preview',icon='PLAY')
        l.operator('motion_tweak.apply',icon='FILE_REFRESH')
        l.label(text='改范围、渐入渐出或开关后，点击更新')
        l.label(text='固定骨长 · 不改手指原动捕 · 可直接渲染')
        l.separator();l.label(text=p.status[:48],icon='INFO')
        l.operator('motion_tweak.clear',icon='LOOP_BACK')


def draw_selection():
    context=bpy.context
    if not context.scene or not hasattr(context.scene,'motion_tweak'):return
    p=context.scene.motion_tweak
    if not p.editing or not p.rig or not p.selected_bone:return
    try:
        import gpu
        from gpu_extras.batch import batch_for_shader
        rig=p.rig.evaluated_get(context.evaluated_depsgraph_get())
        b=rig.pose.bones.get(p.selected_bone)
        if not b:return
        points=[rig.matrix_world@b.head,rig.matrix_world@b.tail]
        shader=gpu.shader.from_builtin('UNIFORM_COLOR')
        gpu.state.depth_test_set('NONE');gpu.state.line_width_set(5)
        shader.bind();shader.uniform_float('color',(1,.62,.06,1))
        batch_for_shader(shader,'LINES',{'pos':points}).draw(shader)
        gpu.state.line_width_set(1);gpu.state.depth_test_set('NONE')
    except (ReferenceError,RuntimeError):pass


from motion_tweak_sections import CLASSES as SECTION_CLASSES

CLASSES=(MTKey,MTPart,MTClip,MTLayerBackup,MTSettings,MT_OT_Setup,MT_OT_UseCurrent,MT_OT_Resume,MT_OT_Previous,
         MT_OT_AddClip,MT_OT_Bake,MT_OT_Compare,
         MT_OT_WholeRange,MT_OT_LocalRange,
         MT_OT_Remove,MT_OT_DeleteKey,MT_OT_DeleteNear,MT_OT_Jump,MT_OT_Preview,MT_OT_Clear,
         MT_OT_Pick,MT_OT_Transform,MT_UL_Clips,*SECTION_CLASSES,MT_PT_Main)


def register():
    global _draw
    for cls in CLASSES:bpy.utils.register_class(cls)
    bpy.types.Scene.motion_tweak=PointerProperty(type=MTSettings)
    if not bpy.app.background:
        _draw=bpy.types.SpaceView3D.draw_handler_add(draw_selection,(),'WINDOW','POST_VIEW')
        kc=bpy.context.window_manager.keyconfigs.addon
        if kc:
            km=kc.keymaps.new(name='Object Mode',space_type='EMPTY')
            _keymaps.append((km,km.keymap_items.new('motion_tweak.pick','LEFTMOUSE','PRESS',head=True)))
            for key,rotate in [('G',False),('R',True)]:
                kmi=km.keymap_items.new('motion_tweak.transform',key,'PRESS',head=True)
                kmi.properties.rotate=rotate;_keymaps.append((km,kmi))


def unregister():
    global _draw
    for km,kmi in _keymaps:km.keymap_items.remove(kmi)
    _keymaps.clear()
    if _draw:bpy.types.SpaceView3D.draw_handler_remove(_draw,'WINDOW');_draw=None
    del bpy.types.Scene.motion_tweak
    for cls in reversed(CLASSES):bpy.utils.unregister_class(cls)


if __name__=='__main__':register()

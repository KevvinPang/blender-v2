"""Support native Camera to View navigation without overwriting capture motion.

The source retains its animation. Blender navigates its child using standard
viewport controls. Optional keys animate that child's local framing offset;
no timer or driver writes its transform.
"""
import bpy
import math
from bpy.props import EnumProperty, IntProperty
from mathutils import Matrix

ENABLED = 'MocapCameraDistanceEnabled'
VIEW = 'MocapCameraDistanceView'
SOURCE = 'mocap_distance_source'
DISTANCE = 'distance_cm'
NATIVE = 'mocap_native_camera_navigation'
_navigation_keymaps = []


def locked_ortho_camera(context):
    """Blender's normal wheel zooms the camera border for an orthographic shot."""
    space = context.space_data
    if (not space or space.type != 'VIEW_3D' or not space.lock_camera
            or not space.region_3d or space.region_3d.view_perspective != 'CAMERA'
            or space.region_quadviews):
        return None
    camera = space.camera if space.use_local_camera else context.scene.camera
    if (not camera or camera.type != 'CAMERA' or camera.data.type != 'ORTHO'
            or not camera.data.is_editable):
        return None
    return camera


class ELLEN_OT_camera_ortho_zoom(bpy.types.Operator):
    bl_idname = 'ellen_camera.ortho_zoom'
    bl_label = '正交摄像机远近'
    bl_description = '锁定摄像头到视图时调整实际画面缩放，滚轮或 Ctrl+中键拖动'
    bl_options = {'REGISTER', 'UNDO'}
    direction: IntProperty(default=0, min=-1, max=1, options={'SKIP_SAVE'})

    @classmethod
    def poll(cls, context):
        return locked_ortho_camera(context) is not None

    def execute(self, context):
        camera = locked_ortho_camera(context)
        camera.data.ortho_scale = max(.000001, camera.data.ortho_scale*math.exp(-self.direction*.08))
        context.area.tag_redraw()
        return {'FINISHED'}

    def invoke(self, context, event):
        if self.direction:
            return self.execute(context)
        self._camera = locked_ortho_camera(context)
        self._start_scale = self._camera.data.ortho_scale
        self._start_y = event.mouse_y
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type in {'ESC', 'RIGHTMOUSE'}:
            self._camera.data.ortho_scale = self._start_scale
            context.area.tag_redraw()
            return {'CANCELLED'}
        if event.type == 'MIDDLEMOUSE' and event.value == 'RELEASE':
            return {'FINISHED'}
        if event.type == 'MOUSEMOVE':
            exponent = max(-20., min(20., (self._start_y-event.mouse_y)*.005))
            self._camera.data.ortho_scale = max(.000001, self._start_scale*math.exp(exponent))
            context.area.tag_redraw()
        return {'RUNNING_MODAL'}


def register_navigation():
    """Only override zoom for a locked ORTHO camera; other views stay native."""
    if _navigation_keymaps:
        return
    config = bpy.context.window_manager.keyconfigs.addon
    if not config:
        return
    keymap = config.keymaps.new(name='3D View', space_type='VIEW_3D', region_type='WINDOW')
    for event, direction in (('WHEELUPMOUSE', 1), ('WHEELDOWNMOUSE', -1)):
        item = keymap.keymap_items.new('ellen_camera.ortho_zoom', event, 'PRESS', head=True)
        item.properties.direction = direction
        _navigation_keymaps.append((keymap, item))
    item = keymap.keymap_items.new('ellen_camera.ortho_zoom', 'MIDDLEMOUSE', 'PRESS', ctrl=True, head=True)
    item.properties.direction = 0
    _navigation_keymaps.append((keymap, item))


def unregister_navigation():
    for keymap, item in _navigation_keymaps:
        keymap.keymap_items.remove(item)
    _navigation_keymaps.clear()


def source_camera(camera):
    """Unwrap only the presentation layer, retaining calibration ancestry."""
    seen = set()
    while camera and camera.as_pointer() not in seen:
        seen.add(camera.as_pointer())
        source = camera.get(SOURCE)
        if not isinstance(source, bpy.types.Object) or source.type != 'CAMERA':
            break
        camera = source
    return camera


def adjustment_camera(scene):
    camera = scene.get(VIEW)
    return camera if isinstance(camera, bpy.types.Object) and camera.type == 'CAMERA' and camera.name in scene.objects else None


def ensure_camera(scene, camera=None):
    """Provide a freely navigable render camera over the animated shot."""
    active = camera or scene.camera
    source = source_camera(active)
    if source is None or source.type != 'CAMERA' or source.name not in scene.objects:
        raise ValueError('请先选择场景中的摄像机')
    view = adjustment_camera(scene)
    if view is None:
        view = bpy.data.objects.new('Camera · 视图调整', source.data)
        scene.collection.objects.link(view)
        view.matrix_basis = Matrix.Identity(4)
        scene[VIEW] = view
    if not view.get(NATIVE):
        # Retain the previous framing, including manual X/Y/rotation edits.
        # Remove only our legacy dolly driver, once. Never drive location again.
        data = view.animation_data
        legacy = next((fc for fc in data.drivers
                       if fc.data_path == 'location' and fc.array_index == 2
                       and any(v.type == 'SINGLE_PROP' and v.targets[0].id == view
                               and v.targets[0].data_path == '["distance_cm"]'
                               for v in fc.driver.variables)), None) if data else None
        if legacy:
            z = -float(view.get(DISTANCE, 0.0)) / (100.0 * max(abs(source.matrix_world.to_scale().z), 1e-8))
            view.driver_remove('location', 2)
            view.location.z = z
        if DISTANCE in view:
            del view[DISTANCE]
        view[NATIVE] = True
    # A generated V2 file may intentionally reuse the render camera itself
    # as the saved close-up view. In that case `source` and `view` are the
    # same datablock; assigning `view.parent = source` would create a
    # self-parenting cycle during Blender's load/save handlers.
    if view != source and (view.parent != source or view.get(SOURCE) != source):
        # Keep the user's local edit when switching to newly captured motion.
        view.parent = source
        view.matrix_parent_inverse = Matrix.Identity(4)
        view[SOURCE] = source
    # Otherwise Camera to View transforms the animated parent, and its
    # keyframes overwrite the navigation on the next frame.
    view.use_camera_lock_parent = False
    # Share camera data so animated focal length, lens shift and DOF remain live.
    view.data = source.data
    scene[ENABLED] = True
    scene.camera = view
    return view


def sync_camera(scene):
    # Also used from camera-view tests with lightweight stand-in scenes.
    if not hasattr(scene, 'get') or not scene.camera:
        return scene.camera
    # Character Director owns the complete transform of its recorded camera.
    # Reusing the old framing child would add the previous shot's local offset.
    if scene.camera.get('character_director_camera'):
        return scene.camera
    camera = source_camera(scene.camera)
    if (scene.get(ENABLED) or scene.camera.get(SOURCE)
            or camera.get('sam_framing_model') or camera.get('sam_original_camera')):
        return ensure_camera(scene)
    return scene.camera


def rotation_path(camera):
    if camera.rotation_mode == 'QUATERNION':
        return 'rotation_quaternion'
    if camera.rotation_mode == 'AXIS_ANGLE':
        return 'rotation_axis_angle'
    return 'rotation_euler'


def framing_curves(camera):
    ad = camera.animation_data if camera else None
    if not ad or not ad.action or not ad.action_slot:
        return []
    paths = {'location', rotation_path(camera)}
    return [fc for layer in ad.action.layers for strip in layer.strips
            for bag in strip.channelbags if bag.slot_handle == ad.action_slot.handle
            for fc in bag.fcurves if fc.data_path in paths]


def framing_frames(camera):
    return sorted({float(k.co.x) for fc in framing_curves(camera) for k in fc.keyframe_points})


def editable_camera(context):
    scene = context.scene
    camera = adjustment_camera(scene)
    space = context.space_data
    if (not camera or scene.camera != camera or camera.library or
            (context.screen and context.screen.is_animation_playing)):
        return None
    if space and space.type == 'VIEW_3D' and space.use_local_camera and space.camera != camera:
        return None
    return camera


def current_frame(scene):
    return scene.frame_current + scene.frame_subframe


TRANSITIONS = (
    ('BEZIER', '平滑', '平滑加减速；在当前关键帧之后生效'),
    ('LINEAR', '匀速', '修正位移匀速变化；在当前关键帧之后生效'),
    ('CONSTANT', '停留', '保持当前构图，直到下一个关键帧再切换'),
)


def key_at(camera, frame):
    return [k for fc in framing_curves(camera) for k in fc.keyframe_points
            if abs(k.co.x-frame) < 1e-4]


class ELLEN_OT_camera_framing_prepare(bpy.types.Operator):
    bl_idname = 'ellen_camera.framing_prepare'
    bl_label = '开始调整镜头'
    bl_description = '启用独立镜头修正层并锁定摄像头到视图；原动捕镜头保留'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (context.scene.camera is not None and context.space_data is not None
                and context.space_data.type == 'VIEW_3D'
                and not context.space_data.use_local_camera
                and not context.screen.is_animation_playing)

    def execute(self, context):
        try:
            camera = ensure_camera(context.scene)
        except ValueError as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        for area in context.screen.areas:
            for space in area.spaces:
                if space.type == 'VIEW_3D' and not space.use_local_camera:
                    space.camera = camera
            area.tag_redraw()
        context.space_data.lock_camera = True
        context.space_data.region_3d.view_perspective = 'CAMERA'
        return {'FINISHED'}


class ELLEN_OT_camera_framing_record(bpy.types.Operator):
    bl_idname = 'ellen_camera.framing_record'
    bl_label = '记录 / 更新当前帧'
    bl_description = '将当前镜头的位置、远近和角度记录到当前帧；再次点击会更新此帧'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return editable_camera(context) is not None

    def execute(self, context):
        camera = editable_camera(context)
        frame = current_frame(context.scene)
        ad = camera.animation_data
        had_action = bool(ad and ad.action)
        if ad and any(not t.mute for t in ad.nla_tracks):
            self.report({'ERROR'}, '镜头修正层存在 NLA 动画，请先退出 NLA 编辑并静音轨道')
            return {'CANCELLED'}
        if ad and ad.action and ad.action.users > 1:
            ad.action = ad.action.copy()
        existing = key_at(camera, frame)
        interpolation = existing[0].interpolation if existing else 'BEZIER'
        # Native view navigation can cross the Euler wrap boundary. Choose the
        # equivalent angles nearest the preceding key, keeping the same view.
        previous = [f for f in framing_frames(camera) if f < frame-1e-4]
        if previous and rotation_path(camera) == 'rotation_euler':
            reference = camera.rotation_euler.copy()
            for fc in framing_curves(camera):
                if fc.data_path == 'rotation_euler':
                    reference[fc.array_index] = fc.evaluate(previous[-1])
            camera.rotation_euler.make_compatible(reference)
        elif previous and rotation_path(camera) == 'rotation_quaternion':
            reference = camera.rotation_quaternion.copy()
            for fc in framing_curves(camera):
                if fc.data_path == 'rotation_quaternion':
                    reference[fc.array_index] = fc.evaluate(previous[-1])
            if camera.rotation_quaternion.dot(reference) < 0:
                camera.rotation_quaternion.negate()
        for path in ('location', rotation_path(camera)):
            camera.keyframe_insert(path, frame=frame, group='镜头构图修正')
        if not had_action:
            camera.animation_data.action.name = '镜头关键帧 / 构图修正'
        for k in key_at(camera, frame):
            k.interpolation = interpolation
            k.handle_left_type = k.handle_right_type = 'AUTO_CLAMPED'
        for fc in framing_curves(camera):
            fc.update()
        camera.update_tag()
        self.report({'INFO'}, '已记录第 %g 帧镜头' % frame)
        return {'FINISHED'}


class ELLEN_OT_camera_framing_delete(bpy.types.Operator):
    bl_idname = 'ellen_camera.framing_delete'
    bl_label = '删除当前帧'
    bl_description = '只删除当前帧的镜头修正关键帧；原动捕镜头不变'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        camera = editable_camera(context)
        return camera is not None and bool(key_at(camera, current_frame(context.scene)))

    def execute(self, context):
        camera = editable_camera(context)
        if camera.animation_data.action.users > 1:
            camera.animation_data.action = camera.animation_data.action.copy()
        frame = current_frame(context.scene)
        for path in ('location', rotation_path(camera)):
            camera.keyframe_delete(path, frame=frame)
        context.scene.frame_set(context.scene.frame_current, subframe=context.scene.frame_subframe)
        return {'FINISHED'}


class ELLEN_OT_camera_framing_jump(bpy.types.Operator):
    bl_idname = 'ellen_camera.framing_jump'
    bl_label = '跳转镜头关键帧'
    bl_description = '跳到上一处或下一处镜头修正关键帧'
    direction: EnumProperty(items=(('PREV', '上一帧', ''), ('NEXT', '下一帧', '')))

    @classmethod
    def poll(cls, context):
        camera = editable_camera(context)
        return camera is not None and bool(framing_frames(camera))

    def execute(self, context):
        frame = current_frame(context.scene)
        frames = framing_frames(editable_camera(context))
        choices = [f for f in frames if f < frame-1e-4] if self.direction == 'PREV' else [f for f in frames if f > frame+1e-4]
        if not choices:
            self.report({'INFO'}, '此方向没有镜头关键帧')
            return {'CANCELLED'}
        target = choices[-1] if self.direction == 'PREV' else choices[0]
        import math
        context.scene.frame_set(math.floor(target), subframe=target-math.floor(target))
        return {'FINISHED'}


class ELLEN_OT_camera_framing_transition(bpy.types.Operator):
    bl_idname = 'ellen_camera.framing_transition'
    bl_label = '镜头过渡'
    bl_description = '设置当前关键帧到下一处关键帧的过渡方式'
    bl_options = {'REGISTER', 'UNDO'}
    interpolation: EnumProperty(items=TRANSITIONS, default='BEZIER')

    @classmethod
    def poll(cls, context):
        camera = editable_camera(context)
        return camera is not None and bool(key_at(camera, current_frame(context.scene)))

    def execute(self, context):
        camera = editable_camera(context)
        if camera.animation_data.action.users > 1:
            camera.animation_data.action = camera.animation_data.action.copy()
        for k in key_at(camera, current_frame(context.scene)):
            k.interpolation = self.interpolation
            k.handle_left_type = k.handle_right_type = 'AUTO_CLAMPED'
        for fc in framing_curves(camera):
            fc.update()
        camera.update_tag()
        context.scene.frame_set(context.scene.frame_current, subframe=context.scene.frame_subframe)
        return {'FINISHED'}


class ELLEN_PT_camera_framing(bpy.types.Panel):
    bl_label = '镜头关键帧'
    bl_idname = 'ELLEN_PT_camera_framing'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'View'
    bl_parent_id = 'VIEW3D_PT_view3d_properties'
    bl_order = 1

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        if not scene.camera:
            layout.label(text='请先设置场景摄像机', icon='INFO')
            return
        if context.space_data.use_local_camera:
            layout.label(text='请取消上方的“局部摄像机”', icon='INFO')
            return
        camera = adjustment_camera(scene)
        if scene.camera != camera:
            layout.operator('ellen_camera.framing_prepare', icon='CAMERA_DATA')
            layout.label(text='启用后，调整构图再记录')
            return
        if not context.space_data.lock_camera or context.space_data.region_3d.view_perspective != 'CAMERA':
            layout.operator('ellen_camera.framing_prepare', text='进入摄像头视图并锁定', icon='VIEW_CAMERA')
        frames = framing_frames(camera)
        layout.label(text='第 %g 帧 · %d 处镜头关键帧' % (current_frame(scene), len(frames)))
        layout.operator('ellen_camera.framing_record', icon='KEY_HLT')
        row = layout.row(align=True)
        row.operator('ellen_camera.framing_jump', text='上一处', icon='PREV_KEYFRAME').direction = 'PREV'
        row.operator('ellen_camera.framing_jump', text='下一处', icon='NEXT_KEYFRAME').direction = 'NEXT'
        layout.operator('ellen_camera.framing_delete', icon='KEY_DEHLT')
        keys = key_at(camera, current_frame(scene))
        row = layout.row(align=True)
        for value, label, _ in TRANSITIONS:
            row.operator('ellen_camera.framing_transition', text=label,
                         depress=bool(keys and all(k.interpolation == value for k in keys))).interpolation = value
        layout.label(text='过渡作用于当前帧 → 下一关键帧')
        layout.label(text='保留动捕镜头移动，叠加构图修正')


CLASSES = (ELLEN_OT_camera_ortho_zoom, ELLEN_OT_camera_framing_prepare, ELLEN_OT_camera_framing_record,
           ELLEN_OT_camera_framing_delete, ELLEN_OT_camera_framing_jump,
           ELLEN_OT_camera_framing_transition, ELLEN_PT_camera_framing)

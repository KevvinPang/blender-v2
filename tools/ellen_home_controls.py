bl_info = {'name': 'Ellen Random Play Rooms', 'author': 'Local project', 'version': (1, 1, 0), 'blender': (5, 2, 0), 'category': 'Animation'}
import json
import bpy
from bpy.props import EnumProperty
from ellen_camera_distance import (CLASSES as CAMERA_DISTANCE_CLASSES, source_camera,
                                   register_navigation, unregister_navigation)

_last_time = None

def live_physics_step(scene, packet):
    """Advance the existing Wiggle solver without advancing the recording timeline."""
    global _last_time
    import ellen_physics
    if ellen_physics.manages(scene):
        if not scene.ellen_physics.enabled or not scene.ellen_physics.preview or packet.get('paused'):
            _last_time = None
            return False
        stamp = packet.get('time', 0.)
        dt = max(0., min(.05, stamp-_last_time)) if _last_time is not None else 0.
        _last_time = stamp
        return ellen_physics.preview_step(scene, dt)
    if not scene.wiggle_enable:
        _last_time = None
        return False
    import wiggle_2
    stamp = packet.get('time')
    if packet.get('paused'):
        _last_time = None
        return False
    if _last_time is None or (stamp is not None and (stamp < _last_time or stamp - _last_time > .5)):
        wiggle_2.build_list()
        bpy.ops.wiggle.reset()
    else:
        state = scene.wiggle
        preroll = state.is_preroll
        state.lastframe = scene.frame_current - 1
        state.is_preroll = True
        try:
            wiggle_2.wiggle_pre(scene)
            wiggle_2.wiggle_post(scene, bpy.context.evaluated_depsgraph_get())
        finally:
            state.is_preroll = preroll
            state.lastframe = scene.frame_current
    _last_time = stamp if stamp is not None else 0
    bpy.context.view_layer.update()
    return True

class ELLEN_OT_home_room(bpy.types.Operator):
    bl_idname = 'ellen_home.room'
    bl_label = 'Switch Room'
    bl_options = {'REGISTER', 'UNDO'}
    room: EnumProperty(items=(('SHOP', '录像店', ''), ('LOUNGE', '休息室', '')))
    def execute(self, context):
        global _last_time
        scene = context.scene
        room = json.loads(scene['EllenHomeRooms'])[self.room]
        root = scene.objects.get('Ellen Room Placement')
        camera = scene.objects.get(room['camera'])
        if not root or not camera:
            self.report({'ERROR'}, 'Room objects are missing')
            return {'CANCELLED'}
        root.location = room['location']
        root.rotation_euler = (0, 0, room['rotation'])
        previous_camera = scene.camera
        scene.camera = camera
        if scene.get('CharacterStudio') and hasattr(scene, 'sam_ellen'):
            for name in ('gaze_camera', 'screen_camera', 'punch_camera'):
                if getattr(scene.sam_ellen, name) in (None, previous_camera, source_camera(previous_camera)):
                    setattr(scene.sam_ellen, name, camera)
        scene['EllenHomeRoom'] = self.room
        from sam_ellen_camera_fit import sync_camera_views
        sync_camera_views(scene)
        context.view_layer.update()
        if scene.wiggle_enable:
            bpy.ops.wiggle.reset()
        _last_time = None
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces.active.region_3d.view_perspective = 'CAMERA'
        return {'FINISHED'}

class ELLEN_PT_home(bpy.types.Panel):
    bl_label = 'Random Play / 房间机位'
    bl_idname = 'ELLEN_PT_home'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SAM 动捕'
    bl_order = -10
    @classmethod
    def poll(cls, context):
        return bool(context.scene.get('EllenHomeProject'))
    def draw(self, context):
        s = context.scene
        row = self.layout.row(align=True)
        row.operator('ellen_home.room', text='录像店', depress=s.get('EllenHomeRoom') == 'SHOP', icon='CAMERA_DATA').room = 'SHOP'
        row.operator('ellen_home.room', text='休息室', depress=s.get('EllenHomeRoom') == 'LOUNGE', icon='CAMERA_DATA').room = 'LOUNGE'
        self.layout.prop(s, 'camera', text='摄像机')
        if hasattr(s, 'ellen_physics') and __import__('ellen_physics').manages(s):
            self.layout.prop(s.ellen_physics, 'enabled', text='统一人物物理（设置见下方）')
        else:
            self.layout.prop(s, 'wiggle_enable', text='人物次级物理' if s.get('CharacterStudio') else '头发 / 衣带 / 尾巴物理')
        environment = bpy.data.collections.get('Random Play House')
        if environment:
            self.layout.prop(environment, 'hide_viewport', text='隐藏房屋（仅预览）')

classes = (ELLEN_OT_home_room, ELLEN_PT_home, *CAMERA_DISTANCE_CLASSES)
def register():
    for c in classes:
        bpy.utils.register_class(c)
    register_navigation()
def unregister():
    unregister_navigation()
    for c in reversed(classes):
        bpy.utils.unregister_class(c)

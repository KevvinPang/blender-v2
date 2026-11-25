"""An unkeyed native navigation camera above the baked shot transform."""
import bpy
from mathutils import Matrix


def ensure_presentation_camera(scene):
    source=scene.camera
    if source.get('mocap_v2_navigation_view'):
        view=source
    else:
        # Do not reuse the previous shot's child: its offset may be unrelated
        # to the already calibrated shot that is being delivered.
        view=bpy.data.objects.new('相机V2 · 视图微调',source.data)
        scene.collection.objects.link(view)
        view.parent=source
        view.matrix_parent_inverse=Matrix.Identity(4)
        view.matrix_basis=Matrix.Identity(4)
        view['mocap_v2_navigation_view']=True
        view['mocap_distance_source']=source
        view['mocap_native_camera_navigation']=True
        view['character_director_camera']=True
        source['character_director_camera']=True
        source.name='V2 运镜轨迹（自动）'
    view.use_camera_lock_parent=False
    view.lock_location=(False,False,False)
    view.lock_rotation=(False,False,False)
    scene['MocapCameraDistanceEnabled']=True
    scene['MocapCameraDistanceView']=view
    scene.camera=view
    for marker in scene.timeline_markers:
        if marker.camera:marker.camera=view
    for screen in bpy.data.screens:
        for area in screen.areas:
            for space in area.spaces:
                if space.type!='VIEW_3D':continue
                space.use_local_camera=False;space.camera=view
                space.lock_camera=True
                space.region_3d.view_perspective='CAMERA'
    return view

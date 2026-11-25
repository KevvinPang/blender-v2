"""One explicit mapping between source observations and the chosen output shot.

A shot changes fixed optics, never bones or recovered subject motion. Solvers
use observation_projection; final checks also sample the physical projection.
Source masks, confidence, contact evidence and crop flags stay in source space.
"""
import json
import math
import numpy as np
from mathutils import Matrix

KEY = 'sam_shot_mapping'


def mapping(camera):
    # Custom metadata is authored on the original ID. Blender's evaluated
    # camera can retain an older custom-property snapshot until file reload,
    # even while its lens/shift values already evaluate to the new optics.
    value = camera.original.data.get(KEY)
    return json.loads(value) if value else None


def render_points(points, shot):
    values = np.asarray(points, dtype=float)
    return values.copy() if not shot else values*shot['scale']+np.asarray(shot['offset'])


def observation_projection(projection, aspect, shot):
    """Undo the output crop only in the solver's image coordinate system."""
    if not shot:
        return projection.copy()
    scale = shot['scale']; bx, by = shot['offset']
    p = np.asarray(projection, dtype=float).copy()
    p[0] = p[0]/scale+p[3]*((1-scale-2*bx/aspect)/scale)
    p[1] = p[1]/scale+p[3]*((scale-1+2*by)/scale)
    return Matrix(p.tolist())


def configure(scene, camera, target, options):
    """Apply an opt-in, adjustable chest shot after base calibration, once."""
    if KEY in camera.data:
        del camera.data[KEY]
    if options.get('camera_mode') != 'FIXED' or options.get('shot_preset', 'SOURCE') == 'SOURCE':
        return None
    scale = float(options.get('shot_scale', 1.25))
    vertical = float(options.get('shot_vertical', .15))
    if not math.isfinite(scale) or not .75 <= scale <= 2. or not math.isfinite(vertical) or abs(vertical) > .3:
        raise ValueError('取景倍率或纵向构图偏移超出支持范围')
    import bpy
    render = scene.render
    aspect = render.resolution_x*render.pixel_aspect_x/(render.resolution_y*render.pixel_aspect_y)
    # Horizontal center stays centered; vertical zoom pivots around the stable
    # reference eye position. The chosen downward offset preserves headroom.
    offset = np.array([(1-scale)*aspect*.5, (1-scale)*float(target[1])+vertical])
    graph = bpy.context.evaluated_depsgraph_get()
    def projection():
        bpy.context.view_layer.update()
        return camera.evaluated_get(graph).calc_matrix_camera(graph,
            x=render.resolution_x, y=render.resolution_y,
            scale_x=render.pixel_aspect_x, scale_y=render.pixel_aspect_y)
    def center(p):
        q = np.asarray(p) @ np.array([0., 0., -1., 1.])
        return np.array([(q[0]/q[3]+1)*aspect*.5, (1-q[1]/q[3])*.5])
    before = projection(); base_optics = {name:getattr(camera.data,name) for name in ('lens','shift_x','shift_y')}
    desired = center(before)*scale+offset
    camera.data.lens *= scale
    for axis, prop in enumerate(('shift_x', 'shift_y')):
        original = getattr(camera.data, prop); current = center(projection())
        setattr(camera.data, prop, original+.1)
        derivative = (center(projection())[axis]-current[axis])/.1
        if abs(derivative) < 1e-8:
            raise ValueError('摄像机取景偏移无法校准')
        setattr(camera.data, prop, original+(desired[axis]-current[axis])/derivative)
    shot = {'model':'fixed-shot-affine-v1', 'preset':'CHEST', 'scale':scale,
            'offset':offset.tolist(), 'reference_eye':list(target), 'vertical_offset':vertical,
            'base_optics':base_optics, 'optics':{name:getattr(camera.data,name) for name in base_optics},
            'headroom_policy':'initial composition only; later approach may leave frame',
            'parameter_source':'user-selected adjustable chest-shot preset; not inferred anatomy'}
    after = projection()
    error = float(np.max(abs(np.asarray(observation_projection(after,aspect,shot))-np.asarray(before))))
    if error > 1e-5:
        raise ValueError('取景镜头与动作观测坐标不一致，已取消生成')
    shot['projection_roundtrip_max_error'] = error
    camera.data[KEY] = json.dumps(shot)
    return shot

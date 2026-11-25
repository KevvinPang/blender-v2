"""Belle-only, deterministic single-burst iris sparkles for expression tracks."""
import math

PRESET = 'belle_star_happy'
LEFT = 'Belle_StarEyeLeft'
RIGHT = 'Belle_StarEyeRight'
PHASE = 'Belle_StarEyePhase'
CONTROLS = (LEFT, RIGHT, PHASE)
SIDES = {LEFT: 'LEFT', RIGHT: 'RIGHT'}
DURATION = 1.1
LABEL = '铃 / 星星眼开心（闪现一次）'
VALUES = {'にこり': .70, '上': .18, 'びっくり左': .16,
          'びっくり右': .16, '口角上げ': .45, 'ワ': .14,
          'Ellen_CheekBlush': .18, LEFT: 1, RIGHT: 1, PHASE: 0}


def is_belle(mesh):
    return bool(mesh and mesh.type == 'MESH' and mesh.data.shape_keys and
                'Belle_MouthWidth_Adjustment' in mesh.data.shape_keys.key_blocks)


def timed_value(name, elapsed):
    t = elapsed / DURATION
    if name == PHASE:
        return max(0.0, min(1.0, t))
    if t <= 0 or t >= 1:
        return 0.0
    def smooth(v):
        v = max(0.0, min(1.0, v))
        return v*v*(3-2*v)
    envelope = smooth(t/.16) * smooth((1-t)/.32)
    return envelope * (.93+.07*math.cos(4*math.pi*t))


def preview_frame(item, rate):
    return min(item.end_frame, item.start_frame+max(1, round(.20*rate)))


def burst_origins(ranges):
    """Touching pieces of the same expression continue a single burst."""
    origins = {}
    previous = None
    for start, end, item in ranges:
        if item.preset == PRESET:
            continuing = previous and previous[2].preset == PRESET and start <= previous[1]+1
            origins[item.as_pointer()] = origins[previous[2].as_pointer()] if continuing else start
        previous = (start, end, item)
    return origins


def ensure(mesh):
    """Add separate iris-bound meshes; the face topology and original keys stay intact."""
    import bpy
    import numpy as np
    from mathutils import Matrix, Vector
    if not is_belle(mesh):
        raise ValueError('星星眼开心仅适用于铃的面部模型')
    rig = mesh.find_armature()
    if not rig:
        raise ValueError('未找到铃的骨骼，无法让星星跟随眼睛')
    keys = mesh.data.shape_keys
    for name in CONTROLS:
        if name not in keys.key_blocks:
            mesh.shape_key_add(name=name, from_mix=False).value = 0
    labels = [m.get('original_material', m.name.rsplit('|', 1)[-1].strip().split('.')[0])
              if m else '' for m in mesh.data.materials]
    iris = sorted({v for p in mesh.data.polygons if labels[p.material_index] == '目' for v in p.vertices})
    if not iris:
        raise ValueError('铃模型缺少虹膜材质，无法定位星星眼')
    base = np.empty((len(mesh.data.vertices), 3), np.float32)
    keys.key_blocks[0].data.foreach_get('co', base.ravel())
    eye_signs = {}
    for control, side in SIDES.items():
        reference = keys.key_blocks.get('ウィンク２' if side == 'LEFT' else 'ウィンク２右')
        if reference is None:
            raise ValueError('铃模型缺少左右wink形态键')
        a = np.empty_like(base);reference.data.foreach_get('co', a.ravel())
        changed = np.linalg.norm(a-base, axis=1) > 1e-6
        eye_signs[control] = 1 if np.median(base[changed, 0]) > 0 else -1
    collection = next(iter(mesh.users_collection), bpy.context.scene.collection)
    mats = []
    for name, color in (('Belle Star Gold', (1.0, .53, .018, 1)),
                        ('Belle Star Light', (1.0, .90, .25, 1))):
        mat = bpy.data.materials.get(name)
        if mat is None:
            mat = bpy.data.materials.new(name);mat.use_nodes = True
            nodes = mat.node_tree.nodes;nodes.clear()
            out = nodes.new('ShaderNodeOutputMaterial')
            emission = nodes.new('ShaderNodeEmission')
            emission.inputs['Color'].default_value = color
            emission.inputs['Strength'].default_value = 1.3
            mat.node_tree.links.new(emission.outputs[0], out.inputs['Surface'])
            mat.diffuse_color = color
        mats.append(mat)
    for control, sign in eye_signs.items():
        object_name = mesh.name+' | Star Eye '+SIDES[control]
        existing = bpy.data.objects.get(object_name)
        if existing and existing.get('belle_star_owner') == mesh.name:
            continue
        ids = [i for i in iris if base[i, 0]*sign > 0]
        a = base[ids]
        center = (a.min(axis=0)+a.max(axis=0))*.5
        center[1] = a[:, 1].min()-.0009
        # Keep the main star inside the iris. Its inner gold face and small
        # satellite provide the four-point sparkle seen in the reference.
        radius = float(np.ptp(a[:, 2]))*.245
        vertices = [];faces = [];material_indices = []
        def star(cx, cz, r, material, depth):
            offset = len(vertices)
            vertices.append((cx, depth, cz))
            for j in range(8):
                angle = math.pi/2+j*math.pi/4
                length = r if j%2 == 0 else r*.29
                vertices.append((cx+math.cos(angle)*length, depth, cz+math.sin(angle)*length))
            for j in range(8):
                faces.append((offset, offset+1+j, offset+1+(j+1)%8));material_indices.append(material)
        star(.0007*sign, -.0007, radius, 0, 0)
        star(.0007*sign, -.0007, radius*.73, 1, -.00003)
        star(-radius*.79*sign, radius*.70, radius*.29, 1, -.00005)
        data = bpy.data.meshes.new(object_name);data.from_pydata(vertices, [], faces);data.update()
        obj = bpy.data.objects.new(object_name, data);collection.objects.link(obj)
        obj.matrix_world = mesh.matrix_world@Matrix.Translation(Vector(center))
        obj['belle_star_owner'] = mesh.name
        obj['belle_star_side'] = SIDES[control]
        obj['belle_star_duration'] = DURATION
        for mat in mats:data.materials.append(mat)
        for polygon, index in zip(data.polygons, material_indices):polygon.material_index = index
        # Inherit the exact eye skinning, including head motion and gaze.
        nearest = min(ids, key=lambda i:float(np.linalg.norm(base[i]-center)))
        for group in mesh.data.vertices[nearest].groups:
            vg = obj.vertex_groups.new(name=mesh.vertex_groups[group.group].name)
            vg.add(list(range(len(vertices))), group.weight, 'REPLACE')
        tree = bpy.data.node_groups.new(object_name+' Transform', 'GeometryNodeTree')
        tree.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
        tree.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
        source = tree.nodes.new('NodeGroupInput');out = tree.nodes.new('NodeGroupOutput')
        transform = tree.nodes.new('GeometryNodeTransform')
        tree.links.new(source.outputs['Geometry'], transform.inputs['Geometry'])
        tree.links.new(transform.outputs['Geometry'], out.inputs['Geometry'])
        def drive(socket, index, name, expression):
            driver = socket.driver_add('default_value', index).driver
            var = driver.variables.new();var.name = 'v';var.type = 'SINGLE_PROP'
            var.targets[0].id_type = 'KEY';var.targets[0].id = keys
            var.targets[0].data_path = keys.key_blocks[name].path_from_id('value')
            driver.expression = expression
        for index in range(3):drive(transform.inputs['Scale'], index, control, 'v')
        drive(transform.inputs['Rotation'], 1, PHASE, '.16 + v*1.12')
        modifier = obj.modifiers.new('Star Burst Scale and Rotation', 'NODES');modifier.node_group = tree
        armature = obj.modifiers.new('Follow Eye and Head', 'ARMATURE');armature.object = rig
        for attr in ('visible_shadow',):
            if hasattr(obj, attr):setattr(obj, attr, False)
    mesh['BelleStarEyesRevision'] = 1

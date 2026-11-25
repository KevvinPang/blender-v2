"""One character-owned phone with reversible left/right wrist grips."""
import math
import json
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector

MODES = {'PHONE_LEFT': 'L', 'PHONE_RIGHT': 'R'}
TAG = 'SAM Phone Grip'
_reference = None
_reference_path = None
DISPLAY = '手机 / 人物一体显示'
_catalog = None


def phone_profile(rig):
    """Resolve the prop from this rig, never from the project filename."""
    global _catalog
    if _catalog is None:
        _catalog = json.loads(Path(__file__).with_name('sam_phone_profiles.json').read_text(encoding='utf8'))
    selected = rig.get('sam_phone_profile', 'AUTO')
    if selected != 'AUTO':
        if selected not in _catalog:
            raise ValueError('此角色的手机配置不存在：' + selected)
        return selected, _catalog[selected]
    labels = []
    owner = rig
    while owner:
        if getattr(owner, 'mmd_type', '') == 'ROOT':
            labels += [owner.mmd_root.name, owner.mmd_root.name_e]
        owner = owner.parent
    # Some imported skins lack an MMD model root; their rig name is still local
    # to the selected character, unlike scene-wide studio metadata.
    labels.append(rig.name.rsplit('_arm', 1)[0])
    matches = set()
    for key, profile in _catalog.items():
        for label in labels:
            value = label.strip().casefold()
            if any(value == alias.casefold() or
                   (alias in ('艾莲', 'エレン', 'ellen') and alias in value)
                   for alias in profile['aliases']):
                matches.add(key)
    if len(matches) != 1:
        raise ValueError('此角色尚未配置专属手机，请先添加角色手机配置；不会套用其他角色款式')
    key = matches.pop()
    return key, _catalog[key]


def reference_profile(rig):
    global _reference, _reference_path
    _, catalog = phone_profile(rig)
    path = Path(__file__).with_name(catalog.get('grip_reference', 'sam_ellen_phone_grips.json'))
    if _reference is None or _reference_path != path:
        _reference = json.loads(path.read_text(encoding='utf8')) if path.exists() else {}
        _reference_path = path
    profile = _reference
    for label, values in profile.get('rest', {}).items():
        bone = next((b for b in rig.pose.bones if b.mmd_bone.name_j == label), None)
        if not bone or max(abs(bone.bone.matrix_local[i][j]-values[i][j])
                           for i in range(4) for j in range(4)) > 1e-4:
            return {}
    return profile


def bones(rig, side='L'):
    label = '左' if side == 'L' else '右'
    result = {b.mmd_bone.name_j: b for b in rig.pose.bones
              if getattr(b, 'mmd_bone', None)}
    required = [label+'手首']+[f'{label}{finger}{number}' for finger in ('人指', '中指', '薬指', '小指')
                         for number in '１２３']+[label+'親指'+number for number in '０１２']
    if any(name not in result for name in required):
        raise ValueError(f'此骨架缺少{label}手手腕或手指映射，未绑定手机')
    return result


def hand_frame(rig):
    mapped = bones(rig)
    wrist = mapped['左手首'].bone
    forward = (mapped['左中指１'].bone.head_local-wrist.head_local).normalized()
    up = (mapped['左人指１'].bone.head_local-mapped['左小指１'].bone.head_local).normalized()
    up = (up-forward*up.dot(forward)).normalized()
    normal = up.cross(forward).normalized()
    rotation = Matrix((forward, normal, up)).transposed()
    return wrist, rotation


def _left_finger_rotations(rig):
    mapped = bones(rig)
    wrist, frame = hand_frame(rig)
    up = frame.col[2]
    _, profile = phone_profile(rig)
    curl_sign = profile.get('grip_curl_sign', 1)
    result = {}
    for finger, angles in (('人指', (42, 88, 58)), ('中指', (43, 90, 62)),
                           ('薬指', (45, 88, 60)), ('小指', (48, 83, 55))):
        for number, angle in zip('１２３', angles):
            bone = mapped['左'+finger+number].bone
            # Positive curl wraps the fingers from the rear edge toward the screen.
            axis = bone.matrix_local.to_3x3().inverted() @ up
            result[bone.name] = Quaternion(axis, math.radians(angle)*curl_sign)
    # Keep the thumb behind the case first, wrap around its edge, then onto the screen.
    thumb = [mapped['左親指'+number] for number in '０１２']
    parent_matrix = thumb[0].parent.bone.matrix_local.copy()
    for bone, direction in zip(thumb, ((-.8, 0, .6), (.15, .98, -.1), (.94, .1, -.33))):
        rest_parent = bone.parent.bone.matrix_local
        neutral = parent_matrix @ rest_parent.inverted() @ bone.bone.matrix_local
        desired = neutral.copy()
        direction = Vector((direction[0], direction[1]*curl_sign, direction[2])).normalized()
        swing = neutral.to_3x3().col[1].rotation_difference(frame @ direction)
        desired_rotation = swing.to_matrix() @ neutral.to_3x3()
        for i in range(3):
            for j in range(3):
                desired[i][j] = desired_rotation[i][j]
        local = bone.bone.convert_local_to_pose(desired, bone.bone.matrix_local, invert=True,
                    parent_matrix=parent_matrix, parent_matrix_local=rest_parent)
        result[bone.name] = local.to_quaternion().normalized()
        parent_matrix = desired
    return result


def finger_rotations(rig, side='L'):
    reference = reference_profile(rig).get(side, {}).get('grip')
    if reference:
        mapped = bones(rig, side)
        return {mapped[label].name:Quaternion(value) for label,value in reference.items()}
    rotations = _left_finger_rotations(rig)
    if side == 'L':
        return rotations
    mapped = bones(rig, 'R')
    reflection = Matrix.Diagonal((-1., 1., 1.))
    posed_left, posed_right, result = {}, {}, {}
    for name, rotation in rotations.items():
        left = rig.pose.bones[name]
        right = mapped['右'+left.mmd_bone.name_j[1:]]
        left_parent = posed_left.get(left.parent.name, left.parent.bone.matrix_local.to_3x3())
        right_parent = posed_right.get(right.parent.name, right.parent.bone.matrix_local.to_3x3())
        left_rest = left.bone.matrix_local.to_3x3()
        right_rest = right.bone.matrix_local.to_3x3()
        left_pose = left_parent @ left.parent.bone.matrix_local.to_3x3().inverted() @ left_rest @ rotation.to_matrix()
        delta = left_pose @ left_rest.inverted()
        desired = reflection @ delta @ reflection @ right_rest
        neutral = right_parent @ right.parent.bone.matrix_local.to_3x3().inverted() @ right_rest
        result[right.name] = (neutral.inverted() @ desired).to_quaternion().normalized()
        posed_left[name], posed_right[right.name] = left_pose, desired
    return result


def default_offset(rig, side='L'):
    reference = reference_profile(rig).get(side, {}).get('offset')
    if reference:
        return Matrix(reference)
    wrist, orientation = hand_frame(rig)
    _, profile = phone_profile(rig)
    placement = orientation.to_4x4() @ Matrix.Rotation(math.radians(profile.get('fit_tilt_degrees',20)), 4, 'Y')
    if profile.get('fit_palm_side', 1) < 0:
        # Flip the case, keeping its long axis; this character's palm is on
        # the opposite side of the geometric left-hand frame from Ellen's.
        placement = placement @ Matrix.Rotation(math.pi, 4, 'Z')
    placement.translation = wrist.head_local+orientation @ Vector(profile.get('fit_translation',(.024, .011, -.002)))
    if side == 'R':
        reflection = Matrix.Diagonal((-1., 1., 1., 1.))
        placement = reflection @ placement @ reflection
        wrist = bones(rig, 'R')['右手首'].bone
    return wrist.matrix_local.inverted() @ placement


def bind_character(scene, rig, anchor, side):
    root = anchor.children[0]
    old_side = anchor.get('sam_phone_side', 'L')
    revision = reference_profile(rig).get('revision', 0)
    _, profile = phone_profile(rig)
    fit_revision = profile.get('fit_revision', 0)
    if fit_revision and root.get('sam_phone_profile_fit_revision') != fit_revision:
        root['sam_phone_previous_fit_'+old_side] = [list(row) for row in root.matrix_local]
        for hand in ('L', 'R'):
            root['sam_phone_fit_'+hand] = [list(row) for row in default_offset(rig, hand)]
        root.matrix_local = default_offset(rig, old_side)
        root['sam_phone_profile_fit_revision'] = fit_revision
    if revision and root.get('sam_phone_grip_revision') != revision:
        for hand in ('L', 'R'):
            root['sam_phone_fit_'+hand] = [list(row) for row in default_offset(rig, hand)]
        root.matrix_local = default_offset(rig, old_side)
        root['sam_phone_grip_revision'] = revision
    if old_side != side:
        root['sam_phone_fit_'+old_side] = [list(row) for row in root.matrix_local]
        root.matrix_local = Matrix(root.get('sam_phone_fit_'+side, default_offset(rig, side)))
    root['sam_phone_offset'] = [list(row) for row in default_offset(rig, side)]
    anchor['sam_phone_side'] = side
    wrist = bones(rig, side)[('左' if side == 'L' else '右')+'手首'].bone
    # Native bone parenting places the complete prop under the character, once.
    for constraint in list(anchor.constraints):
        if constraint.type == 'COPY_TRANSFORMS' and constraint.target == rig:
            anchor.constraints.remove(constraint)
    anchor.parent, anchor.parent_type, anchor.parent_bone = rig, 'BONE', wrist.name
    anchor.matrix_parent_inverse = Matrix.Identity(4)
    anchor.matrix_basis = Matrix.Translation((0, -wrist.length, 0))
    anchor.name = '手机 / '+('左' if side == 'L' else '右')+'手腕挂点'
    character = rig.users_collection[0] if rig.users_collection else None
    if character:
        for collection in anchor.users_collection:
            if collection == character:
                continue
            if collection.name not in character.children:
                character.children.link(collection)
            if collection.name in scene.collection.children:
                scene.collection.children.unlink(collection)


def ensure_local_color(mat):
    """Keep the phone's base colors visible when local view excludes room lights.

    The studio uses Eevee, like the character's existing toon materials. Clamp
    the lit surface to a dim base-color floor while retaining brighter lighting
    and metallic reflections from the original Principled shader.
    """
    if mat.get('sam_phone_local_color') == 1:
        return
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    shader = next(n for n in nodes if n.type == 'BSDF_PRINCIPLED')
    output = next(n for n in nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output)
    lit = nodes.new('ShaderNodeShaderToRGB')
    lit.name = 'Phone lit surface'
    floor = nodes.new('ShaderNodeVectorMath')
    floor.name = 'Phone minimum base color'
    floor.operation = 'MAXIMUM'
    floor.inputs[1].default_value = tuple(c * .35 for c in shader.inputs['Base Color'].default_value[:3])
    surface = nodes.new('ShaderNodeEmission')
    surface.name = 'Phone local-view color'
    shader.location, lit.location = (-600, 0), (-300, 0)
    floor.location, surface.location, output.location = (-80, 0), (140, 0), (360, 0)
    links.new(shader.outputs['BSDF'], lit.inputs['Shader'])
    links.new(lit.outputs['Color'], floor.inputs[0])
    links.new(floor.outputs['Vector'], surface.inputs['Color'])
    links.new(surface.outputs[0], output.inputs['Surface'])
    mat['sam_phone_local_color'] = 1


def material(name, color, metallic=0., roughness=.35, emission=0.):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    node = next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    node.inputs['Base Color'].default_value = (*color, 1)
    node.inputs['Metallic'].default_value = metallic
    node.inputs['Roughness'].default_value = roughness
    if emission:
        node.inputs['Emission Color'].default_value = (*color, 1)
        node.inputs['Emission Strength'].default_value = emission
    ensure_local_color(mat)
    return mat


def rounded_phone_mesh(name, size, radius, roll):
    """Round the silhouette independently of the thin panel's edge bevel."""
    width, depth, height = size
    rings = []
    for front in (False, True):
        steps = range(5) if not front else range(4, -1, -1)
        for step in steps:
            angle = step * math.pi / 8
            inset = roll * (1 - math.sin(angle))
            y = -depth / 2 + roll * (1 - math.cos(angle))
            if front:
                y = -y
            r = radius - inset
            ring = []
            for corner in range(4):
                angle0 = corner * math.pi / 2
                cx = (width / 2 - radius) * (1 if corner in (0, 3) else -1)
                cz = (height / 2 - radius) * (1 if corner in (0, 1) else -1)
                for step2 in range(17):
                    a = angle0 + step2 * math.pi / 32
                    ring.append((cx + r * math.cos(a), y, cz + r * math.sin(a)))
            rings.append(ring)
    count = len(rings[0])
    vertices = [v for ring in rings for v in ring]
    faces = [tuple(range(count))]
    for ring in range(len(rings) - 1):
        for i in range(count):
            j = (i + 1) % count
            faces.append((ring * count + i, (ring + 1) * count + i,
                          (ring + 1) * count + j, ring * count + j))
    faces.append(tuple(reversed(range((len(rings) - 1) * count, len(rings) * count))))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    for face in mesh.polygons:
        face.use_smooth = len(face.vertices) == 4
    return mesh


def shark_outline():
    # Cubic curves round the snout and belly; the dorsal fin and tail taper.
    start = (-.024, -.011)
    segments = [
        ((-.024, -.007), (-.018, -.003), (-.010, -.003)),
        ((-.008, -.003), (-.006, -.003), (-.005, -.002)),
        ((-.004, .002), (-.002, .010), (0, .011)),
        ((.0015, .012), (.001, .001), (.004, -.003)),
        ((.007, -.005), (.011, -.006), (.015, -.008)),
        ((.018, -.006), (.022, .000), (.024, .001)),
        ((.027, .003), (.021, -.008), (.021, -.010)),
        ((.021, -.013), (.025, -.018), (.024, -.019)),
        ((.023, -.020), (.018, -.015), (.015, -.014)),
        ((.011, -.017), (.008, -.019), (.003, -.019)),
        ((.001, -.020), (-.002, -.025), (-.004, -.026)),
        ((-.007, -.028), (-.005, -.020), (-.007, -.018)),
        ((-.013, -.020), (-.021, -.018), (-.023, -.014)),
        ((-.024, -.013), (-.024, -.012), start),
    ]
    points, previous = [], start
    for control1, control2, end in segments:
        for step in range(12):
            t = step / 12
            points.append(tuple((1-t)**3 * previous[i] + 3*(1-t)**2*t*control1[i]
                                + 3*(1-t)*t*t*control2[i] + t**3*end[i] for i in (0, 1)))
        previous = end
    return points


def refine_phone_geometry(anchor):
    if anchor.get('sam_phone_geometry_revision') == 2:
        return
    panels = {
        '金属中框': ((.061, .009, .136), .0085, .0014),
        '深蓝后盖': ((.059, .0025, .134), .008, .0007),
        '正面玻璃': ((.057, .0013, .129), .0075, .0004),
        '屏幕': ((.050, .0006, .109), .0055, .00018),
        '后置相机底座': ((.017, .002, .034), .0075, .0005),
        '通知卡片': ((.043, .00025, .012), .0035, .00007),
        '后盖红色装饰条': ((.026, .00025, .0015), .0007, .00007),
    }
    for obj in anchor.children[0].children_recursive:
        label = obj.name.removeprefix('手机 /').strip()
        replacement = None
        if label in panels:
            size, radius, roll = panels[label]
            replacement = rounded_phone_mesh(obj.name + ' / 圆润轮廓', size, radius, roll)
            for modifier in list(obj.modifiers):
                if modifier.type in {'BEVEL', 'WEIGHTED_NORMAL'}:
                    obj.modifiers.remove(modifier)
        elif label in {'后盖鲨鱼标识', '屏幕鲨鱼'}:
            points = shark_outline()
            y = -.006 if label == '后盖鲨鱼标识' else .0067
            if label == '屏幕鲨鱼':
                points = [(x * .92, z - .016) for x, z in points]
            replacement = bpy.data.meshes.new(obj.name + ' / 曲线轮廓')
            coords = [Vector((x, y, z)) for x, z in points]
            from mathutils.geometry import tessellate_polygon
            lookup = {tuple(v): i for i, v in enumerate(coords)}
            triangles = [[v if isinstance(v, int) else lookup[tuple(v)] for v in triangle]
                         for triangle in tessellate_polygon([coords])]
            replacement.from_pydata(coords, [], triangles)
            replacement.update()
        elif label.startswith('红色鲨鱼鳃'):
            # Rounded, gently bowed gills instead of sharp rectangular slashes.
            old_points = [v.co for v in obj.data.vertices]
            x = min(p.x for p in old_points) + .00085
            coords = []
            for i in range(32):
                t = i * 2 * math.pi / 32
                z = -.012 + .0031 * math.sin(t)
                coords.append((x + .0006 * math.cos(t) + .00065 * (1-math.sin(t)**2),
                               -.00615, z))
            replacement = bpy.data.meshes.new(obj.name + ' / 圆润鱼鳃')
            replacement.from_pydata(coords, [], [tuple(range(len(coords)))])
            replacement.update()
        if replacement:
            old = obj.data
            for mat in old.materials:
                replacement.materials.append(mat)
            obj.data = replacement
            if old.users == 0:
                bpy.data.meshes.remove(old)
        if label == '鲨鱼眼睛':
            obj.location.x = -.017
    anchor['sam_phone_geometry_revision'] = 2


def build_shark(scene, rig):
    wrist, orientation = hand_frame(rig)
    collection = bpy.data.collections.new('艾莲手机 / 雾蓝鲨鱼款')
    scene.collection.children.link(collection)
    anchor = bpy.data.objects.new('手机 / 左手腕挂点', None)
    collection.objects.link(anchor)
    anchor.empty_display_size = .012
    anchor['sam_phone_rig'] = rig.name
    follow = anchor.constraints.new('COPY_TRANSFORMS')
    follow.name = '跟随左手手腕'
    follow.target, follow.subtarget = rig, wrist.name
    root = bpy.data.objects.new('手机 / 位置与角度', None)
    collection.objects.link(root)
    root.parent = anchor
    root.empty_display_size = .01
    root.matrix_local = default_offset(rig)
    root['sam_phone_offset'] = [list(row) for row in root.matrix_local]
    mats = {
        'shell': material('手机 / 浅雾蓝外壳', (.12, .27, .48), .3, .28),
        'edge': material('手机 / 浅蓝金属边框', (.19, .34, .52), .65, .24),
        'ink': material('手机 / 深蓝屏幕图案', (.016, .047, .16)),
        'glass': material('手机 / 黑蓝玻璃', (.006, .012, .025), .15, .18),
        'screen': material('手机 / 薄荷蓝屏幕', (.26, .72, .79), 0, .4, .55),
        'white': material('手机 / 雾白图案', (.78, .89, .93), .05),
        'pink': material('手机 / 珊瑚红点缀', (.84, .13, .25), .15),
        'lens': material('手机 / 镜头蓝玻璃', (.015, .08, .16), .6, .13),
    }

    def move(obj, name, mat):
        obj.name = name
        for old in list(obj.users_collection):
            old.objects.unlink(obj)
        collection.objects.link(obj)
        obj.parent = root
        obj.data.materials.append(mats[mat])
        return obj

    def box(name, size, location, mat, bevel=.002):
        bpy.ops.mesh.primitive_cube_add(size=1)
        obj = move(bpy.context.object, name, mat)
        obj.scale = size
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        obj.location = location
        if bevel:
            modifier = obj.modifiers.new('圆角', 'BEVEL')
            modifier.width, modifier.segments = bevel, 4
            obj.modifiers.new('加权法线', 'WEIGHTED_NORMAL')
        return obj

    def disk(name, radius, y, x, z, mat):
        bpy.ops.mesh.primitive_cylinder_add(vertices=40, radius=radius, depth=.001,
                                            rotation=(math.pi/2, 0, 0))
        obj = move(bpy.context.object, name, mat)
        obj.location = (x, y, z)
        bevel = obj.modifiers.new('镜头倒角', 'BEVEL')
        bevel.width, bevel.segments = .0003, 2
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
        return obj

    def polygon(name, points, y, mat):
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata([(x, y, z) for x, z in points], [], [list(range(len(points)))])
        obj = bpy.data.objects.new(name, mesh)
        collection.objects.link(obj)
        obj.parent = root
        mesh.materials.append(mats[mat])
        return obj

    def label(name, text, location, size, mat, rear=False):
        curve = bpy.data.curves.new(name, 'FONT')
        curve.body, curve.size, curve.align_x = text, size, 'CENTER'
        curve.extrude = .000015
        obj = bpy.data.objects.new(name, curve)
        collection.objects.link(obj)
        obj.parent, obj.location = root, location
        obj.rotation_euler = (math.pi/2, 0, 0 if rear else math.pi)
        curve.materials.append(mats[mat])
        return obj

    box('手机 / 金属中框', (.061, .009, .136), (0, 0, 0), 'edge', .004)
    box('手机 / 深蓝后盖', (.059, .0025, .134), (0, -.0044, 0), 'shell', .0038)
    box('手机 / 正面玻璃', (.057, .0013, .129), (0, .005, 0), 'glass', .003)
    box('手机 / 屏幕', (.050, .0006, .109), (0, .0059, -.001), 'screen', .0015)
    box('手机 / 顶部听筒', (.019, .0007, .0015), (0, .0059, .060), 'glass', .0007)
    disk('手机 / 前置摄像头', .0011, .006, -.015, .060, 'lens')
    box('手机 / 电源键', (.0014, .004, .015), (.0305, 0, .022), 'pink', .0006)
    box('手机 / 音量键', (.0014, .004, .020), (-.0305, 0, .022), 'edge', .0006)
    box('手机 / 充电接口', (.010, .003, .001), (0, 0, -.068), 'glass', .0004)
    box('手机 / 后置相机底座', (.017, .002, .034), (-.0175, -.006, .043), 'glass', .003)
    for z in (.052, .037):
        disk('手机 / 后置镜头金属环', .0056, -.0076, -.0175, z, 'edge')
        disk('手机 / 后置镜片', .0042, -.0083, -.0175, z, 'lens')
        disk('手机 / 镜片反光', .0012, -.0089, -.0185, z+.001, 'screen')
    disk('手机 / 闪光灯', .0016, -.0061, -.004, .051, 'white')
    # A shark silhouette and coral gill marks give the rear a clear character motif.
    shark = [(-.020,-.012),(-.012,-.004),(-.005,-.001),(.002,.011),(.004,.000),
             (.013,-.004),(.024,.000),(.020,-.010),(.024,-.018),(.012,-.014),
             (.005,-.019),(-.004,-.027),(-.005,-.018),(-.014,-.019)]
    polygon('手机 / 后盖鲨鱼标识', shark, -.006, 'white')
    for x in (-.008, -.004, 0):
        polygon('手机 / 红色鲨鱼鳃', [(x,-.009),(x+.0015,-.009),(x-.0002,-.015),
                                    (x-.0017,-.015)], -.00615, 'pink')
    disk('手机 / 鲨鱼眼睛', .001, -.00625, -.014, -.011, 'glass')
    label('手机 / 后盖型号', 'SHARK / 07', (0, -.00615, -.050), .0038, 'white', True)
    box('手机 / 后盖红色装饰条', (.026, .00025, .0015), (0, -.00615, -.056), 'pink', .0004)
    label('手机 / 时间', '07:30', (0, .0064, .031), .012, 'ink')
    label('手机 / 日期', 'MON. 18', (0, .0064, .044), .0035, 'ink')
    box('手机 / 通知卡片', (.043, .00025, .012), (0, .0065, .018), 'white', .0017)
    label('手机 / 通知文字', 'BACK TO SCHOOL', (0, .0068, .016), .0032, 'ink')
    polygon('手机 / 屏幕鲨鱼', [(x*.92, z- .016) for x, z in shark], .0067, 'ink')
    for x in (-.007, 0, .007):
        disk('手机 / 屏幕分页点', .00075, .0068, x, -.048, 'ink')
    refine_phone_geometry(anchor)
    return anchor, root


def build(scene, rig):
    key, profile = phone_profile(rig)
    if profile.get('builder') == 'shark':
        anchor, root = build_shark(scene, rig)
    else:
        path = Path(__file__).parent / profile['asset']
        if not path.is_file():
            raise ValueError('缺少此角色的手机资源：' + str(path))
        with bpy.data.libraries.load(str(path), link=False) as (source, target):
            if profile['collection'] not in source.collections:
                raise ValueError('手机资源缺少配置的模型集合')
            target.collections = [profile['collection']]
        collection = target.collections[0]
        scene.collection.children.link(collection)
        model = next(o for o in collection.objects if o.name.startswith(profile['root']) and not o.parent)
        anchor = bpy.data.objects.new('手机 / 左手腕挂点', None)
        collection.objects.link(anchor)
        anchor['sam_phone_rig'] = rig.name
        anchor.empty_display_size = .012
        root = bpy.data.objects.new('手机 / 位置与角度', None)
        collection.objects.link(root)
        root.parent = anchor
        root.matrix_local = default_offset(rig)
        root['sam_phone_offset'] = [list(row) for row in root.matrix_local]
        # Asset: +Y long axis, +Z rear. Grip: +Z long axis, -Y rear.
        model.parent = root
        model.matrix_local = Matrix.Rotation(math.pi/2, 4, 'X')
        model['stage'] = 'Character wrist binding / frame-range visibility'
        for obj in model.children_recursive:
            label = obj.name.removeprefix('玲手机 /').strip().split('.')[0]
            obj['sam_phone_outline_solid'] = label in profile.get('outline_parts', [])
    anchor['sam_phone_profile'] = key
    root['sam_phone_profile'] = key
    return anchor, root


def remove_phone(anchor):
    for child in reversed(list(anchor.children_recursive)):
        bpy.data.objects.remove(child, do_unlink=True)
    bpy.data.objects.remove(anchor, do_unlink=True)


def ensure(scene, rig=None):
    rig = rig or scene.sam_ellen.rig
    if not rig:
        raise ValueError('请先选择人物骨架')
    side = scene.sam_ellen.phone_side
    bones(rig, side)
    key, profile = phone_profile(rig)
    anchor = next((o for o in scene.objects if o.get('sam_phone_rig') == rig.name), None)
    if anchor and (anchor.get('sam_phone_profile', 'ELLEN_SHARK') != key or
                   not any(child.type == 'MESH' for child in anchor.children_recursive)):
        # Load and validate the replacement before retiring an existing prop.
        replacement, root = build(scene, rig)
        remove_phone(anchor)
        anchor = replacement
    if anchor is None:
        anchor, root = build(scene, rig)
    anchor['sam_phone_profile'] = key
    rig['sam_phone_profile'] = key
    bind_character(scene, rig, anchor, side)
    sync(scene)
    return anchor


def apply_grip(rig, side):
    for name, rotation in finger_rotations(rig, side).items():
        bone = rig.pose.bones[name]
        constraint = bone.constraints.get(TAG) or bone.constraints.new('LIMIT_ROTATION')
        constraint.name = TAG
        constraint.owner_space = 'LOCAL'
        constraint.euler_order = 'XYZ'
        angles = rotation.to_euler('XYZ')
        for axis, value in zip('xyz', angles):
            setattr(constraint, 'use_limit_'+axis, True)
            setattr(constraint, 'min_'+axis, value)
            setattr(constraint, 'max_'+axis, value)


def ensure_phone_outline(group):
    """Outline a whitelist of solid parts before adding them to the character."""
    if group.get('sam_phone_outline_revision') == 1:
        return
    outline = bpy.data.materials.get('Ellen Fine Dark Outline')
    if outline is None:
        return
    solid_labels = ('金属中框', '深蓝后盖', '正面玻璃', '后置相机底座',
                    '后置镜头金属环', '后置镜片', '前置摄像头', '电源键', '音量键')
    solid_nodes = [node for node in group.nodes if node.type == 'OBJECT_INFO'
                   and node.inputs['Object'].default_value
                   and (node.inputs['Object'].default_value.get('sam_phone_outline_solid') or
                        node.inputs['Object'].default_value.name.removeprefix('手机 /').strip()
                        .split('.')[0] in solid_labels)]
    if not solid_nodes:
        return
    enabled = group.interface.new_socket(name='手机实体描边', in_out='INPUT', socket_type='NodeSocketBool')
    enabled.default_value = True
    width = group.interface.new_socket(name='手机描边宽度', in_out='INPUT', socket_type='NodeSocketFloat')
    width.default_value = .0004
    width.min_value, width.max_value = 0, .0015
    width.subtype = 'DISTANCE'
    nodes, links = group.nodes, group.links
    incoming = next(n for n in nodes if n.type == 'GROUP_INPUT')
    join = next(n for n in nodes if n.type == 'JOIN_GEOMETRY')
    solids = nodes.new('GeometryNodeJoinGeometry')
    solids.name = '手机 / 仅实体部件'
    for node in solid_nodes:
        links.new(node.outputs['Geometry'], solids.inputs['Geometry'])
    normal = nodes.new('GeometryNodeInputNormal')
    scale = nodes.new('ShaderNodeVectorMath')
    scale.operation = 'SCALE'
    links.new(normal.outputs['Normal'], scale.inputs[0])
    links.new(incoming.outputs[width.identifier], scale.inputs['Scale'])
    position = nodes.new('GeometryNodeSetPosition')
    links.new(solids.outputs['Geometry'], position.inputs['Geometry'])
    links.new(scale.outputs['Vector'], position.inputs['Offset'])
    flip = nodes.new('GeometryNodeFlipFaces')
    links.new(position.outputs['Geometry'], flip.inputs['Mesh'])
    material = nodes.new('GeometryNodeSetMaterial')
    material.inputs['Material'].default_value = outline
    links.new(flip.outputs['Mesh'], material.inputs['Geometry'])
    switch = nodes.new('GeometryNodeSwitch')
    switch.input_type = 'GEOMETRY'
    switch.name = '手机 / XIYAG 实体描边'
    links.new(incoming.outputs[enabled.identifier], switch.inputs['Switch'])
    links.new(material.outputs['Geometry'], switch.inputs['True'])
    links.new(switch.outputs['Output'], join.inputs['Geometry'])
    for index, node in enumerate((solids, normal, scale, position, flip, material, switch)):
        node.location = (-300 + index * 210, 500)
    group['sam_phone_outline_revision'] = 1
    group['sam_phone_outline_sources'] = [n.inputs['Object'].default_value.name for n in solid_nodes]


def ensure_display(rig, anchor):
    candidates = [o for o in rig.children if o.type == 'MESH' and
                  any(m.type == 'ARMATURE' and m.object == rig for m in o.modifiers)]
    if not candidates:
        raise ValueError('没有找到人物的蒙皮网格，未启用手机一体显示')
    body = max(candidates, key=lambda o:len(o.data.vertices))
    modifier = next((m for m in body.modifiers if m.type == 'NODES' and m.node_group
                     and m.node_group.get('sam_phone_display')),None)
    sources = sorted((o for o in anchor.children[0].children_recursive
                      if o.type in {'MESH','CURVE','FONT'}),key=lambda o:o.name)
    names = [o.name for o in sources]
    for mat in {slot.material for obj in sources for slot in obj.material_slots if slot.material}:
        if mat.name.startswith(('手机 /', '玲手机 /')):
            ensure_local_color(mat)
    if modifier is None:
        modifier = body.modifiers.new(DISPLAY,'NODES')
    group = modifier.node_group
    if group is None or list(group.get('sam_phone_sources',[])) != names:
        group = bpy.data.node_groups.new(DISPLAY+' / '+rig.name,'GeometryNodeTree')
        group.interface.new_socket(name='Geometry',in_out='INPUT',socket_type='NodeSocketGeometry')
        group.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
        incoming = group.nodes.new('NodeGroupInput')
        output = group.nodes.new('NodeGroupOutput')
        join = group.nodes.new('GeometryNodeJoinGeometry')
        incoming.location,join.location,output.location = (-500,200),(100,200),(320,200)
        group.links.new(incoming.outputs['Geometry'],join.inputs['Geometry'])
        for index,obj in enumerate(sources):
            node = group.nodes.new('GeometryNodeObjectInfo')
            node.transform_space = 'RELATIVE'
            node.inputs['Object'].default_value = obj
            node.inputs['As Instance'].default_value = False
            node.location = (-300,-index*90)
            group.links.new(node.outputs['Geometry'],join.inputs['Geometry'])
        group.links.new(join.outputs['Geometry'],output.inputs['Geometry'])
        group['sam_phone_sources'] = names
        group['sam_phone_display'] = True
        modifier.node_group = group
    ensure_phone_outline(group)
    if not group.get('sam_phone_frame_visibility'):
        socket = group.interface.new_socket(name='按帧显示手机', in_out='INPUT', socket_type='NodeSocketBool')
        socket.default_value = False
        incoming = next(n for n in group.nodes if n.type == 'GROUP_INPUT')
        output = next(n for n in group.nodes if n.type == 'GROUP_OUTPUT')
        join = output.inputs['Geometry'].links[0].from_node
        phone = group.nodes.new('GeometryNodeJoinGeometry')
        phone.name = '手机 / 部件与描边'
        for link in list(join.inputs['Geometry'].links):
            if link.from_node != incoming:
                source = link.from_socket
                group.links.remove(link)
                group.links.new(source, phone.inputs['Geometry'])
        switch = group.nodes.new('GeometryNodeSwitch')
        switch.input_type = 'GEOMETRY'
        switch.name = '手机 / 帧区间显隐'
        group.links.new(incoming.outputs[socket.identifier], switch.inputs['Switch'])
        group.links.new(phone.outputs['Geometry'], switch.inputs['True'])
        group.links.new(switch.outputs['Output'], join.inputs['Geometry'])
        phone.location, switch.location = (100, -250), (320, -100)
        join.location, output.location = (550, 200), (770, 200)
        group['sam_phone_frame_visibility'] = True
    if list(body.modifiers)[-1] != modifier:
        with bpy.context.temp_override(object=body,active_object=body):
            bpy.ops.object.modifier_move_to_index(modifier=modifier.name,index=len(body.modifiers)-1)
    return body,modifier,sources


def sync(scene):
    from sam_ellen_phone_ranges import configured_side, install_timing, migrate
    migrate(scene)
    side = configured_side(scene)
    for anchor in [o for o in scene.objects if o.get('sam_phone_rig')]:
        selected = scene.sam_ellen.rig
        enabled = bool(side and selected and selected.name == anchor['sam_phone_rig'])
        rig = scene.objects.get(anchor['sam_phone_rig'])
        if rig:
            bind_character(scene, rig, anchor, side if enabled else anchor.get('sam_phone_side', 'L'))
        sources = set()
        if rig:
            body,display,objects = ensure_display(rig,anchor)
            display.show_viewport = display.show_render = enabled
            sources = set(objects)
        stack = [anchor]
        while stack:
            obj = stack.pop()
            obj.hide_viewport = not enabled
            obj.hide_render = not enabled or obj in sources
            if obj in sources:
                # Object Info still evaluates these editable sources. Their
                # geometry is drawn once, as part of the character's mesh.
                for layer in scene.view_layers:
                    if obj.name in layer.objects:
                        obj.hide_set(True,view_layer=layer)
            stack.extend(obj.children)
        if rig:
            install_timing(scene, rig, display, enabled)
    scene.update_tag()


class SAMCharacterLocalView(bpy.types.Operator):
    bl_idname = 'sam_ellen.character_local_view'
    bl_label = '局部视图（人物与手机）'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.area and context.area.type == 'VIEW_3D' and bpy.ops.view3d.localview.poll()

    def execute(self, context):
        return bpy.ops.view3d.localview(frame_selected=False)


class SAMPhoneFit(bpy.types.Operator):
    bl_idname = 'sam_ellen.phone_fit'
    bl_label = '创建 / 重新贴合持握手机'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            anchor = ensure(context.scene)
            root = anchor.children[0]
            root.matrix_local = Matrix(root['sam_phone_offset'])
            side = context.scene.sam_ellen.phone_side
            label = '左' if side == 'L' else '右'
            self.report({'INFO'}, f'手机已跟随{label}手；双臂继续动捕')
            return {'FINISHED'}
        except Exception as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}


class SAMSelectCharacter(bpy.types.Operator):
    bl_idname = 'sam_ellen.select_phone_character'
    bl_label = '选择人物整体（人物＋手机）'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        rig = context.scene.sam_ellen.rig
        if not rig:
            return {'CANCELLED'}
        root = rig.parent or rig
        for obj in context.selected_objects:
            obj.select_set(False)
        root.hide_set(False)
        root.select_set(True)
        context.view_layer.objects.active = root
        self.report({'INFO'}, '已选择人物整体，可按 G 移动或 R 旋转')
        return {'FINISHED'}


CLASSES = (SAMPhoneFit, SAMSelectCharacter, SAMCharacterLocalView)

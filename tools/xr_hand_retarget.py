"""Length-preserving arm IK for head-relative XR hand targets."""
import math
from urllib.parse import unquote, urlparse
from mathutils import Matrix, Vector


def rotate_about(rotation, origin):
    return Matrix.Translation(origin) @ rotation.to_matrix().to_4x4() @ Matrix.Translation(-origin)


def solve_arm(rig, upper, lower, wrist, target, pole=None):
    """Change pose rotations only; keep rest bones, scale and translations intact."""
    a, b, c = upper.head.copy(), lower.head.copy(), wrist.head.copy()
    l1, l2 = (b-a).length, (c-b).length
    direction = target-a
    distance = max(abs(l1-l2)+1e-5, min(l1+l2-1e-5, direction.length))
    if direction.length < 1e-6:
        return
    direction.normalize()
    pole = Vector(pole) if pole is not None else b-a
    pole -= direction * pole.dot(direction)
    if pole.length < 1e-5:
        pole = direction.cross(Vector((0, 1, 0)))
    if pole.length < 1e-5:
        pole = direction.cross(Vector((1, 0, 0)))
    pole.normalize()
    along = (l1*l1-l2*l2+distance*distance)/(2*distance)
    mid = a + direction*along + pole*math.sqrt(max(0, l1*l1-along*along))
    end = a + direction*distance
    r1 = (b-a).rotation_difference(mid-a)
    d1 = rotate_about(r1, a)
    r2 = (r1 @ (c-b)).rotation_difference(end-mid)
    d2 = rotate_about(r2, mid)
    transforms = [(upper, d1 @ upper.matrix, upper.parent.matrix),
                  (lower, d2 @ d1 @ lower.matrix, d1 @ lower.parent.matrix)]
    # Preserve the hand's world orientation while the elbow moves.
    hand_matrix = wrist.matrix.copy()
    hand_matrix.translation = end
    transforms.append((wrist, hand_matrix, d2 @ d1 @ wrist.parent.matrix))
    for bone, desired, parent_matrix in transforms:
        local = bone.bone.convert_local_to_pose(desired, bone.bone.matrix_local,
            parent_matrix=parent_matrix, parent_matrix_local=bone.parent.bone.matrix_local,
            invert=True)
        bone.rotation_quaternion = local.to_quaternion()


def head_relative_target(rig, names, source, side):
    c = Matrix(((1, 0, 0), (0, 0, -1), (0, 1, 0)))
    head = names['頭']
    upper, lower, wrist = (names[side + n] for n in ('腕', 'ひじ', '手首'))
    source_length = sum((Vector(source[side+b]['origin']) - Vector(source[side+a]['origin'])).length
                        for a,b in [('腕', 'ひじ'), ('ひじ', '手首')])
    target_length = (lower.bone.head_local-upper.bone.head_local).length + (wrist.bone.head_local-lower.bone.head_local).length
    offset = c @ (Vector(source[side+'手首']['displayed']) - Vector(source['頭']['displayed']))
    return head.head + offset * (target_length/source_length)


def mouth_anchor(rig, mesh, head):
    """Locate the existing mouth from its vowel shape, without editing geometry."""
    if not mesh or not mesh.data.shape_keys:
        return None
    keys = mesh.data.shape_keys.key_blocks
    opened = keys.get('あ')
    if opened is None:
        return None
    basis = keys[0]
    points = [base.co for base, moved in zip(basis.data, opened.data)
              if (base.co-moved.co).length_squared > 1e-8]
    if not points:
        return None
    rest = rig.matrix_world.inverted() @ mesh.matrix_world @ (sum(points, Vector())/len(points))
    return head.bone.matrix_local.inverted() @ rest


def smooth_gate(value, inner, outer):
    t = max(0.0, min(1.0, (outer-value)/(outer-inner)))
    return t*t*(3-2*t)


CONTACT_PROPERTIES = ('contact_video', 'contact_fade', 'contact_strength',
    'contact_mouth', 'contact_mouth_side', 'contact_mouth_start', 'contact_mouth_end',
    'contact_mouth_height', 'contact_mouth_depth', 'contact_head', 'contact_head_side',
    'contact_head_start', 'contact_head_end', 'contact_head_height', 'contact_head_depth',
    'contact_head_side_offset')


def contact_signature(settings):
    return tuple(getattr(settings, name) for name in CONTACT_PROPERTIES)


def contact_segments(settings, packet):
    """Explicit source-video contact windows, faded in source time, not scene time."""
    if packet.get('source') != 'video' or not settings.contact_video:
        return []
    def normalize(path):
        path = path.replace('\\', '/')
        if path.lower().startswith('file:'):
            path = urlparse(path).path
        return unquote(path).lstrip('/').casefold()
    # Full path matching avoids applying a previous clip's timing to another video.
    if normalize(packet.get('source_video', '')) != normalize(settings.contact_video):
        return []
    timestamp = packet.get('time')
    if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
        return []
    rows = []
    for kind in ('mouth', 'head'):
        if not getattr(settings, 'contact_'+kind):
            continue
        start, end = (getattr(settings, 'contact_'+kind+'_'+p) for p in ('start', 'end'))
        if end <= start or not start < timestamp < end:
            continue
        fade = min(settings.contact_fade, (end-start)/2)
        weight = settings.contact_strength
        if fade > 0:
            t = min(1., (timestamp-start)/fade, (end-timestamp)/fade)
            weight *= t*t*(3-2*t)
        rows.append((kind, getattr(settings, 'contact_'+kind+'_side'), weight))
    # One target per hand; overlapping explicit windows prefer head contact.
    return list({side: (kind, side, weight) for kind, side, weight in rows}.values())


def head_contact_anchor(rig, head):
    collider = next((o for o in rig.children_recursive if o.name == 'Ellen_Collider_Head'), None)
    if collider:
        to_rest = head.bone.matrix_local @ head.matrix.inverted() @ rig.matrix_world.inverted()
        points = [to_rest @ collider.matrix_world @ Vector(corner)
                  for corner in collider.bound_box]
        center = sum(points, Vector())/len(points)
        center.z = max(p.z for p in points)
    else:
        center = head.bone.head_local + Vector((0, 0, head.bone.length))
    return head.bone.matrix_local.inverted() @ center


def correct_contact_hands(rig, names, mouth, crown, settings, packet, segments):
    """Solve artist-specified contact intent; retain local finger motion and arm lengths."""
    result = {}
    head = names.get('頭')
    if head is None:
        return result
    rotation = head.matrix.to_quaternion() @ head.bone.matrix_local.to_quaternion().conjugated()
    for kind, side, weight in segments:
        required = [side+p for p in ('腕', 'ひじ', '手首', '人指３')]
        if not all(n in names for n in required) or weight <= 0:
            continue
        anchor = mouth if kind == 'mouth' else crown
        if anchor is None:
            continue
        upper, lower, wrist, tip = (names[n] for n in required)
        row = (packet.get('mouth_hands') or {}).get(side, {})
        source = Vector(row.get('tip_from_mouth', (0, 0, 0)))
        # Keep small lateral variations instead of pinning the hand to one point.
        if kind == 'mouth':
            offset = Vector((max(-.025, min(.025, source.x*.08)),
                             -settings.contact_mouth_depth, settings.contact_mouth_height))
        else:
            sign = 1 if side == '左' else -1
            offset = Vector((sign*settings.contact_head_side_offset + max(-.012, min(.012, source.x*.05)),
                             settings.contact_head_depth, settings.contact_head_height))
        goal = head.matrix @ anchor + rotation @ offset
        solve_arm(rig, upper, lower, wrist, wrist.head + (goal-tip.head)*weight)
        result[side] = {'target': kind, 'weight': weight}
    return result


def correct_mouth_hands(rig, names, anchor, payload, strength, depth, height=.025):
    """Near-mouth correction only; source positions are normalized head-local XYZ."""
    if anchor is None or not payload:
        return {}
    head = names.get('頭')
    if head is None:
        return {}
    result = {}
    for side, row in payload.items():
        if side not in ('左', '右'):
            continue
        required = [side+p for p in ('腕', 'ひじ', '手首', '人指３')]
        if not all(name in names for name in required):
            continue
        # Deliberately narrow: no correction for a hand outside the mouth area,
        # or fingers pointing toward the camera rather than raised along the face.
        weight = strength * smooth_gate(row['screen_distance'], .18, .75)
        weight *= 1-smooth_gate(row['finger_up'], .18, .5)
        if weight < 1e-4:
            continue
        upper, lower, wrist, tip = (names[name] for name in required)
        length = (lower.bone.head_local-upper.bone.head_local).length
        length += (wrist.bone.head_local-lower.bone.head_local).length
        delta = Vector(row['tip_from_mouth']) * length
        # Single-camera depth is ambiguous. Compress only near-mouth forward
        # distance; a positive clearance prevents the target entering the face.
        delta.y = min(-length*.08, delta.y*depth)
        delta.z += height
        mouth = head.matrix @ anchor
        head_rotation = head.matrix.to_quaternion() @ head.bone.matrix_local.to_quaternion().conjugated()
        goal_tip = mouth + head_rotation @ delta
        target = wrist.head + (goal_tip-tip.head)*weight
        solve_arm(rig, upper, lower, wrist, target)
        result[side] = weight
    return result

"""Keep Ellen's original raised shoulder frills attached to the shoulders.

An extra Arm follow folds the frills down. Disable that grant on both sides;
the original shoulder parents still carry the garment with the body.
"""
import math
from pathlib import Path


def supported(rig):
    return bool(rig and rig.type == 'ARMATURE' and all(
        f'Shoulder饰{i}_{side}' in rig.pose.bones for side in ('L', 'R') for i in (1, 2, 3)))


def protected_file(filepath):
    if not filepath:
        return False
    path = Path(filepath).resolve()
    protected = Path.home()/'Desktop/角色场景'
    return path == protected or protected in path.parents


def consistent(rig):
    if not supported(rig):
        return False
    for side in ('L', 'R'):
        for i in (1, 2, 3):
            bone = rig.pose.bones[f'Shoulder饰{i}_{side}']
            m = bone.mmd_bone
            if (not bone.parent or bone.parent.name != 'Shoulder_'+side
                    or m.has_additional_rotation or m.has_additional_location
                    or m.additional_transform_bone
                    or any(bone.constraints.get(name) for name in
                           ('mmd_additional_rotation', 'mmd_additional_location'))):
                return False
    return True


def repair(rig):
    names = [f'Shoulder饰{i}_{side}' for side in ('L', 'R') for i in (1, 2, 3)]
    if not supported(rig):
        raise ValueError('This repair requires the Ellen shoulder ornament bones')
    if rig.library or rig.data.library:
        raise ValueError('衣领来自链接骨架，请在本地模型副本中修复')
    # Accept both the original asymmetric import and the previous v2 repair.
    # Validate the whole set first; never partly change an unknown custom rig.
    for name in names:
        bone = rig.pose.bones[name]
        m = bone.mmd_bone
        if (not bone.parent or bone.parent.name != 'Shoulder_'+name[-1]
                or m.additional_transform_bone not in ('', 'Arm_'+name[-1])
                or m.has_additional_location
                or not math.isfinite(m.additional_transform_influence)):
            raise ValueError('衣领存在未验证的自定义绑定，未自动覆盖')
        for cname in ('mmd_additional_rotation', 'mmd_additional_location'):
            c = bone.constraints.get(cname)
            if c and (c.type != 'TRANSFORM' or c.target != rig
                      or c.subtarget != '_shadow_'+name):
                raise ValueError('衣领存在未验证的自定义约束，未自动覆盖')
    if consistent(rig):
        return []
    changes = []
    for name in names:
        bone = rig.pose.bones[name]
        m = bone.mmd_bone
        old = (m.has_additional_rotation, m.additional_transform_bone,
               m.additional_transform_influence)
        # Remove only the extra upper-arm grant, including its MMD metadata so
        # rebuilding MMD constraints cannot bring the drooping rule back.
        m.has_additional_rotation = False
        m.has_additional_location = False
        m.additional_transform_bone = ''
        for cname in ('mmd_additional_rotation', 'mmd_additional_location'):
            constraint = bone.constraints.get(cname)
            if constraint:
                bone.constraints.remove(constraint)
        changes.append({'bone': name, 'before': old, 'after': (False, '')})
    # Unused non-deforming MMD helpers can remain for old capture baselines.
    # No bone/rest geometry, weights or animation curves need to be rewritten.
    rig['ellen_collar_binding_version'] = 3
    if not consistent(rig):
        raise RuntimeError('衣领绑定重建未通过验证')
    return changes


def ensure_scene(scene):
    """Repair loaded local Ellen rigs in memory. Never save a file here."""
    import bpy
    if protected_file(bpy.data.filepath):
        return {'status': 'protected', 'changes': []}
    rig = getattr(getattr(scene, 'sam_ellen', None), 'rig', None)
    if not supported(rig):
        return {'status': 'unsupported', 'changes': []}
    changes = repair(rig)
    scene.view_layers[0].update()
    return {'status': 'repaired' if changes else 'ready', 'changes': changes}


if __name__ == '__main__':
    import bpy
    rig = bpy.data.objects.get('艾莲_arm')
    if rig is None:
        raise ValueError('No Ellen rig in the current file')
    print(repair(rig))

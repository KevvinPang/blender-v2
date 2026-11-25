"""Action and bake compatibility for Blender 3.6 and newer."""
import bpy


def curves(obj, action=None):
    action = action or (obj.animation_data.action if obj.animation_data else None)
    if not action:
        return []
    if getattr(action, 'is_action_layered', False):
        slot = obj.animation_data.action_slot if obj.animation_data else None
        if slot not in list(action.slots):
            slot = next(iter(action.slots), None)
        if not slot:
            return []
        for layer in action.layers:
            for strip in layer.strips:
                bag = strip.channelbag(slot)
                if bag:
                    return bag.fcurves
        return []
    return action.fcurves


def set_selected(pb, selected):
    if bpy.app.version >= (5, 2, 0):
        pb.select = selected
    else:
        pb.bone.select = selected


def is_selected(pb):
    return pb.select if bpy.app.version >= (5, 2, 0) else pb.bone.select


def assign_action(obj, action, slot_identifier=''):
    ad = obj.animation_data_create()
    ad.action = action
    if action and hasattr(ad, 'action_slot'):
        slot = next((s for s in action.slots if s.identifier == slot_identifier), None)
        if slot:
            ad.action_slot = slot


def slot_id(obj):
    ad = obj.animation_data
    return ad.action_slot.identifier if ad and getattr(ad, 'action_slot', None) else ''


def add_constant_key(obj, bone, prop, value, start, end):
    pb = obj.pose.bones[bone]
    path = pb.path_from_id() + '["' + prop + '"]'
    cs = curves(obj)
    for fc in list(cs):
        if fc.data_path == path:
            cs.remove(fc)
    pb[prop] = value
    for f in sorted(set([start, end])):
        obj.keyframe_insert(data_path=path, frame=f, group='MMR / IK-FK')
    for fc in curves(obj):
        if fc.data_path == path:
            for k in fc.keyframe_points:
                k.interpolation = 'CONSTANT'


def bake_controls(obj, names, start, end, step=1, action=None, custom_props=False, visual=True, visual_overrides=()):
    from bpy_extras import anim_utils
    selection = {p.name: is_selected(p) for p in obj.pose.bones}
    try:
        for p in obj.pose.bones:
            set_selected(p, p.name in names)
        kwargs = dict(only_selected=True, do_pose=True, do_object=False,
            do_visual_keying=visual, do_constraint_clear=False, do_parents_clear=False,
            do_clean=False)
        if hasattr(anim_utils,'BakeOptions'):
            options=anim_utils.BakeOptions(**kwargs,do_location=True,do_rotation=True,do_scale=True,
                                           do_bbone=False,do_custom_props=custom_props)
            kwargs={'bake_options':options}
        frames = list(range(start, end + 1, step))
        if frames[-1] != end:
            frames.append(end)
        iterator=anim_utils.bake_action_objects_iter([(obj,action)],**kwargs)
        iterator.send(None);samples={n:[] for n in visual_overrides}
        for f in frames:
            iterator.send(f)
            for n in samples:
                pb=obj.pose.bones[n]
                samples[n].append(obj.convert_space(pose_bone=pb,matrix=pb.matrix,from_space='POSE',to_space='LOCAL').copy())
        baked=iterator.send(None)[0]
        if samples:write_visual_samples(obj,frames,samples)
        return baked
    finally:
        for p in obj.pose.bones:
            set_selected(p, selection[p.name])


def write_visual_samples(obj,frames,samples):
    """Fold modified FK controls into the action, leaving solved IK inputs alone."""
    cs=curves(obj)
    for n,matrices in samples.items():
        pb=obj.pose.bones[n];previous=None;channels={}
        for f,m in zip(frames,matrices):
            loc,q,scale=m.decompose()
            if pb.rotation_mode=='QUATERNION':
                if previous is not None:q.make_compatible(previous)
                previous=q.copy();rotation=list(q);rotpath='rotation_quaternion'
            elif pb.rotation_mode=='AXIS_ANGLE':rotation=[q.angle,*q.axis];rotpath='rotation_axis_angle'
            else:
                e=q.to_euler(pb.rotation_mode,previous) if previous is not None else q.to_euler(pb.rotation_mode)
                previous=e.copy();rotation=list(e);rotpath='rotation_euler'
            for field,values in [('location',loc),(rotpath,rotation),('scale',scale)]:
                for i,val in enumerate(values):channels.setdefault((pb.path_from_id()+'.'+field,i),[]).append((f,val))
        for (path,i),values in channels.items():
            fc=next((fc for fc in cs if fc.data_path==path and fc.array_index==i),None)
            if fc is None:fc=cs.new(path,index=i)
            points={round(p.co.x,6):p for p in fc.keyframe_points}
            for f,val in values:
                p=points.get(f)
                if p is None:p=fc.keyframe_points.insert(f,val,options={'FAST'})
                else:p.co.y=val
            fc.update()

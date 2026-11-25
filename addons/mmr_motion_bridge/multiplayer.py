"""Coordinate independent retarget sessions on one Blender timeline."""

import math
import re

import bpy

from . import baking, compat, layers, mapping, runtime


_FLOWHMR_RIG = re.compile(r'^FlowHMR Person (\d+)(?:\.\d+)?$')
_RENDERABLE = {'MESH','CURVE','CURVES','SURFACE','FONT','META','POINTCLOUD','VOLUME','GREASEPENCIL'}


def source_span(obj):
    if not obj or not obj.animation_data or not obj.animation_data.action:
        raise ValueError('源骨架需要有已导入的动作。')
    # An action may have several slots. Read this rig's keys, not the union
    # of every slot and not the viewport playback range.
    curves=compat.curves(obj)
    spans=[tuple(fc.range()) for fc in curves if fc.data_path.startswith('pose.bones[')
           and (len(fc.keyframe_points) or len(fc.sampled_points))]
    if not spans:
        raise ValueError(f'{obj.name} 的动作没有骨骼关键帧。')
    first=math.ceil(min(span[0] for span in spans)-1e-4)
    last=math.floor(max(span[1] for span in spans)+1e-4)
    if last<first:raise ValueError(f'{obj.name} 的动作没有完整帧。')
    return first,last


def discover_sources(scene):
    sources=[]
    for obj in scene.objects:
        if obj.type!='ARMATURE' or mapping.is_mmr(obj) or not mapping.auto_map(obj).get('spine_fk'):
            continue
        try:source_span(obj)
        except ValueError:continue
        sources.append(obj)
    named=[(obj,_FLOWHMR_RIG.fullmatch(obj.name)) for obj in sources]
    combined=[(obj,int(m.group(1))) for obj,m in named if m]
    if combined:return [obj for obj,_ in sorted(combined,key=lambda item:(item[1],item[0].name))]
    return sorted(sources,key=lambda obj:obj.name)


def actor_objects(scene,cfg):
    """Only this character's controllers, skinned meshes and attached props."""
    rigs={rig for rig in (cfg.target,cfg.deform) if rig is not None}
    result=set(rigs)
    for obj in scene.objects:
        if obj.type not in _RENDERABLE:continue
        if any(mod.type=='ARMATURE' and mod.object in rigs for mod in obj.modifiers):
            result.add(obj);continue
        parent=obj.parent
        while parent is not None:
            if parent in rigs:
                result.add(obj);break
            parent=parent.parent
    return result


def _has_visibility_animation(obj,path):
    ad=obj.animation_data
    return ad is not None and (ad.drivers.find(path) is not None
                               or any(fc.data_path==path for fc in compat.curves(obj))
                               or any(track.strips for track in ad.nla_tracks))


def clear_visibility(cfg):
    for link in cfg.visibility:
        obj=link.obj
        if not obj:continue
        if link.modifier_name:
            modifier=obj.modifiers.get(link.modifier_name)
            if modifier and modifier.type=='MASK' and modifier.vertex_group==link.vertex_group:
                for path in ('show_viewport','show_render'):modifier.driver_remove(path)
                obj.modifiers.remove(modifier)
                group=obj.vertex_groups.get(link.vertex_group)
                if group:obj.vertex_groups.remove(group)
            continue
        if not obj.animation_data:continue
        driver=obj.animation_data.drivers.find(link.path)
        if driver and driver.driver.expression==link.expression and not driver.driver.variables:
            obj.driver_remove(link.path)
            setattr(obj,link.path,False)
    cfg.visibility.clear()


def _add_mesh_mask(cfg,obj,expression):
    # An empty group makes the evaluated mesh empty only while the mask is on.
    # Unlike object visibility, it doesn't remove the rig from the depsgraph.
    entry=cfg.visibility.add()
    entry.obj=obj
    group=obj.vertex_groups.new(name='MMRB 入场')
    entry.vertex_group=group.name
    modifier=obj.modifiers.new(name='MMRB 入场',type='MASK')
    entry.modifier_name=modifier.name
    modifier.vertex_group=group.name
    for path in ('show_viewport','show_render'):modifier.driver_add(path).driver.expression=expression


def _add_visibility(cfg,objects):
    # Pure frame expressions persist with the baked scene and do not require
    # the source action or a custom script to be enabled during rendering.
    expression=f'frame < {cfg.frame_start} or frame > {cfg.frame_end}'
    for obj in sorted(objects,key=lambda ob:ob.name):
        if obj.library is not None:continue
        # Older dependency graphs can crash when an object is disabled by
        # a driver mid-bake. Animate a mesh mask instead on those versions.
        if bpy.app.version < (5,2,0):
            if obj.type=='MESH':_add_mesh_mask(cfg,obj,expression)
            continue
        paths=('hide_viewport','hide_render') if obj.type in _RENDERABLE else ('hide_viewport',)
        for path in paths:
            if getattr(obj,path) or _has_visibility_animation(obj,path):continue
            entry=cfg.visibility.add()
            entry.obj,entry.path,entry.expression=obj,path,expression
            obj.driver_add(path).driver.expression=expression


def _preflight(scene):
    group=scene.mmrb_multi
    if scene.mmrb.status!='NONE':
        raise ValueError('当前场景还有单人会话，请先拆除单人会话。')
    pairs=list(group.pairs)
    if not pairs:raise ValueError('请先添加源骨架并为它们选择目标。')
    sources,targets,owned=set(),set(),set()
    prepared=[]
    for index,cfg in enumerate(pairs,1):
        name=cfg.source.name if cfg.source else f'配对 {index}'
        try:
            if not cfg.source or not cfg.target:raise ValueError('请设置源骨架和目标控制骨架。')
            if cfg.source in sources:raise ValueError('这副源骨架已经用于另一配对。')
            if cfg.target in targets:raise ValueError('目标控制骨架已经用于另一配对。')
            if cfg.deform and cfg.deform.name not in scene.objects:raise ValueError('变形骨架不在当前场景。')
            if cfg.status=='NONE':cfg.frame_start,cfg.frame_end=source_span(cfg.source)
            allowed=layers.read_state(cfg).get('switch_drivers',()) if cfg.helpers else ()
            runtime.validate(scene,cfg,allowed)
            objects=actor_objects(scene,cfg)
            if owned & objects:raise ValueError('目标角色与另一配对共用骨架或网格。')
        except (ValueError,KeyError) as error:
            raise ValueError(f'{name}：{error}') from error
        sources.add(cfg.source);targets.add(cfg.target);owned.update(objects)
        prepared.append((cfg,objects))
    if sources & owned:raise ValueError('源骨架不能同时作为多人配对的目标角色。')
    return prepared


def _expand_timeline(scene,group,prepared):
    prior=(scene.frame_start,scene.frame_end)
    wider=(min(prior[0],*(cfg.frame_start for cfg,_ in prepared)),
           max(prior[1],*(cfg.frame_end for cfg,_ in prepared)))
    if wider!=prior:
        if not group.changed_range:group.original_range=prior
        group.changed_range=True
        group.expanded_range=wider
        scene.frame_start,scene.frame_end=wider


def _start_missing(scene,prepared):
    created=[]
    try:
        for cfg,objects in prepared:
            if cfg.status!='NONE':continue
            runtime.start(scene,cfg)
            created.append(cfg)
            _add_visibility(cfg,objects)
    except Exception:
        for cfg in reversed(created):
            clear_visibility(cfg)
            runtime.remove(scene,cfg)
        raise


def preview_all(scene):
    prepared=_preflight(scene)
    _start_missing(scene,prepared)
    for cfg,_ in prepared:
        if cfg.status in {'PAUSED','BAKED'}:baking.resume(scene,cfg)
    _expand_timeline(scene,scene.mmrb_multi,prepared)
    scene.frame_set(scene.frame_current,subframe=scene.frame_subframe)
    return len(prepared)


def bake_all(scene):
    prepared=_preflight(scene)
    _start_missing(scene,prepared)
    _expand_timeline(scene,scene.mmrb_multi,prepared)
    baked=sum(cfg.status=='BAKED' for cfg,_ in prepared)
    try:
        for cfg,_ in prepared:
            if cfg.status=='BAKED':continue
            try:
                with runtime.focus(cfg):baking.bake(scene,cfg)
            except Exception as error:
                raise RuntimeError(f'已烘焙 {baked}/{len(prepared)} 人；{cfg.source.name}：{error}') from error
            baked+=1
    finally:
        scene.frame_set(scene.frame_current,subframe=scene.frame_subframe)
    return baked


def remove_all(scene):
    group=scene.mmrb_multi
    for cfg in reversed(list(group.pairs)):
        if cfg.status!='NONE':
            clear_visibility(cfg)
            runtime.remove(scene,cfg)
    if group.changed_range and tuple(group.expanded_range)==(scene.frame_start,scene.frame_end):
        scene.frame_start,scene.frame_end=group.original_range
    group.changed_range=False
    scene.frame_set(scene.frame_current,subframe=scene.frame_subframe)

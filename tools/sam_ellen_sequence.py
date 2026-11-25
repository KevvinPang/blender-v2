"""Append SAM takes without changing preceding keys or the scene timebase."""
import json
import math

import bpy


def curves(action, slot=None):
    if not action:
        return []
    return [curve for layer in action.layers for strip in layer.strips
            for bag in strip.channelbags
            if slot is None or bag.slot_handle == slot.handle
            for curve in bag.fcurves]


def timeline_frame(data, seconds):
    timing = data.get('_sam_timeline')
    rate = timing['fps'] if timing else data['fps']
    start = timing['start'] if timing else 1
    return start + (seconds-data['frames'][0]['time'])*rate


def validate_animation(ad):
    if not ad:
        return
    if any(not track.mute for track in ad.nla_tracks):
        raise ValueError('追加前请先将人物和面部的 NLA 轨道静音')
    if ad.action and any(c.modifiers or c.mute for c in curves(ad.action, ad.action_slot)):
        raise ValueError('当前动作有曲线修改器或静音曲线，请先烘焙后再追加')


def make_plan(scene, rig, mesh, settings, reapply=False):
    ad = rig.animation_data
    current = ad.action if ad else None
    previous = json.loads(current.get('sam_sequence', 'null')) if current else None
    if reapply and previous:
        if current != settings.take:
            raise ValueError('人物动作已切换，请先选择最后导入的动作再重新适配')
        plan = previous.copy()
        plan['fps'] = scene.render.fps / scene.render.fps_base
        if abs(plan['fps']-previous['fps']) > 1e-5:
            raise ValueError('追加后帧率已更改，请恢复原帧率再重新适配')
    elif reapply or settings.import_mode != 'APPEND' or not current:
        return None
    else:
        targets = [ad]
        # SAM body keys define the captured clip duration. Old expression
        # curves can extend far beyond it, even with all manual blocks disabled.
        if not current.get('sam_job_id') and mesh and mesh.data.shape_keys.animation_data:
            targets.append(mesh.data.shape_keys.animation_data)
        end = max((point.co.x for target in targets for c in curves(target.action, target.action_slot)
                   for point in c.keyframe_points), default=scene.frame_start-1)
        if end < scene.frame_start:
            return None
        prefix_end = math.ceil(end)
        plan = {'start': prefix_end+max(1, settings.transition_frames), 'prefix_end': prefix_end,
                'frame_start': scene.frame_start, 'frame_end': scene.frame_end,
                'fps': scene.render.fps/scene.render.fps_base,
                'render_fps': scene.render.fps, 'fps_base': scene.render.fps_base}
    validate_animation(ad)
    if mesh:
        validate_animation(mesh.data.shape_keys.animation_data)
    return plan


def merge_action(owner, base, base_slot, incoming, source_fps, plan, paths=None):
    """Copy the old action, adding only the new segment's controlled channels."""
    ad = owner.animation_data_create()
    new_slot = ad.action_slot
    additions = {(c.data_path, c.array_index): c for c in curves(incoming, new_slot)
                 if paths is None or c.data_path in paths}
    end = plan['prefix_end']
    start = plan['start']
    rate = plan['fps']/source_fps
    # Align quaternion signs across the boundary to avoid a q/-q interpolation flip.
    old = {(c.data_path, c.array_index): c for c in curves(base, base_slot)}
    signs = {}
    for path, index in additions:
        if not path.endswith('rotation_quaternion') or index != 0:
            continue
        if all((path, i) in old and (path, i) in additions for i in range(4)):
            dot = sum(old[path, i].evaluate(end)*additions[path, i].evaluate(1) for i in range(4))
            signs[path] = -1 if dot < 0 else 1
    result = base.copy() if base else bpy.data.actions.new('SAM Sequence')
    result.name = incoming.name+' Sequence'
    result.use_fake_user = True
    try:
        ad.action = result
        if base_slot:
            ad.action_slot = next(s for s in result.slots if s.identifier == base_slot.identifier)
        existing = {(c.data_path, c.array_index): c for c in curves(result, ad.action_slot)}
        for key, incoming_curve in additions.items():
            path, index = key
            curve = existing.get(key)
            if curve is None:
                curve = result.fcurve_ensure_for_datablock(owner, path, index=index)
            original = old.get(key)
            value = original.evaluate(end) if original else owner.path_resolve(path)
            if not isinstance(value, (int, float)):
                value = value[index]
            retained = []
            for point in curve.keyframe_points:
                if point.co.x <= end:
                    retained.append((point.co.x, tuple(point.handle_left), tuple(point.handle_right),
                                     point.interpolation))
            for i in range(len(curve.keyframe_points)-1, -1, -1):
                if curve.keyframe_points[i].co.x > end:
                    curve.keyframe_points.remove(curve.keyframe_points[i], fast=True)
            boundary = curve.keyframe_points.insert(end, value, options={'FAST'})
            boundary_interpolation = 'LINEAR' if start-end > 1 else 'CONSTANT'
            boundary.interpolation = boundary_interpolation
            for point in incoming_curve.keyframe_points:
                frame = start+(point.co.x-1)*rate
                new = curve.keyframe_points.insert(frame, point.co.y*signs.get(path, 1), options={'FAST'})
                new.interpolation = 'LINEAR'
            curve.update()
            # Adding points must not recalculate the prefix's automatic handles.
            by_frame = {p.co.x: p for p in curve.keyframe_points}
            for frame, left, right, interpolation in retained:
                point = by_frame[frame]
                point.handle_left_type = point.handle_right_type = 'FREE'
                point.handle_left, point.handle_right = left, right
                point.interpolation = interpolation
                if frame == end:
                    point.interpolation = boundary_interpolation
        for key, value in incoming.items():
            result[key] = value
        return result
    except BaseException:
        ad.action = incoming
        bpy.data.actions.remove(result, do_unlink=True)
        raise

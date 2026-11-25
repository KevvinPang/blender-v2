"""Non-destructive expression edits over captured facial animation."""
bl_info = {'name': 'Ellen Expression Timeline', 'author': 'Local project',
           'version': (1, 6, 0), 'blender': (5, 2, 0), 'category': 'Animation'}
import json
import math
import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty,
                       IntProperty, PointerProperty, StringProperty)
import xr_ellen_bridge as bridge
import belle_star_eyes as star_eyes
from ellen_expression_library import (BLUSH_CONTROL, DISGUST_CONTROL, GRIMACE_CONTROL, TEETH_PRESETS,
    MOUTH_PROFILES as EXTRA_MOUTH_PROFILES, ensure_blush_control, ensure_disgust_control,
    ensure_disgust_grimace, REFERENCE_CONTROLS, ensure_reference_disgust,
    PAIR_CONTROLS, ensure_disgust_pair, SIDE_GLANCE_CONTROL, ensure_reference_side_glance,
    SOFT_SMILE_CONTROL, ensure_reference_soft_smile, SELECTED_REFERENCE_PRESETS,
    BLANK_BROWS_CONTROL, ensure_reference_blank_brows, BELLE_REFERENCE_PRESET,
    BELLE_WEARY_BROWS, ensure_belle_weary_brows)
from ellen_expression_library import (ELLEN_CORE_PRESETS, ELLEN_CORE_CONTROLS,
    ELLEN_CORE_MOUTH_PROFILES, ELLEN_CORE_TEETH_PRESETS, ensure_ellen_core_controls)

TEETH_PRESETS = TEETH_PRESETS | ELLEN_CORE_TEETH_PRESETS
# These smiles show teeth only while the recorded mouth is open.
LAUGH_PRESETS = frozenset(('laugh', 'ellen_core_laugh'))

PRESETS = [(name, label, '') for name, (label, _) in bridge.FACE_PRESETS.items()]
SHARKBOO_PRESETS = {
    'neutral': ('自然 / 重置', {}),
    'sharkboo_half_closed': ('半眯眼', {'じと目': 1.0}),
    'sharkboo_cross_eyes': ('叉叉眼', {'XX': 1.0}),
    'sharkboo_squeezed_eyes': ('挤眼 ><', {'><': 1.0}),
    'sharkboo_worried': ('困惑', {'困る': 1.0}),
    'sharkboo_serious': ('严肃', {'真面目': 1.0}),
    'sharkboo_angry': ('生气', {'怒り': 1.0}),
    'sharkboo_blink': ('眨眼 / 闭眼', {'まばたき': 1.0}),
    'sharkboo_smile': ('笑眼', {'笑い': 1.0}),
}
SHARKBOO_FACE_NAMES = tuple(n for _, values in SHARKBOO_PRESETS.values() for n in values)
# Keep existing enum numbers stable so saved human expression tracks retain
# their meanings. Module-owned lists also keep callback strings alive for RNA.
STANDARD_PRESET_ITEMS = [(*item, i) for i, item in enumerate(PRESETS)]
BELLE_PRESETS = {**bridge.FACE_PRESETS, star_eyes.PRESET: (star_eyes.LABEL, star_eyes.VALUES)}
BELLE_PRESET_ITEMS = [*STANDARD_PRESET_ITEMS, (star_eyes.PRESET, star_eyes.LABEL, '', 3000)]
SHARKBOO_PRESET_ITEMS = [('neutral', '自然 / 重置', '', 0)] + [
    (name, label, '', len(PRESETS) + i)
    for i, (name, (label, _)) in enumerate(SHARKBOO_PRESETS.items()) if name != 'neutral']
ELLEN_CORE_PRESET_ITEMS = [
    (name, label, '', 0 if name == 'neutral' else 2 if name == 'ellen_core_smile' else 2000+i)
    for i, (name, (label, _)) in enumerate(ELLEN_CORE_PRESETS.items())]


def is_sharkboo(mesh):
    return bool(mesh and mesh.get('expression_profile') == 'sharkboo')


def is_ellen_core(mesh):
    return bool(mesh and mesh.get('expression_profile') == 'ellen_core_v1')


def presets_for(scene):
    mesh = face_object(scene)
    if is_ellen_core(mesh):
        return ELLEN_CORE_PRESETS
    if star_eyes.is_belle(mesh):
        return BELLE_PRESETS
    return SHARKBOO_PRESETS if is_sharkboo(mesh) else bridge.FACE_PRESETS


def controls_for(scene):
    mesh = face_object(scene)
    if is_ellen_core(mesh):
        return (*bridge.FACE_NAMES, *ELLEN_CORE_CONTROLS)
    if star_eyes.is_belle(mesh):
        return (*bridge.FACE_NAMES, *star_eyes.CONTROLS)
    return SHARKBOO_FACE_NAMES if is_sharkboo(mesh) else bridge.FACE_NAMES


def preset_items(self, context):
    if is_ellen_core(self.id_data):
        return ELLEN_CORE_PRESET_ITEMS
    if star_eyes.is_belle(self.id_data):
        return BELLE_PRESET_ITEMS
    return SHARKBOO_PRESET_ITEMS if is_sharkboo(self.id_data) else STANDARD_PRESET_ITEMS


VOWELS = {'あ', 'い', 'う', 'え', 'お', 'ワ', '口横広げ', '口横狭め'}
EMOTION = set(bridge.FACE_NAMES) - VOWELS
SPEECH_NAMES = ('あ', 'い', 'う', 'え', 'お', 'ワ')
# Fade competing eye shapes out during closure so a tilted or narrowed eye
# closes cleanly. Brows, teeth, mouth and cheek controls are intentionally absent.
BLINK_CONTROL = 'まばたき'
WINK_CONTROLS = {'LEFT': 'ウィンク２', 'RIGHT': 'ウィンク２右'}
WINK_CANCEL_PREFIX = 'ExpressionWinkCancel_'
WINK_CLOSE_PREFIX = 'ExpressionWinkClose_'
BLINK_SIDES = (('BOTH','双眼','两只眼睛同时眨眼'),
               ('LEFT','左眼','角色自身左眼；正面看位于画面右侧'),
               ('RIGHT','右眼','角色自身右眼；正面看位于画面左侧'))
BLINK_EYE_CONTROLS = (BLINK_CONTROL, 'ウィンク２右', 'ウィンク２', 'ウィンク右',
    'ウィンク', '笑い', 'びっくり右', 'びっくり左', 'じと目', '怒り目', '怒り目２',
    '悲しむ', '眼角下', REFERENCE_CONTROLS[0], SIDE_GLANCE_CONTROL,
    'Ellen_CoreDownGaze', 'Ellen_CoreStartleIris', 'Ellen_SoftSadEyeTilt',
    'Ellen_SoftSadEyeTailCurve', star_eyes.LEFT, star_eyes.RIGHT)
# Legacy opening gain/budget and the authored smiling-open blend. Captured speech
# now uses a unit gain/budget; keep these tuples compatible with preset packages.
MOUTH_PROFILES = {
    'neutral': (1.0, 0.0, 1.0), 'talk': (.95, 0.0, .90),
    'smile': (.85, .22, .80), 'smile_closed': (.85, .24, .80),
    'unimpressed': (.68, 0.0, .75), 'serious': (.88, 0.0, .85),
    'laugh': (.90, .55, .95), 'emphasis': (1.10, 0.0, 1.0),
    'brooding': (.72, 0.0, .80), 'worried': (.78, 0.0, .80),
}
MOUTH_PROFILES.update(EXTRA_MOUTH_PROFILES)
MOUTH_PROFILES[star_eyes.PRESET] = (1.0, .22, 1.0)
MOUTH_PROFILES.update({n: v for n, v in ELLEN_CORE_MOUTH_PROFILES.items() if n != 'neutral'})

def linked_mouth(preset, source, gain=1.0):
    opening_gain, smiling_mix, budget = MOUTH_PROFILES[preset]
    # Expression presets retain the vowel redistribution and mouth corners.
    # Captured speech controls opening: do not quietly shrink it a second time.
    opening_gain, budget = 1.0, 1.0
    result = {name: max(0.0, min(1.0, source.get(name, 0))) * opening_gain * gain
              for name in SPEECH_NAMES}
    # Redistribute open vowels into the existing smiling-open shape, rather than
    # adding a second full-strength mouth deformation on top of them.
    smile_open = (result['あ'] + result['え'] * .35) * smiling_mix
    result['あ'] *= 1 - smiling_mix
    result['え'] *= 1 - smiling_mix * .35
    result['ワ'] += smile_open
    total = sum(result.values())
    if total > budget:
        result = {name: value * budget / total for name, value in result.items()}
    for name in ('口横広げ', '口横狭め'):
        result[name] = min(1.0, max(0.0, min(1.0, source.get(name, 0))) * opening_gain * gain)
    result[bridge.TEETH_CONTROL] = float(preset in TEETH_PRESETS - LAUGH_PRESETS or (
        preset in LAUGH_PRESETS and sum(result[n] for n in SPEECH_NAMES) > .08))
    return result

def fps(scene):
    return scene.render.fps / scene.render.fps_base

def frame_at(scene, seconds):
    return scene.frame_start + round(seconds * fps(scene))

def seconds_at(scene):
    return max(0, (scene.frame_current - scene.frame_start) / fps(scene))

def action_curves(action, slot):
    if not action:
        return []
    return [curve for layer in action.layers for strip in layer.strips
            if strip.type == 'KEYFRAME'
            for bag in strip.channelbags if slot is None or bag.slot_handle == slot.handle
            for curve in bag.fcurves]

def has_speech_curves(keys, action, slot=None):
    if not action:
        return False
    paths = {keys.key_blocks[n].path_from_id('value') for n in SPEECH_NAMES if n in keys.key_blocks}
    return any(c.data_path in paths and len(c.keyframe_points) >= 2 for c in action_curves(action, slot))

def speech_source_available(scene):
    mesh = face_object(scene)
    if not mesh:
        return False
    settings = mesh.ellen_expression_track
    keys = mesh.data.shape_keys
    ad = keys.animation_data
    if not ad:
        return False
    source = settings.base_action if ad.action == settings.generated_action else ad.action
    return has_speech_curves(keys, source)

def face_object(scene):
    return scene.ellen_expression_object or bridge.face_mesh(scene)

def preset_values(scene, preset):
    definitions = presets_for(scene)
    if preset not in definitions:
        raise ValueError('此角色不支持所选表情，请重新选择')
    values = dict(definitions[preset][1])
    mesh = face_object(scene)
    if mesh:
        overrides = json.loads(mesh.get('expression_preset_overrides', '{}'))
        values.update(overrides.get(preset, {}))
    return values

def settings_for(scene, *, migrate=True):
    mesh = face_object(scene)
    if not mesh:
        raise ValueError('请先选择角色的面部网格')
    settings = mesh.ellen_expression_track
    if migrate:
        migrate_frames(scene, settings)
    return settings


def migrate_frames(scene, settings):
    """Convert legacy seconds once, preserving the old evaluated frame bounds."""
    pending = [item for item in settings.items if item.timing_version < 1]
    if not pending:
        return
    status = settings.status
    for item in pending:
        item.start_frame = frame_at(scene, item.start)
        item.end_frame = frame_at(scene, item.end)
        item.fade_frames = round(item.fade * fps(scene))
        item.timing_version = 1
    settings.status = status


def add_segment(scene, settings):
    start = max((i.end_frame for i in settings.items), default=scene.frame_current)
    item = settings.items.add()
    item.preset = ('neutral' if is_sharkboo(face_object(scene)) else
                   'ellen_core_smile' if is_ellen_core(face_object(scene)) else 'smile')
    if is_sharkboo(face_object(scene)):
        item.auto_mouth = False
    item.start_frame = start
    item.end_frame = start + max(1, round(2 * fps(scene)))
    item.fade_frames = round(.2 * fps(scene))
    item.timing_version = 1
    settings.index = len(settings.items) - 1
    return item

def migrate_scene(scene):
    if scene.get('EllenExpressionObjectMigration'):
        for obj in scene.objects:
            if obj.type == 'MESH' and obj.data.shape_keys:
                migrate_frames(scene, obj.ellen_expression_track)
        return
    mesh = bridge.face_mesh(scene)
    if not mesh:
        return
    old = scene.ellen_expressions
    new = mesh.ellen_expression_track
    if not new.items and not new.target:
        for prop in ('index', 'target', 'base_action', 'generated_action', 'base_values',
                     'base_slot', 'generated_slot', 'status', 'advanced'):
            setattr(new, prop, getattr(old, prop))
        for source in old.items:
            dest = new.items.add()
            for prop in ExpressionSegment.__annotations__:
                setattr(dest, prop, getattr(source, prop))
            dest.timing_version = source.timing_version
    if not scene.ellen_expression_object:
        scene.ellen_expression_object = mesh
    scene['EllenExpressionObjectMigration'] = True
    for obj in scene.objects:
        if obj.type == 'MESH' and obj.data.shape_keys:
            migrate_frames(scene, obj.ellen_expression_track)

def edit_available(scene):
    if bridge._runtime is not None:
        raise ValueError('请先结束录制并断开 XR 连接，再编辑表情')
    mesh = face_object(scene)
    if not mesh or mesh.type != 'MESH' or not mesh.data.shape_keys:
        raise ValueError('没有找到角色面部网格')
    if mesh.name not in scene.objects:
        raise ValueError('选择的角色不在当前场景')
    keys = mesh.data.shape_keys
    if sum(name in keys.key_blocks for name in controls_for(scene) if name not in VOWELS) < 3:
        raise ValueError('此角色的表情名称尚未适配，请先配置表情映射')
    if any(o != mesh and o.type == 'MESH' and o.data.shape_keys == keys for o in scene.objects):
        raise ValueError('面部网格与其他对象共享形态键，请先将网格设为单用户副本')
    if keys.animation_data and any(not t.mute for t in keys.animation_data.nla_tracks):
        raise ValueError('面部存在启用的 NLA 轨道，请先烘焙或静音后再编辑')
    return keys

def intervals(scene):
    result = []
    for item in settings_for(scene).items:
        if not item.enabled:
            continue
        start, end = item.start_frame, item.end_frame
        if end < start:
            raise ValueError('结束帧不能早于开始帧')
        result.append((start, end, item))
    result.sort(key=lambda row: row[0])
    for previous, current in zip(result, result[1:]):
        if current[0] < previous[1] or current[0] == previous[0]:
            raise ValueError('表情帧范围重叠，请调整起止帧')
    return result

def amount(item, frame, start, end, rate, adjacent_next=False):
    if item.transition == 'CUT':
        return item.intensity
    fade = min(item.fade_frames, (end - start) // 2)
    if fade <= 0:
        return item.intensity
    t = max(0, min(1, (frame - start) / fade,
                   1 if adjacent_next else (end - frame) / fade))
    return item.intensity * t * t * (3 - 2 * t)


def smooth_speech(values, rate, speed):
    """Symmetric filtering reduces rapid changes without moving the timeline.

    Speed 1 is the exact original. Smaller values remove short fluctuations;
    the kernel uses seconds so the control behaves the same at different FPS.
    Leading/trailing neutral samples stay neutral, including append boundaries.
    """
    import numpy as np
    data = np.asarray(values, dtype=float)
    if not math.isfinite(speed) or not .1 <= speed <= 1:
        raise ValueError('口型变化速度必须在 0.1 到 1 之间')
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError('场景帧率无效')
    if speed == 1 or len(data) < 2:
        return data.copy()
    sigma = .06 * (1 / speed - 1) * rate
    radius = max(1, math.ceil(3 * sigma))
    offsets = np.arange(-radius, radius+1)
    weights = np.exp(-.5 * (offsets / sigma)**2)
    weights /= weights.sum()
    result = np.convolve(np.pad(data, radius, mode='edge'), weights, mode='valid')
    active = np.flatnonzero(np.abs(data) > 1e-8)
    if len(active):
        result[:active[0]] = data[:active[0]]
        result[active[-1]+1:] = data[active[-1]+1:]
    return result


def blink_amount(item, frame):
    distance = frame - item.frame
    duration = item.close_frames if distance < 0 else item.open_frames
    t = max(0.0, 1.0 - abs(distance) / duration)
    return t*t*(3-2*t)


def split_curve_boundary(curve, frame):
    """Insert a boundary without changing the Bezier segment outside it."""
    from mathutils import Vector
    points = list(curve.keyframe_points)
    if any(abs(p.co.x-frame) < 1e-5 for p in points):
        return
    left = next((p for p in reversed(points) if p.co.x < frame), None)
    right = next((p for p in points if p.co.x > frame), None)
    value = curve.evaluate(frame)
    # Freeze existing handles before insertion, which otherwise recalculates
    # AUTO handles beyond the blink interval as well.
    saved = [(p, p.handle_left.copy(), p.handle_right.copy()) for p in points]
    handles = {p.co.x: [hl.copy(), hr.copy()] for p,hl,hr in saved}
    for p, hl, hr in saved:
        p.handle_left_type = p.handle_right_type = 'FREE'
        p.handle_left = hl; p.handle_right = hr
    incoming = outgoing = None
    interpolation = left.interpolation if left else 'LINEAR'
    if left and right and interpolation == 'BEZIER':
        a, b, c, d = (Vector(v) for v in
                      (left.co, left.handle_right, right.handle_left, right.co))
        # Blender scales crossing handle lengths to the segment's time span.
        span = d.x-a.x
        length = abs(b.x-a.x)+abs(d.x-c.x)
        if length > span:
            factor = span/length
            b = a+(b-a)*factor; c = d+(c-d)*factor
        lo, hi = 0.0, 1.0
        for _ in range(45):
            t = (lo+hi)*.5
            x = (1-t)**3*a.x+3*(1-t)**2*t*b.x+3*(1-t)*t*t*c.x+t**3*d.x
            if x < frame: lo = t
            else: hi = t
        t = (lo+hi)*.5
        ab = a.lerp(b,t); bc = b.lerp(c,t); cd = c.lerp(d,t)
        abc = ab.lerp(bc,t); bcd = bc.lerp(cd,t)
        middle = abc.lerp(bcd,t)
        handles[left.co.x][1] = ab; handles[right.co.x][0] = cd
        incoming, outgoing = abc, bcd
        value = middle.y
    point = curve.keyframe_points.insert(frame, value, options={'FAST'})
    point.interpolation = interpolation
    point.handle_left_type = point.handle_right_type = 'FREE'
    if incoming is not None:
        point.handle_left = incoming; point.handle_right = outgoing
    elif not left or not right:
        # Preserve constant or linear extrapolation on the exterior side.
        slope = 0.0
        if curve.extrapolation == 'LINEAR' and len(points) > 1:
            edge = points[0] if not left else points[-1]
            neighbor = points[1] if not left else points[-2]
            handle = edge.handle_left if not left else edge.handle_right
            if edge.interpolation == 'BEZIER' and abs(handle.x-edge.co.x) > 1e-6:
                slope = (handle.y-edge.co.y)/(handle.x-edge.co.x)
            else:
                slope = (neighbor.co.y-edge.co.y)/(neighbor.co.x-edge.co.x)
        point.handle_left = (frame-1, value-slope)
        point.handle_right = (frame+1, value+slope)
    # Inserting a key can shorten neighboring handles even in FAST mode.
    # Restore the exact split handles after the key exists.
    for p in curve.keyframe_points:
        if p.co.x in handles:
            p.handle_left_type = p.handle_right_type = 'FREE'
            p.handle_left, p.handle_right = handles[p.co.x]
    curve.update()


def wink_cancel_control(mesh, original, side, closing=False):
    """Cancel an existing eye deformation on just one side of the face."""
    import numpy as np
    keys = mesh.data.shape_keys.key_blocks
    name = (WINK_CLOSE_PREFIX+side if closing else
            WINK_CANCEL_PREFIX+('L_' if side=='LEFT' else 'R_')+original)
    if name in keys:
        return name
    base = np.empty((len(keys[0].data),3),np.float32)
    delta = np.empty_like(base)
    keys[0].data.foreach_get('co',base.ravel())
    # Match the model's authored left/right controls rather than guessing from
    # screen space or the object's transform.
    reference = np.empty_like(base)
    keys[WINK_CONTROLS[side]].data.foreach_get('co',reference.ravel())
    affected = np.linalg.norm(reference-base,axis=1)>1e-6
    if not np.any(affected):
        raise ValueError('单眼眨眼形态键没有有效变形：'+WINK_CONTROLS[side])
    sign = 1 if np.median(base[affected,0])>0 else -1
    keys[original].data.foreach_get('co',delta.ravel())
    delta -= base
    delta[base[:,0]*sign<=0] = 0
    if np.max(np.abs(delta))<1e-8:
        return None
    key = mesh.shape_key_add(name=name,from_mix=False)
    key.data.foreach_set('co',(base+delta if closing else base-delta).ravel())
    key.relative_key = keys[0]
    key.value = 0
    return name


def apply_blinks(scene, keys, action, slot, base_values):
    settings = settings_for(scene)
    if not settings.blinks:
        return
    if BLINK_CONTROL not in keys.key_blocks:
        raise ValueError('此角色缺少“まばたき”眨眼形态键')
    for item in settings.blinks:
        if item.side!='BOTH' and WINK_CONTROLS[item.side] not in keys.key_blocks:
            raise ValueError('此角色缺少单眼眨眼形态键：'+WINK_CONTROLS[item.side])
    windows = []
    for item in sorted(settings.blinks, key=lambda i: i.frame-i.close_frames):
        start, end = item.frame-item.close_frames, item.frame+item.open_frames
        if windows and start <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(end, windows[-1][1]))
        else:
            windows.append((start, end))
    curves = {c.data_path: c for c in action_curves(action, slot) if c.array_index == 0}
    weights = {f: {side:max((blink_amount(item,f) for item in settings.blinks
                            if item.side==side),default=0)
                   for side in ('BOTH','LEFT','RIGHT')}
               for start,end in windows for f in range(start,end+1)}
    # Reset previously generated local correction channels before rebuilding.
    for key in keys.key_blocks:
        if key.name.startswith((WINK_CANCEL_PREFIX,WINK_CLOSE_PREFIX)):
            key.value = 0
            path = key.path_from_id('value')
            existing = curves.get(path)
            if existing:
                existing.keyframe_points.clear()
                existing.keyframe_points.insert(scene.frame_start,0)
    corrections = {}
    for name in BLINK_EYE_CONTROLS:
        key = keys.key_blocks.get(name)
        if not key:
            continue
        path = key.path_from_id('value')
        curve = curves.get(path)
        if keys.animation_data and any(d.data_path == path for d in keys.animation_data.drivers):
            raise ValueError('眨眼相关形态键存在驱动器，无法叠加：'+name)
        if curve and (curve.modifiers or curve.mute):
            raise ValueError('眨眼相关曲线有修改器或静音，无法叠加：'+name)
        samples = {f: curve.evaluate(f) if curve else base_values.get(name, 0)
                   for start, end in windows for f in range(start, end+1)}
        for side in ('LEFT','RIGHT'):
            if name in star_eyes.SIDES:
                continue
            local = {f: original*max(0,w[side]-w['BOTH'])
                     for f,original in samples.items() for w in (weights[f],)}
            if any(abs(v)>1e-8 for v in local.values()):
                cancel = wink_cancel_control(face_object(scene),name,side)
                if cancel:
                    corrections[cancel] = local
        curve = curve or action.fcurve_ensure_for_datablock(keys, path)
        for start, end in windows:
            split_curve_boundary(curve, start)
            split_curve_boundary(curve, end)
        # Preserve the exterior handles of all original keys. Interior samples
        # use linear interpolation over a smooth closing/opening envelope.
        exterior = [(p.co.x, p.handle_left.copy(), p.handle_right.copy(), p.interpolation)
                    for p in curve.keyframe_points
                    if not any(a < p.co.x < b for a, b in windows)]
        for index in range(len(curve.keyframe_points)-1, -1, -1):
            if any(a < curve.keyframe_points[index].co.x < b for a, b in windows):
                curve.keyframe_points.remove(curve.keyframe_points[index], fast=True)
        for frame, original in sorted(samples.items()):
            w = weights[frame]
            weight = w['BOTH']
            if name in star_eyes.SIDES:
                weight = max(weight, w[star_eyes.SIDES[name]])
            value = original*(1-weight)+(weight if name == BLINK_CONTROL else 0)
            point = curve.keyframe_points.insert(frame, value, options={'FAST'})
            point.interpolation = 'LINEAR'
        curve.update()
        for frame, hl, hr, interpolation in exterior:
            point = next(p for p in curve.keyframe_points if abs(p.co.x-frame)<1e-5)
            point.handle_left_type = point.handle_right_type = 'FREE'
            point.handle_left = hl; point.handle_right = hr
            if not any(a <= frame < b for a,b in windows):
                point.interpolation = interpolation
        curve.update()
    for side,original in WINK_CONTROLS.items():
        samples = {f:max(0,w[side]-w['BOTH']) for f,w in weights.items()}
        if any(v>0 for v in samples.values()):
            name = wink_cancel_control(face_object(scene),original,side,closing=True)
            corrections[name] = samples
    for name,samples in corrections.items():
        path = keys.key_blocks[name].path_from_id('value')
        curve = action.fcurve_ensure_for_datablock(keys,path)
        curve.keyframe_points.clear()
        # Guards ensure each local correction is zero before/after the blink.
        samples = dict(samples)
        for start,end in windows:
            samples[start-1]=0; samples[end+1]=0
        for frame,value in sorted(samples.items()):
            point=curve.keyframe_points.insert(frame,value,options={'FAST'})
            point.interpolation='LINEAR'
        curve.update()

def repair_school_transfer_rest(scene):
    """Repair the two composite sliders accidentally enabled by the old importer.

    An animated source is authoritative. Only the known simultaneous unit-value
    rest contamination in transferred school files is eligible for migration.
    """
    mesh = face_object(scene)
    if (not mesh or not mesh.get('EllenExpressionTransferSource') or
            mesh.get('EllenSchoolExpressionRestRevision', 0) >= 1):
        return False
    settings = settings_for(scene)
    keys = mesh.data.shape_keys
    ad = keys.animation_data
    source = (settings.base_action if ad and ad.action == settings.generated_action
              else ad.action if ad else None)
    paths = {keys.key_blocks[n].path_from_id('value') for n in PAIR_CONTROLS
             if n in keys.key_blocks}
    if any(c.data_path in paths for c in action_curves(source, None)):
        return False
    values = json.loads(settings.base_values or '{}')
    generated = bool(ad and ad.action and ad.action == settings.generated_action)
    rest = values if generated else {k.name: k.value for k in keys.key_blocks}
    if not all(abs(rest.get(n, 0)-1) < 1e-6 for n in PAIR_CONTROLS):
        return False
    for name in PAIR_CONTROLS:
        keys.key_blocks[name].value = 0
        if name in values:
            values[name] = 0
    if settings.base_values:
        settings.base_values = json.dumps(values)
    mesh['EllenSchoolExpressionRestRevision'] = 1
    return True


def rebuild(scene):
    keys = edit_available(scene)
    repair_school_transfer_rest(scene)
    ranges = intervals(scene)
    star_origins = star_eyes.burst_origins(ranges)
    settings = settings_for(scene)
    if any(item.preset == star_eyes.PRESET for _, _, item in ranges):
        star_eyes.ensure(face_object(scene))
    if is_ellen_core(face_object(scene)):
        ensure_ellen_core_controls(face_object(scene))
    if any(BLUSH_CONTROL in preset_values(scene, item.preset) for _, _, item in ranges):
        ensure_blush_control(face_object(scene))
    if any(DISGUST_CONTROL in preset_values(scene, item.preset) for _, _, item in ranges):
        ensure_disgust_control(face_object(scene))
    if any(GRIMACE_CONTROL in preset_values(scene, item.preset) for _, _, item in ranges):
        ensure_disgust_grimace(face_object(scene))
    if any(any(name in preset_values(scene, item.preset) for name in REFERENCE_CONTROLS)
           for _, _, item in ranges):
        ensure_reference_disgust(face_object(scene))
    if any(any(name in preset_values(scene, item.preset) for name in PAIR_CONTROLS)
           for _, _, item in ranges):
        ensure_disgust_pair(face_object(scene))
    if any(SIDE_GLANCE_CONTROL in preset_values(scene, item.preset) for _, _, item in ranges):
        ensure_reference_side_glance(face_object(scene))
    if any(SOFT_SMILE_CONTROL in preset_values(scene, item.preset) for _, _, item in ranges):
        ensure_reference_soft_smile(face_object(scene))
    if any(BLANK_BROWS_CONTROL in preset_values(scene, item.preset) for _, _, item in ranges):
        ensure_reference_blank_brows(face_object(scene))
    if any(BELLE_WEARY_BROWS in preset_values(scene, item.preset) for _, _, item in ranges):
        ensure_belle_weary_brows(face_object(scene))
    for _, _, item in ranges:
        required = set(preset_values(scene, item.preset))
        if item.auto_mouth and not item.override_mouth and not is_sharkboo(face_object(scene)):
            required.update(SPEECH_NAMES)
        missing = required - set(keys.key_blocks.keys())
        if missing:
            raise ValueError('当前角色缺少表情映射：' + '、'.join(sorted(missing)))
    keys.animation_data_create()
    ad = keys.animation_data
    current = ad.action
    # Snapshot only when the source changes, not on every edit of the generated take.
    if settings.target != keys or current != settings.generated_action:
        settings.target = keys
        settings.base_action = current.copy() if current else None
        if settings.base_action:
            settings.base_action.name = 'Expression Source Backup'
            settings.base_action.use_fake_user = True
            settings.base_slot = ad.action_slot.handle if ad.action_slot else 0
        else:
            settings.base_slot = 0
        settings.base_values = json.dumps({k.name: k.value for k in keys.key_blocks})
    if not settings.base_values:
        settings.base_values = json.dumps({k.name: (0 if current == settings.generated_action and k.name in controls_for(scene) else k.value)
                                          for k in keys.key_blocks})
    base = settings.base_action
    # A return-to-source key is only needed when the timeline continues after
    # the expression. Do not create an extra rendered frame at a trimmed end.
    timeline_end = max(scene.frame_end, base.frame_range[1] if base else scene.frame_end,
                       max((end for _, end, _ in ranges), default=scene.frame_end))
    base_slot = next((slot for slot in base.slots if slot.handle == settings.base_slot), None) if base else None
    lookup = {c.data_path: c for c in action_curves(base, base_slot) if c.array_index == 0}
    values = json.loads(settings.base_values or '{}')
    speech_available = has_speech_curves(keys, base, base_slot)
    smoothed = {}
    if speech_available and settings.speech_speed < 1:
        import numpy as np
        for name in VOWELS:
            key = keys.key_blocks.get(name)
            curve = lookup.get(key.path_from_id('value')) if key else None
            if curve is None or len(curve.keyframe_points) < 2:
                continue
            if curve.modifiers or curve.mute:
                raise ValueError('面部曲线有修改器或静音，暂不支持安全叠加：' + name)
            first, last = curve.keyframe_points[0].co.x, curve.keyframe_points[-1].co.x
            frames = np.arange(math.floor(first), math.ceil(last)+1, dtype=float)
            output = smooth_speech([curve.evaluate(f) for f in frames], fps(scene), settings.speech_speed)
            # Retain exact subframe endpoints and neutral boundary values.
            positions = np.unique(np.r_[first, frames[(frames > first) & (frames < last)], last])
            samples = np.interp(positions, frames, output)
            samples[0], samples[-1] = curve.evaluate(first), curve.evaluate(last)
            smoothed[name] = positions, samples
    def source_value(name, frame):
        if name in smoothed:
            positions, samples = smoothed[name]
            return float(np.interp(frame, positions, samples))
        key = keys.key_blocks.get(name)
        curve = lookup.get(key.path_from_id('value')) if key else None
        if curve and (curve.modifiers or curve.mute):
            raise ValueError('面部曲线有修改器或静音，暂不支持安全叠加：' + name)
        return curve.evaluate(frame) if curve else values.get(name, 0)
    mouth_samples = {}
    speech_gain = settings.global_speech_gain if speech_available else 1.0
    for start, end, item in ranges:
        if item.auto_mouth and not item.override_mouth and speech_available:
            mouth_samples[item.as_pointer()] = {
                frame: linked_mouth(item.preset, {name: source_value(name, frame) for name in VOWELS}, item.speech_gain)
                for frame in range(start, end + 1)
            }
    old = settings.generated_action
    action = base.copy() if base else bpy.data.actions.new('Ellen Timed Expressions')
    action.name = 'Ellen Timed Expressions'
    action.use_fake_user = True
    try:
        ad.action = action
        if base_slot:
            ad.action_slot = next(slot for slot in action.slots if slot.handle == base_slot.handle)
        # Adjust the copied source too: speech in gaps and outside expression
        # intervals belongs to the same controls. Start from the original
        # backup so neither gain nor temporal smoothing compounds on reapply.
        speech_paths = {keys.key_blocks[n].path_from_id('value')
                        for n in VOWELS if n in keys.key_blocks}
        smoothed_paths = {keys.key_blocks[n].path_from_id('value'): samples
                          for n, samples in smoothed.items()}
        for curve in action_curves(action, ad.action_slot):
            if curve.data_path not in speech_paths:
                continue
            if curve.data_path in smoothed_paths:
                positions, samples = smoothed_paths[curve.data_path]
                curve.keyframe_points.clear()
                curve.keyframe_points.add(len(positions))
                curve.keyframe_points.foreach_set('co', [v for f, value in zip(positions, samples)
                                                        for v in (f, value * speech_gain)])
                for point in curve.keyframe_points:
                    point.interpolation = 'LINEAR'
                curve.update()
                continue
            if speech_gain == 1.0:
                continue
            if curve.modifiers or curve.mute:
                raise ValueError('面部曲线有修改器或静音，暂不支持安全叠加：' + curve.data_path)
            for point in curve.keyframe_points:
                point.co.y *= speech_gain
                point.handle_left.y *= speech_gain
                point.handle_right.y *= speech_gain
            curve.update()
        for name in controls_for(scene):
            key = keys.key_blocks.get(name)
            if not key:
                continue
            relevant = [(a, b, item) for a, b, item in ranges if name not in VOWELS or item.override_mouth or item.auto_mouth]
            if not relevant:
                continue
            path = key.path_from_id('value')
            source = lookup.get(path)
            if source and (source.modifiers or source.mute):
                raise ValueError('面部曲线有修改器或静音，暂不支持安全叠加：' + name)
            scale = speech_gain if name in VOWELS else 1.0
            baseline = lambda f: source_value(name, f) * scale
            sampled = {}
            def target_value(item, frame):
                target = preset_values(scene, item.preset).get(name, 0)
                if item.preset == star_eyes.PRESET and name in star_eyes.CONTROLS:
                    target = star_eyes.timed_value(name, (frame-star_origins[item.as_pointer()])/fps(scene))
                linked = mouth_samples.get(item.as_pointer())
                if linked is not None and (name in VOWELS or name == bridge.TEETH_CONTROL):
                    if frame not in linked:
                        linked[frame] = linked_mouth(item.preset,
                            {n: source_value(n, frame) for n in VOWELS}, item.speech_gain)
                    target = linked[frame][name]
                    if name in VOWELS:
                        target *= speech_gain
                if name == bridge.TEETH_CONTROL:
                    if item.override_mouth:
                        target = float(item.preset in TEETH_PRESETS)
                    elif linked is None:
                        target = float(item.preset in TEETH_PRESETS - LAUGH_PRESETS)
                return target
            for pos, (start, end, item) in enumerate(relevant):
                sampled.setdefault(start - 1, baseline(start - 1))
                if end < timeline_end:
                    sampled.setdefault(end + 1, baseline(end + 1))
                previous = relevant[pos - 1] if pos and relevant[pos - 1][1] == start else None
                following = pos + 1 < len(relevant) and relevant[pos + 1][0] == end
                for frame in range(start, end + 1):
                    blend = amount(item, frame, start, end, fps(scene), following)
                    target = target_value(item, frame)
                    origin = baseline(frame)
                    # Touching segments blend directly from the preceding expression,
                    # rather than briefly returning to the captured neutral face.
                    if previous and item.transition == 'SMOOTH':
                        fade = min(item.fade_frames, (end - start) // 2)
                        t = min(1, (frame - start) / fade) if fade > 0 else 1
                        t = t * t * (3 - 2 * t)
                        prior = previous[2]
                        prior_target = target_value(prior, frame)
                        origin = baseline(frame) * (1 - prior.intensity) + prior_target * prior.intensity
                        out = 1 if following or fade <= 0 else max(0, min(1, (end - frame) / fade))
                        out = out * out * (3 - 2 * out)
                        strength = item.intensity * out
                        current_value = baseline(frame) * (1 - strength) + target * strength
                        sampled[frame] = origin * (1 - t) + current_value * t
                    else:
                        sampled[frame] = origin * (1 - blend) + target * blend
                    if name == bridge.TEETH_CONTROL and item.preset not in TEETH_PRESETS and blend > 0:
                        sampled[frame] = 0
            curve = action.fcurve_ensure_for_datablock(keys, path)
            for index in range(len(curve.keyframe_points) - 1, -1, -1):
                point = curve.keyframe_points[index]
                if any(start - 1 <= point.co.x <= end + 1 for start, end, _ in relevant):
                    curve.keyframe_points.remove(point, fast=True)
            for frame, value in sorted(sampled.items()):
                point = curve.keyframe_points.insert(frame, value, options={'FAST'})
                point.interpolation = 'LINEAR'
            curve.update()
        apply_blinks(scene, keys, action, ad.action_slot, values)
        action['ellen_timed_expressions'] = True
        action['ellen_applied_speech_gain'] = settings.global_speech_gain
        action['ellen_applied_speech_speed'] = settings.speech_speed
    except Exception:
        ad.action = current
        if current and settings.generated_slot:
            slot = next((slot for slot in current.slots if slot.handle == settings.generated_slot), None)
            if slot: ad.action_slot = slot
        bpy.data.actions.remove(action)
        raise
    settings.generated_action = action
    settings.generated_slot = ad.action_slot.handle if ad.action_slot else 0
    if face_object(scene) == bridge.face_mesh(scene):
        scene.xr_ellen.face_take = action
        scene.xr_ellen.face_mode = 'OFF'
    if ranges:
        scene.frame_end = max(scene.frame_end, ranges[-1][1])
    settings.status = (f'已应用：说话幅度 {settings.global_speech_gain:.2f} / '
                       f'变化速度 {settings.speech_speed:.2f} / {len(ranges)} 个表情段')
    if old and old != action:
        old.use_fake_user = False
        if old.users == 0:
            bpy.data.actions.remove(old)
    for key in keys.key_blocks:
        if key.name in values:
            key.value = values[key.name] * (speech_gain if key.name in VOWELS else 1.0)
        elif key.name in BLINK_EYE_CONTROLS:
            key.value = 0
        elif key.name.startswith((WINK_CANCEL_PREFIX,WINK_CLOSE_PREFIX)):
            key.value = 0
    scene.frame_set(scene.frame_current)
    return action

def dirty(self, context):
    if context and face_object(context.scene):
        face_object(context.scene).ellen_expression_track.status = '有修改，待应用'


def apply_speech_gain(scene, mesh):
    """Apply speech gain and response without running capture or decoding."""
    if not mesh or mesh.type != 'MESH' or not mesh.data.shape_keys:
        raise ValueError('请先选择角色面部网格')
    keys = mesh.data.shape_keys
    ad = keys.animation_data
    if not ad or not has_speech_curves(keys, ad.action, ad.action_slot):
        raise ValueError('此角色还没有生成的说话口型')
    self = mesh.ellen_expression_track
    from sam_ellen_face import face_state, restore_face
    before = face_state(mesh, scene)
    selected = scene.ellen_expression_object
    end = scene.frame_end
    old_generated_name = self.generated_action.name if self.generated_action else None
    sam_target = (hasattr(scene, 'sam_ellen') and scene.sam_ellen.face_take == ad.action)
    try:
        from sam_ellen_editing import available
        available()
        scene.ellen_expression_object = mesh
        result = rebuild(scene)
        if sam_target:
            scene.sam_ellen.face_take = result
        old_generated = bpy.data.actions.get(old_generated_name) if old_generated_name else None
        if old_generated and old_generated != result and old_generated.users == 0:
            bpy.data.actions.remove(old_generated)
        self.status = f'已应用口型：幅度 {self.global_speech_gain:.2f} / 变化速度 {self.speech_speed:.2f}'
        return result
    except Exception:
        restore_face(before, scene)
        raise
    finally:
        scene.ellen_expression_object = selected
        scene.frame_end = end


def live_speech_gain_changed(self, context):
    """Reapply recorded mouth curves immediately, always from the source backup."""
    scene = context.scene if context else None
    mesh = self.id_data
    if not scene or not isinstance(mesh, bpy.types.Object) or mesh.name not in scene.objects:
        return
    keys = mesh.data.shape_keys if mesh.type == 'MESH' else None
    ad = keys.animation_data if keys else None
    if not ad or not has_speech_curves(keys, ad.action, ad.action_slot):
        dirty(self, context)
        return
    previous_gain = ad.action.get('ellen_applied_speech_gain', 1.0)
    try:
        apply_speech_gain(scene, mesh)
    except Exception as error:
        # Writing the stored value avoids recursively invoking this callback.
        self['global_speech_gain'] = previous_gain
        self.status = '幅度未更改：'+str(error)


def live_speech_speed_changed(self, context):
    scene = context.scene if context else None
    mesh = self.id_data
    if not scene or not isinstance(mesh, bpy.types.Object) or mesh.name not in scene.objects:
        return
    keys = mesh.data.shape_keys if mesh.type == 'MESH' else None
    ad = keys.animation_data if keys else None
    if not ad or not has_speech_curves(keys, ad.action, ad.action_slot):
        dirty(self, context)
        return
    previous = ad.action.get('ellen_applied_speech_speed', 1.0)
    try:
        apply_speech_gain(scene, mesh)
    except Exception as error:
        self['speech_speed'] = previous
        self.status = '速度未更改：'+str(error)


def draw_live_speech_gain(layout, scene, mesh=None):
    mesh = mesh or face_object(scene)
    if not mesh or not hasattr(mesh, 'ellen_expression_track') or not mesh.data.shape_keys:
        return
    keys = mesh.data.shape_keys
    ad = keys.animation_data
    column = layout.column()
    column.enabled = bool(ad and has_speech_curves(keys, ad.action, ad.action_slot))
    column.prop(mesh.ellen_expression_track, 'global_speech_gain', text='已生成口型幅度', slider=True)
    op = column.operator('sam_ellen.apply_speech_gain', text='仅应用口型幅度', icon='CHECKMARK')
    op.mesh_name = mesh.name
    column.label(text='拖动即时生效，1 为原幅度')
    column.prop(mesh.ellen_expression_track, 'speech_speed', text='口型变化速度', slider=True)
    op = column.operator('sam_ellen.apply_speech_speed', icon='CHECKMARK')
    op.mesh_name = mesh.name
    column.label(text='1 为原速度，越小越柔和；建议先试 0.5')
    if mesh.ellen_expression_track.status.startswith(('幅度未更改', '速度未更改')):
        column.label(text=mesh.ellen_expression_track.status, icon='ERROR')


def legacy_timing_changed(self, context):
    # Keep old project scripts which set start/end/fade in seconds compatible.
    # New UI and baking use the integer frame fields exclusively.
    self.timing_version = 0
    dirty(self, context)


def frame_timing_changed(self, context):
    self.timing_version = 1
    dirty(self, context)

class ExpressionSegment(bpy.types.PropertyGroup):
    enabled: BoolProperty(name='启用', default=True, update=dirty)
    preset: EnumProperty(name='表情', items=preset_items, default=2, update=dirty)
    start: FloatProperty(default=0, min=0, max=3600, options={'HIDDEN'}, update=legacy_timing_changed)
    end: FloatProperty(default=2, min=0, max=3600, options={'HIDDEN'}, update=legacy_timing_changed)
    timing_version: IntProperty(default=0, options={'HIDDEN'})
    start_frame: IntProperty(name='开始帧', default=1, min=-1048574, max=1048574, update=frame_timing_changed)
    end_frame: IntProperty(name='结束帧', default=61, min=-1048574, max=1048574, update=frame_timing_changed,
        description='包含结束帧；相邻两段共用同一边界帧时，由后一段接续')
    intensity: FloatProperty(name='强度', default=1, min=0, max=1, subtype='FACTOR', update=dirty)
    transition: EnumProperty(name='切换', items=(('SMOOTH','平滑',''),('CUT','直接切换','')), default='SMOOTH', update=dirty)
    fade: FloatProperty(default=.2, min=0, max=5, options={'HIDDEN'}, update=legacy_timing_changed)
    fade_frames: IntProperty(name='渐入 / 渐出（帧）', default=6, min=0, max=1048574, update=frame_timing_changed)
    override_mouth: BoolProperty(name='固定嘴形', description='使用预设的固定嘴形，不跟随录好的说话开合', default=False, update=dirty)
    auto_mouth: BoolProperty(name='嘴型随表情', description='保留台词开合节奏，自动适配当前表情的嘴形', default=True, update=dirty)
    speech_gain: FloatProperty(name='此段额外倍率', default=1, min=0, max=2, update=dirty,
        description='旧工程保留的单段口型倍率，与全局说话幅度相乘')

class ExpressionBlink(bpy.types.PropertyGroup):
    frame: IntProperty(name='闭眼帧', min=-1048574, max=1048574)
    side: EnumProperty(name='眼睛', items=BLINK_SIDES, default='BOTH')
    close_frames: IntProperty(name='闭合帧数', default=3, min=1, max=60)
    open_frames: IntProperty(name='睁开帧数', default=5, min=1, max=60)


class ExpressionSettings(bpy.types.PropertyGroup):
    items: CollectionProperty(type=ExpressionSegment)
    blinks: CollectionProperty(type=ExpressionBlink)
    blink_side: EnumProperty(name='眼睛', items=BLINK_SIDES, default='BOTH')
    blink_close_frames: IntProperty(name='闭合', default=3, min=1, max=60,
        description='从原眼型到完全闭眼所用的帧数；当前帧为完全闭眼')
    blink_open_frames: IntProperty(name='睁开', default=5, min=1, max=60,
        description='完全闭眼后恢复原眼型所用的帧数')
    global_speech_gain: FloatProperty(name='全局说话幅度', default=1, min=0, max=2, update=live_speech_gain_changed,
        description='实时缩放此角色整段已生成的说话口型，包含未设置表情的帧；1 为原幅度，0 为无说话开合，无需重新识别音频或生成动捕')
    speech_speed: FloatProperty(name='口型变化速度', default=1, min=.1, max=1,
        update=live_speech_speed_changed,
        description='1 保留原口型，越小过渡越柔和、减少快速抖动；不改变台词时长或移动口型时间轴。直接调整已有口型，无需重新分析音频')
    index: IntProperty(default=0)
    target: PointerProperty(type=bpy.types.Key)
    base_action: PointerProperty(type=bpy.types.Action)
    generated_action: PointerProperty(type=bpy.types.Action)
    base_values: StringProperty()
    base_slot: IntProperty()
    generated_slot: IntProperty()
    status: StringProperty(default='未应用')
    advanced: BoolProperty(name='嘴型微调', default=False)

class ELLEN_UL_expressions(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row=layout.row(align=True)
        row.prop(item, 'enabled', text='')
        row.label(text=presets_for(context.scene).get(item.preset, ('请选择表情', {}))[0])
        row.label(text=f'{item.start_frame}–{item.end_frame} 帧')

class ExpressionChoose(bpy.types.Operator):
    bl_idname='ellen_expression.choose'
    bl_label='设置当前表情段'
    bl_description='修改选中表情段并预览，保留动捕口型和原始动画备份'
    bl_options={'REGISTER','UNDO'}
    preset: StringProperty(default='neutral')

    def execute(self, context):
        scene=context.scene
        if self.preset not in presets_for(scene):
            return {'CANCELLED'}
        try:
            edit_available(scene)
            settings=settings_for(scene)
            added=not 0 <= settings.index < len(settings.items)
            if added:
                item=add_segment(scene,settings)
            else:
                item=settings.items[settings.index]
            previous=item.preset
            item.preset=self.preset
            try:
                rebuild(scene)
            except Exception:
                if added:
                    settings.items.remove(settings.index)
                    settings.index=max(0,len(settings.items)-1)
                else:
                    item.preset=previous
                raise
            scene.frame_set(star_eyes.preview_frame(item, fps(scene)) if item.preset == star_eyes.PRESET
                            else (item.start_frame+item.end_frame)//2)
            return {'FINISHED'}
        except (ValueError,RuntimeError) as error:
            self.report({'ERROR'},str(error));return {'CANCELLED'}

class ExpressionBlinkEdit(bpy.types.Operator):
    bl_idname='ellen_expression.blink'
    bl_label='当前帧眨眼'
    bl_description='当前帧完全闭眼，前后自动补上睁眼过渡；重新应用表情后仍保留'
    bl_options={'REGISTER','UNDO'}
    operation: EnumProperty(items=(('ADD','添加 / 更新眨眼',''),('REMOVE','删除当前帧眨眼','')))

    def execute(self, context):
        scene=context.scene
        try:
            keys=edit_available(scene)
            if BLINK_CONTROL not in keys.key_blocks:
                raise ValueError('此角色缺少“まばたき”眨眼形态键')
            settings=settings_for(scene)
            previous=[(i.frame,i.close_frames,i.open_frames,i.side) for i in settings.blinks]
            status=settings.status
            side=settings.blink_side
            if side!='BOTH' and WINK_CONTROLS[side] not in keys.key_blocks:
                raise ValueError('此角色缺少单眼眨眼形态键：'+WINK_CONTROLS[side])
            index=next((n for n,i in enumerate(settings.blinks)
                        if i.frame==scene.frame_current and i.side==side),None)
            if self.operation=='REMOVE':
                if index is None:raise ValueError('当前帧没有手动添加的眨眼，请回到添加时的闭眼帧')
                settings.blinks.remove(index)
            else:
                item=settings.blinks[index] if index is not None else settings.blinks.add()
                item.frame=scene.frame_current
                item.side=side
                item.close_frames=settings.blink_close_frames
                item.open_frames=settings.blink_open_frames
            try:
                rebuild(scene)
            except Exception:
                settings.blinks.clear()
                for frame,close_frames,open_frames,side in previous:
                    item=settings.blinks.add();item.frame=frame
                    item.close_frames=close_frames;item.open_frames=open_frames
                    item.side=side
                settings.status=status
                raise
            label=dict((key,label) for key,label,_ in BLINK_SIDES)[settings.blink_side]
            message=f'已在第 {scene.frame_current} 帧'+('删除' if self.operation=='REMOVE' else '添加')+label+'眨眼'
            self.report({'INFO'},message)
            return {'FINISHED'}
        except (ValueError,RuntimeError) as error:
            self.report({'ERROR'},str(error));return {'CANCELLED'}


class ExpressionEdit(bpy.types.Operator):
    bl_idname='ellen_expression.edit'
    bl_label='表情时间段'
    bl_options={'REGISTER','UNDO'}
    operation: EnumProperty(items=[(v,v,'') for v in ('ADD','REMOVE','APPLY','PREVIEW','START','END','JUMP','SAVE','VIEW_SOLID','VIEW_MATERIAL')])
    @classmethod
    def description(cls, context, properties):
        return {'ADD':'接在最后一段结束帧之后添加；第一段从当前帧开始，默认长度为当前帧率的两秒', 'REMOVE':'删除选中段并恢复原表情',
                'START':'用当前时间轴帧设置开始帧', 'END':'用当前时间轴帧设置结束帧',
                'APPLY':'应用全局说话幅度和所有表情时间段，保留原动捕备份', 'PREVIEW':'应用并跳到选中段的中间',
                'JUMP':'跳到选中段的开始', 'SAVE':'保存当前工程和表情时间段',
                'VIEW_SOLID':'切换到实体视口，不改变最终渲染材质',
                'VIEW_MATERIAL':'恢复材质预览，不改变最终渲染设置'}.get(properties.operation, '')
    def execute(self, context):
        scene=context.scene
        try:
            if self.operation in {'VIEW_SOLID','VIEW_MATERIAL'}:
                if not context.screen:raise ValueError('没有可用的视口')
                for area in context.screen.areas:
                    if area.type=='VIEW_3D':
                        area.spaces.active.shading.type='SOLID' if self.operation=='VIEW_SOLID' else 'MATERIAL'
                return {'FINISHED'}
            if self.operation=='SAVE':
                if not bpy.data.filepath: raise ValueError('请先另存工程')
                if settings_for(scene).status in {'有修改，待应用', '待应用'}:
                    rebuild(scene)
                bpy.ops.wm.save_mainfile()
                self.report({'INFO'}, '工程已保存')
                return {'FINISHED'}
            edit_available(scene)
            settings=settings_for(scene)
            if self.operation=='ADD':
                add_segment(scene,settings)
                settings.status='待应用'
                return {'FINISHED'}
            if self.operation=='APPLY':
                rebuild(scene)
            else:
                if not 0<=settings.index<len(settings.items):raise ValueError('请先添加或选中一个时间段')
                item=settings.items[settings.index]
                if self.operation=='START':item.start_frame=scene.frame_current
                elif self.operation=='END':item.end_frame=scene.frame_current
                elif self.operation=='JUMP':scene.frame_set(item.start_frame)
                elif self.operation=='PREVIEW':
                    rebuild(scene)
                    scene.frame_set(star_eyes.preview_frame(item, fps(scene)) if item.preset == star_eyes.PRESET
                                    else (item.start_frame+item.end_frame)//2)
                elif self.operation=='REMOVE':
                    # Validate remaining edits before deleting the selected item.
                    was=item.enabled;item.enabled=False
                    try:rebuild(scene)
                    except Exception:
                        item.enabled=was
                        raise
                    settings.items.remove(settings.index)
                    settings.index=max(0,min(settings.index,len(settings.items)-1))
            return {'FINISHED'}
        except (ValueError,RuntimeError) as error:
            self.report({'ERROR'},str(error));return {'CANCELLED'}

class ExpressionPanel(bpy.types.Panel):
    bl_label='表情时间段'
    bl_idname='ELLEN_PT_expression_timeline'
    bl_space_type='VIEW_3D';bl_region_type='UI';bl_category='表情时间段'
    def draw(self, context):
        scene=context.scene;layout=self.layout
        layout.prop(scene,'ellen_expression_object',text='控制角色')
        mesh=face_object(scene)
        if not mesh:
            return
        # Blender forbids ID writes during draw, including legacy migration.
        # Loading/registering and edit operators perform that conversion instead.
        s=settings_for(scene, migrate=False)
        connected=bridge._runtime is not None
        if connected:
            layout.operator('xr_ellen.finish_record',text='结束录制 / 断开 XR',icon='PAUSE')
        column=layout.column();column.enabled=not connected
        row=column.row(align=True)
        row.prop(scene,'frame_current',text='当前帧')
        blink=column.box()
        blink.enabled=bool(mesh.data.shape_keys and BLINK_CONTROL in mesh.data.shape_keys.key_blocks)
        blink.prop(s,'blink_side',expand=True)
        row=blink.row(align=True)
        row.operator('ellen_expression.blink',text='当前帧眨眼',icon='HIDE_ON').operation='ADD'
        remove=row.row(align=True)
        remove.enabled=any(i.frame==scene.frame_current and i.side==s.blink_side for i in s.blinks)
        remove.operator('ellen_expression.blink',text='删除此帧眨眼',icon='X').operation='REMOVE'
        row=blink.row(align=True)
        row.prop(s,'blink_close_frames');row.prop(s,'blink_open_frames')
        blink.label(text=f'当前帧完全闭眼 · 已添加 {len(s.blinks)} 次眨眼')
        if s.blink_side!='BOTH':blink.label(text='左右按角色自身区分')
        if hasattr(scene, 'ellen_physics') and __import__('ellen_physics').manages(scene):
            column.prop(scene.ellen_physics, 'enabled', text='统一人物物理（设置见 SAM / 角色操控）')
        elif hasattr(scene,'wiggle_enable'):
            if scene.get('SecondaryPhysicsBaked') and not scene.wiggle_enable:
                column.label(text='头发 / 衣带物理：已烘焙为动画',icon='ACTION')
            else:
                column.prop(scene,'wiggle_enable',text='头发 / 衣带 / 尾巴物理')
        row=column.row(align=True)
        row.operator('ellen_expression.edit',text='流畅预览',icon='SHADING_SOLID').operation='VIEW_SOLID'
        row.operator('ellen_expression.edit',text='材质预览',icon='SHADING_TEXTURE').operation='VIEW_MATERIAL'
        is_belle = bool(mesh.data.shape_keys and
                        'Belle_MouthWidth_Adjustment' in mesh.data.shape_keys.key_blocks)
        sharkboo = is_sharkboo(mesh)
        core = is_ellen_core(mesh)
        if core:
            column.label(text=f'艾莲 · {len(ELLEN_CORE_PRESETS)} 个表情')
        featured_presets = (() if sharkboo or core else (star_eyes.PRESET, BELLE_REFERENCE_PRESET) if is_belle else
                            (*SELECTED_REFERENCE_PRESETS, 'blank_stare_shy'))
        if featured_presets:
            column.label(text='本次新增 / 参考选定')
            featured=column.grid_flow(columns=2,align=True)
            for preset in featured_presets:
                featured.operator('ellen_expression.choose',text=presets_for(scene)[preset][0]).preset=preset
            if is_belle:
                column.label(text='星星眼：段首闪现约 1.1 秒，随后保留开心脸')
            column.separator()
        grid=column.grid_flow(columns=2,align=True)
        for preset,(label,_) in presets_for(scene).items():
            if preset in featured_presets or (preset == BELLE_REFERENCE_PRESET and not is_belle):
                continue
            grid.operator('ellen_expression.choose',text=label).preset=preset
        if not sharkboo:
            if speech_source_available(scene):
                column.label(text='口型来源：已录制的面部动画',icon='ACTION')
            else:
                column.label(text='尚无口型动画，当前为静态表情',icon='INFO')
            column.prop(s,'global_speech_gain',slider=True)
            column.prop(s,'speech_speed',slider=True)
            column.label(text='整段口型即时生效，表情修改后点击应用')
        row=column.row()
        row.template_list('ELLEN_UL_expressions','',s,'items',s,'index',rows=4)
        buttons=row.column(align=True)
        buttons.operator('ellen_expression.edit',text='',icon='ADD').operation='ADD'
        buttons.operator('ellen_expression.edit',text='',icon='REMOVE').operation='REMOVE'
        if 0<=s.index<len(s.items):
            item=s.items[s.index]
            column.prop(item,'preset')
            row=column.row(align=True)
            row.prop(item,'start_frame');row.operator('ellen_expression.edit',text='',icon='TIME').operation='START'
            row=column.row(align=True)
            row.prop(item,'end_frame');row.operator('ellen_expression.edit',text='',icon='TIME').operation='END'
            column.prop(item,'intensity',slider=True)
            column.prop(item,'transition',expand=True)
            if item.transition=='SMOOTH':column.prop(item,'fade_frames')
            if not sharkboo:
                column.prop(item,'auto_mouth',text='保留动捕说话口型')
                column.prop(s,'advanced',icon='DISCLOSURE_TRI_DOWN' if s.advanced else 'DISCLOSURE_TRI_RIGHT',emboss=False)
            if s.advanced and not sharkboo:
                if abs(item.speech_gain - 1.0) > 1e-6:
                    column.prop(item,'speech_gain',slider=True)
                column.prop(item,'override_mouth',text='固定嘴形（不跟随动捕）')
            row=column.row(align=True)
            row.operator('ellen_expression.edit',text='预览此表情',icon='HIDE_OFF').operation='PREVIEW'
            row.operator('ellen_expression.edit',text='跳到开始',icon='REW').operation='JUMP'
        row=column.row();row.scale_y=1.3
        row.operator('ellen_expression.edit',text='应用所有表情' if sharkboo else '应用幅度与所有表情',icon='CHECKMARK').operation='APPLY'
        column.label(text=s.status)
        layout.operator('ellen_expression.edit',text='保存工程',icon='FILE_TICK').operation='SAVE'

CLASSES=(ExpressionSegment,ExpressionBlink,ExpressionSettings,ELLEN_UL_expressions,
         ExpressionChoose,ExpressionBlinkEdit,ExpressionEdit,ExpressionPanel)
def register():
    for cls in CLASSES:bpy.utils.register_class(cls)
    bpy.types.Scene.ellen_expressions=PointerProperty(type=ExpressionSettings)
    bpy.types.Object.ellen_expression_track=PointerProperty(type=ExpressionSettings)
    bpy.types.Scene.ellen_expression_object=PointerProperty(name='控制角色',type=bpy.types.Object,
        poll=lambda self,obj: obj.type=='MESH' and obj.data.shape_keys is not None)
    if migrate_on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(migrate_on_load)
    if hasattr(bpy.data,'scenes'):
        for scene in bpy.data.scenes:migrate_scene(scene)
def unregister():
    if migrate_on_load in bpy.app.handlers.load_post:bpy.app.handlers.load_post.remove(migrate_on_load)
    del bpy.types.Scene.ellen_expression_object
    del bpy.types.Object.ellen_expression_track
    del bpy.types.Scene.ellen_expressions
    for cls in reversed(CLASSES):bpy.utils.unregister_class(cls)

@bpy.app.handlers.persistent
def migrate_on_load(_):
    for scene in bpy.data.scenes:migrate_scene(scene)

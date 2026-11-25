"""Local XR MMD preview receiver, using MMD Tools' existing VMD conversion."""

bl_info = {
    "name": "XR Animator - Ellen Live Preview", "author": "Local Tools",
    "version": (0, 3, 13), "blender": (4, 5, 0), "category": "Animation",
}

import importlib
import json
import math
import secrets
import socket
import time
from datetime import datetime
from pathlib import Path

import bpy
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from bpy_extras.io_utils import ExportHelper, ImportHelper
from mathutils import Matrix
from xr_hand_retarget import (correct_mouth_hands, mouth_anchor, head_contact_anchor,
    correct_contact_hands, contact_segments, contact_signature)
from ellen_expression_library import (BLUSH_CONTROL, DISGUST_CONTROL, GRIMACE_CONTROL, EXTRA_PRESETS,
    REFERENCE_CONTROLS, REFERENCE_EYES, REFERENCE_BROWS, REFERENCE_MOUTH,
    PAIR_CONTROLS, ensure_disgust_pair, SIDE_GLANCE_CONTROL, ensure_reference_side_glance,
    SOFT_SMILE_CONTROL, ensure_reference_soft_smile,
    BLANK_BROWS_CONTROL, ensure_reference_blank_brows,
    BELLE_WEARY_BROWS, ensure_belle_weary_brows,
    ensure_blush_control, ensure_disgust_control, ensure_disgust_grimace, ensure_reference_disgust)

PORT = 38383
_runtime = None

FIXED_STAGE_BONES = frozenset({"全ての親", "センター", "グルーブ", "腰", "下半身"})
FIXED_LEG_BONES = frozenset(side + part for side in ("左", "右") for part in (
    "足", "ひざ", "足首", "つま先", "足先EX", "足D", "ひざD", "足首D",
    "足ＩＫ", "つま先ＩＫ", "足IK", "つま先IK", "足IK親", "足ＩＫ親"))
FIXED_LOWER_BONES = FIXED_STAGE_BONES | FIXED_LEG_BONES

# Artistic presets reuse the original MMD shapes; they do not edit the mesh.
TEETH_CONTROL = "Ellen_ShowTeeth"
SHADOW_CONTROL = "Ellen_ForeheadShadow"
SHADOW_ATTRIBUTE = "Ellen_ForeheadMask"
FACE_PRESETS = {
    "neutral": ("自然 / 重置", {}),
    "talk": ("平静说话", {"じと目": .12, "真面目": .15, "あ": .18}),
    "smile": ("轻微微笑", {"じと目": .12, "にこり": .8, "上": .2, "口角上げ": .35}),
    "unimpressed": ("半眯 / 不耐烦", {"じと目": .6, "まばたき": .22, "真面目": .55, "下": .4, "あ": .12}),
    "serious": ("认真 / 反驳", {"怒り目": .35, "怒り": .85, "下": .2, "あ": .24}),
    "laugh": ("大笑 / 露牙", {"笑い": 1.0, "にこり": .85, "上": .18, "ワ": .6, "口角上げ": .25, TEETH_CONTROL: 1}),
    "emphasis": ("强调说话", {"じと目": .15, "上": .7, "真面目": .2, "あ": .4}),
    "brooding": ("阴沉 / 额头阴影", {"怒り": .95, "下": .3, "じと目": .65, "まばたき": .18, "あ": .2, "口角下げ": .15, SHADOW_CONTROL: 1.0}),
    "worried": ("担忧 / 委屈", {"困る": .95, "じと目": .15, "まばたき": .1, "あ": .1, "口角下げ": .3}),
    "smile_closed": ("闭眼笑着说话", {"笑い": 1.0, "にこり": .7, "上": .15, "口角上げ": .4}),
}
FACE_PRESETS.update(EXTRA_PRESETS)
FACE_CONTROLS = (
    ("まばたき", "闭眼"), ("笑い", "笑眼"), ("じと目", "半眯眼"),
    ("ウィンク２", "左眼闭合"), ("ウィンク２右", "右眼闭合"),
    ("怒り目", "认真眼"), ("びっくり左", "左眼睁大"), ("びっくり右", "右眼睁大"),
    ("真面目", "认真眉"), ("にこり", "微笑眉"), ("困る", "担忧眉"),
    ("怒り", "皱眉"), ("上", "抬眉"), ("下", "压眉"),
    ("あ", "张嘴 A"), ("い", "嘴型 I"), ("う", "嘴型 U"),
    ("え", "嘴型 E"), ("お", "嘴型 O"), ("ワ", "笑嘴"),
    ("口角上げ", "嘴角上扬"), ("口角下げ", "嘴角下压"),
    ("口横広げ", "嘴部加宽"), ("口横狭め", "嘴部收窄"),
    (TEETH_CONTROL, "露牙（大笑 / 嫌弃）"),
    (SHADOW_CONTROL, "额头阴影"),
    (BLUSH_CONTROL, "双颊脸红"),
    ('上左', '左眉上挑'), ('上右', '右眉上挑'),
    (DISGUST_CONTROL, '闭嘴下垂（旧版）'),
    (GRIMACE_CONTROL, '嫌弃咧嘴'),
    (REFERENCE_EYES, '嫌弃眼形'), (REFERENCE_BROWS, '嫌弃眉形'),
    (REFERENCE_MOUTH, '嫌弃弧形露牙嘴'),
    (PAIR_CONTROLS[0], '嫌弃小开口'), (PAIR_CONTROLS[1], '嫌弃弧形露牙'),
    (SIDE_GLANCE_CONTROL, '参考侧视'),
    (SOFT_SMILE_CONTROL, '参考轻笑嘴角'),
    (BLANK_BROWS_CONTROL, '参考呆萌眉形'),
    (BELLE_WEARY_BROWS, '铃倦怠参考眉形'),
)
FACE_NAMES = tuple(name for name, _ in FACE_CONTROLS)
ARTISTIC_NAMES = frozenset((SHADOW_CONTROL, BLUSH_CONTROL, DISGUST_CONTROL, GRIMACE_CONTROL,
                           *REFERENCE_CONTROLS, *PAIR_CONTROLS, SIDE_GLANCE_CONTROL,
                           SOFT_SMILE_CONTROL, BLANK_BROWS_CONTROL, BELLE_WEARY_BROWS, '上左', '上右'))
MOUTH_NAMES = tuple(name for name in FACE_NAMES[14:] if name not in ARTISTIC_NAMES)
TRACKED_NAMES = tuple(name for name in FACE_NAMES if name not in ARTISTIC_NAMES)


def ensure_shadow_control(mesh):
    """Skin-attached shading mask; copy materials and leave source shaders intact."""
    keys = mesh.data.shape_keys
    if SHADOW_CONTROL not in keys.key_blocks:
        mesh.shape_key_add(name=SHADOW_CONTROL, from_mix=False).value = 0
    if not mesh.data.attributes.get(SHADOW_ATTRIBUTE) or mesh.get('ellen_shadow_mask_revision', 0) < 2:
        brow_slots = {i for i, m in enumerate(mesh.data.materials)
                      if m and m.name.rsplit('|', 1)[-1].strip() == '眉'}
        brow_verts = {v for p in mesh.data.polygons if p.material_index in brow_slots for v in p.vertices}
        if not brow_verts:
            raise RuntimeError("未找到眉毛，无法定位额头阴影")
        # Use rest coordinates so the mask follows skinning, not the world Z axis.
        basis = keys.key_blocks['Basis']
        brow_z = sum(basis.data[v].co.z for v in brow_verts) / len(brow_verts)
        fade_bottom, fade_top = brow_z - .060, brow_z - .018
        values = []
        for point in basis.data:
            t = max(0.0, min(1.0, (point.co.z - fade_bottom) / (fade_top - fade_bottom)))
            values.append(t*t*(3-2*t))
        attribute = mesh.data.attributes.get(SHADOW_ATTRIBUTE) or mesh.data.attributes.new(SHADOW_ATTRIBUTE, 'FLOAT', 'POINT')
        attribute.data.foreach_set('value', values)
        mesh['ellen_shadow_mask_revision'] = 2
    labels = {'颜', '颜2', '眉', '白目', '目', '睫', '二重影', '二重'}
    for index, original in enumerate(list(mesh.data.materials)):
        if original and original.get('ellen_shadow_key') == keys.name:
            original.node_tree.nodes['Ellen Forehead Tint'].inputs[2].default_value = (.09, .10, .135, 1)
            continue
        if not original or original.name.rsplit('|', 1)[-1].strip() not in labels:
            continue
        tree = original.node_tree
        target = next((n for n in tree.nodes if n.type == 'GROUP' and n.inputs.get('DData') and n.inputs['DData'].is_linked), None)
        if not target:
            continue
        material = original.copy()
        material.name = 'Ellen Expression | ' + original.name.rsplit('|', 1)[-1].strip()
        material['ellen_shadow_key'] = keys.name
        material['ellen_original_material'] = original.name
        mesh.data.materials[index] = material
        nodes, links = material.node_tree.nodes, material.node_tree.links
        target = nodes[target.name]
        source = target.inputs['DData'].links[0].from_socket
        mask = nodes.new('ShaderNodeAttribute')
        mask.name, mask.attribute_name = 'Ellen Forehead Mask', SHADOW_ATTRIBUTE
        strength = nodes.new('ShaderNodeValue')
        strength.name, strength.label = 'Ellen Forehead Strength', '额头阴影强度'
        driver = strength.outputs[0].driver_add('default_value').driver
        variable = driver.variables.new()
        variable.name, variable.type = 'shadow', 'SINGLE_PROP'
        variable.targets[0].id_type = 'KEY'
        variable.targets[0].id = keys
        variable.targets[0].data_path = keys.key_blocks[SHADOW_CONTROL].path_from_id('value')
        driver.expression = 'shadow'
        amount = nodes.new('ShaderNodeMath')
        amount.name, amount.operation = 'Ellen Shadow Amount', 'MULTIPLY'
        tint = nodes.new('ShaderNodeMixRGB')
        tint.name, tint.label, tint.blend_type = 'Ellen Forehead Tint', '额头冷灰渐变', 'MULTIPLY'
        tint.inputs[2].default_value = (.09, .10, .135, 1)
        links.new(mask.outputs['Fac'], amount.inputs[0])
        links.new(strength.outputs[0], amount.inputs[1])
        links.new(amount.outputs[0], tint.inputs[0])
        links.new(source, tint.inputs[1])
        links.new(tint.outputs[0], target.inputs['DData'])
        mask.location, strength.location = (-800, -350), (-800, -550)
        amount.location, tint.location = (-550, -350), (-350, -150)


def ensure_teeth_control(mesh):
    """Filter the original tooth faces; keep mesh, weights and materials intact."""
    teeth = next((m for m in mesh.data.materials if m and m.name.rsplit('|', 1)[-1].strip() == '齿'), None)
    if not teeth:
        raise RuntimeError("未找到独立的牙齿材质，无法安全隐藏牙齿")
    keys = mesh.data.shape_keys
    if TEETH_CONTROL not in keys.key_blocks:
        key = mesh.shape_key_add(name=TEETH_CONTROL, from_mix=False)
        key.value = 0
    modifier = mesh.modifiers.get('Ellen Teeth Visibility')
    if modifier:
        return
    group = bpy.data.node_groups.new('Ellen Teeth Visibility', 'GeometryNodeTree')
    group.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    group.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    nodes, links = group.nodes, group.links
    source = nodes.new('NodeGroupInput')
    output = nodes.new('NodeGroupOutput')
    material = nodes.new('GeometryNodeMaterialSelection')
    material.inputs['Material'].default_value = teeth
    hidden = nodes.new('ShaderNodeMath')
    hidden.operation = 'LESS_THAN'
    hidden.inputs[1].default_value = .5
    selection = nodes.new('FunctionNodeBooleanMath')
    selection.operation = 'AND'
    delete = nodes.new('GeometryNodeDeleteGeometry')
    delete.domain = 'FACE'
    links.new(source.outputs['Geometry'], delete.inputs['Geometry'])
    links.new(material.outputs['Selection'], selection.inputs[0])
    links.new(hidden.outputs[0], selection.inputs[1])
    links.new(selection.outputs[0], delete.inputs['Selection'])
    links.new(delete.outputs['Geometry'], output.inputs['Geometry'])
    modifier = mesh.modifiers.new('Ellen Teeth Visibility', 'NODES')
    modifier.node_group = group
    driver = hidden.inputs[0].driver_add('default_value').driver
    variable = driver.variables.new()
    variable.name, variable.type = 'show_teeth', 'SINGLE_PROP'
    variable.targets[0].id_type = 'KEY'
    variable.targets[0].id = keys
    variable.targets[0].data_path = keys.key_blocks[TEETH_CONTROL].path_from_id('value')
    driver.expression = 'show_teeth'
    mesh.modifiers.move(len(mesh.modifiers) - 1, 0)


def face_mesh(scene):
    rig = scene.xr_ellen.rig
    chosen = scene.xr_ellen.face_mesh
    if chosen and chosen.type == "MESH" and chosen.data.shape_keys:
        return chosen
    return next((obj for obj in scene.objects if obj.type == "MESH" and obj.data.shape_keys
                 and any(mod.type == "ARMATURE" and mod.object == rig for mod in obj.modifiers)), None)


def face_values(scene):
    mesh = face_mesh(scene)
    if not mesh:
        return {}
    keys = mesh.data.shape_keys.key_blocks
    return {name: keys[name].value for name in FACE_NAMES if name in keys}


def tracked_face_weights(face, settings):
    """Map calibrated XR MMD/ARKit channels to Ellen's existing shapes."""
    m = face.get("mmd", {})
    a = face.get("arkit", {})
    def weight(name):
        return max(0.0, min(1.0, m.get(name, 0.0)))
    def arkit(name):
        return max(0.0, min(1.0, a.get(name, 0.0)))
    values = {name: 0.0 for name in FACE_NAMES}
    # Prefer XR's calibrated MMD vowels; use ARKit only when that channel is absent.
    for name in ("あ", "い", "う", "え", "お"):
        values[name] = weight(name) * settings.mouth_gain
    if "あ" not in m:
        values["あ"] = arkit("JawOpen") * .7 * settings.mouth_gain
    if "う" not in m:
        values["う"] = arkit("MouthPucker") * .5 * settings.mouth_gain
    if "お" not in m:
        values["お"] = arkit("MouthFunnel") * .4 * settings.mouth_gain
    total = sum(values[name] for name in ("あ", "い", "う", "え", "お"))
    if total > 1:
        for name in ("あ", "い", "う", "え", "お"):
            values[name] /= total
    values["口角上げ"] = weight("口角上げ") if "口角上げ" in m else (arkit("MouthSmileLeft") + arkit("MouthSmileRight")) * .35
    values["口角下げ"] = weight("口角下げ") if "口角下げ" in m else (arkit("MouthFrownLeft") + arkit("MouthFrownRight")) * .25
    values["口横広げ"] = weight("にやり") * .35
    values["口横狭め"] = weight("ω") * .3
    values["笑い"] = weight("笑い")
    values["ワ"] = weight("ワ")
    left = weight("まばたきL") if "まばたきL" in m else arkit("EyeBlinkLeft")
    right = weight("まばたきR") if "まばたきR" in m else arkit("EyeBlinkRight")
    if "まばたき" in m and not any(k in m for k in ("まばたきL", "まばたきR")):
        left = right = weight("まばたき")
    values["ウィンク２"], values["ウィンク２右"] = left, right
    values["じと目"] = weight("じと目") * .5
    values["びっくり左"] = arkit("EyeWideLeft") * .35
    values["びっくり右"] = arkit("EyeWideRight") * .35
    values["上"] = weight("上") if "上" in m else arkit("BrowInnerUp") * .45
    values["下"] = weight("下") if "下" in m else (arkit("BrowDownLeft") + arkit("BrowDownRight")) * .4
    values["にこり"] = weight("にこり") if "にこり" in m else values["口角上げ"] * .7
    values["困る"] = weight("困る") if "困る" in m else arkit("BrowInnerUp") * .65
    values["怒り"] = weight("怒り") if "怒り" in m else (arkit("BrowDownLeft") + arkit("BrowDownRight")) * .4
    values["真面目"] = weight("真面目")
    # Teeth are reserved for a clear laugh: an open mouth plus a strong smile.
    laugh_signal = max(values["笑い"], values["口角上げ"], values["ワ"] * .8)
    mouth_signal = max(values["あ"], values["お"], values["ワ"])
    values[TEETH_CONTROL] = 1.0 if laugh_signal >= .7 and mouth_signal >= .45 else 0.0
    return {name: max(0.0, min(1.0, value)) for name, value in values.items()}


def validate_face(face):
    if not isinstance(face, dict) or not isinstance(face.get("active"), bool):
        raise ValueError("Invalid face payload")
    for kind in ("mmd", "arkit"):
        values = face.get(kind, {})
        if not isinstance(values, dict) or len(values) > 80:
            raise ValueError("Invalid face channels")
        for name, weight in values.items():
            if not isinstance(name, str) or len(name) > 80 or not isinstance(weight, (int, float)) or not math.isfinite(weight) or not 0 <= weight <= 1:
                raise ValueError("Invalid face weight")


def bone_converter():
    for addon in bpy.context.preferences.addons:
        if addon.module.endswith("mmd_tools"):
            return importlib.import_module(addon.module + ".core.vmd.importer").BoneConverter
    raise RuntimeError("Enable MMD Tools first")


def validate_packet(raw, token):
    packet = json.loads(raw)
    if (packet.get("protocol") != "XR_MMD_VMD_POSE" or packet.get("version") != 1
            or packet.get("token") != token or not isinstance(packet.get("seq"), int)):
        raise ValueError("Unexpected packet")
    if not isinstance(packet.get("bones"), list) or not 1 <= len(packet["bones"]) <= 128:
        raise ValueError("Invalid bone list")
    seen = set()
    for bone in packet["bones"]:
        name, pos, rot = bone["name"], bone["p"], bone["q"]
        if not isinstance(name, str) or len(name) > 80 or name in seen:
            raise ValueError("Invalid bone name")
        seen.add(name)
        if len(pos) != 3 or len(rot) != 4:
            raise ValueError("Invalid transform size")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in pos + rot):
            raise ValueError("Non-finite transform")
        if max(abs(v) for v in pos) > 10000 or not 0.5 < sum(v*v for v in rot) < 1.5:
            raise ValueError("Invalid transform magnitude")
    timestamp = packet.get("time")
    if timestamp is not None and (not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or timestamp < 0):
        raise ValueError("Invalid video time")
    if 'source_video' in packet and (not isinstance(packet['source_video'], str) or len(packet['source_video']) > 4096):
        raise ValueError('Invalid source video path')
    if "face" in packet:
        validate_face(packet["face"])
    hands = packet.get("mouth_hands")
    if hands is not None:
        if not isinstance(hands, dict) or set(hands)-{'左', '右'}:
            raise ValueError("Invalid hand targets")
        for row in hands.values():
            if not isinstance(row, dict):
                raise ValueError("Invalid hand target")
            tip = row.get('tip_from_mouth')
            if not isinstance(tip, list) or len(tip) != 3:
                raise ValueError("Invalid fingertip target")
            values = tip + [row.get('screen_distance'), row.get('finger_up')]
            if not all(isinstance(v, (int, float)) and math.isfinite(v) and abs(v) < 100 for v in values):
                raise ValueError("Invalid hand target value")
            if row['screen_distance'] < 0 or not -1.001 <= row['finger_up'] <= 1.001:
                raise ValueError("Invalid hand target range")
    return packet


def apply_packet(rig, converters, packet):
    matched = 0
    for value in packet["bones"]:
        item = converters.get(value["name"])
        if item is None:
            continue
        name, converter = item
        bone = rig.pose.bones[name]
        bone.location = converter.convert_location(value["p"])
        rotation = converter.convert_rotation(value["q"])
        if rotation.dot(bone.rotation_quaternion) < 0:
            rotation.negate()
        bone.rotation_quaternion = rotation
        matched += 1
    return matched


def redraw():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


class Recording:
    """Buffer source-time samples; build an action only after preview ends."""

    def __init__(self, session):
        self.session = session
        self.samples = {}
        self.face_samples = {}
        self.origin = self.last_time = None
        self.clock_origin = time.monotonic()
        self.source = None
        self.request_id = None
        self.fps = session.scene.render.fps / session.scene.render.fps_base

    def capture(self, packet):
        if self.request_id and packet.get("request_id") != self.request_id:
            return None
        source = packet.get("source")
        if self.source is not None and source != self.source:
            return "Source changed"
        if source == "video":
            timestamp = packet.get("time")
            if timestamp is None:
                return None
            if self.last_time is not None and timestamp < self.last_time - 0.25:
                return "Video loop finished"
            if packet.get("paused"):
                duration = packet.get("duration")
                if self.samples and isinstance(duration, (int, float)) and timestamp >= duration - 0.1:
                    return "Video finished"
                return None
            if self.last_time is not None and timestamp <= self.last_time:
                return None
        else:
            timestamp = time.monotonic() - self.clock_origin
        values = {}
        corrected = self.session.apply_pose(packet) if self.session.hand_correction_enabled(packet) else False
        for value in packet["bones"]:
            item = self.session.converters.get(value["name"])
            if item:
                name, converter = item
                if corrected:
                    bone = self.session.rig.pose.bones[name]
                    values[name] = (tuple(bone.location), tuple(bone.rotation_quaternion))
                else:
                    values[name] = (tuple(converter.convert_location(value["p"])),
                                    tuple(converter.convert_rotation(value["q"])))
        if not values:
            return None
        if self.origin is None:
            self.origin, self.source = timestamp, source
        frame = 1 + round((timestamp - self.origin) * self.fps)
        self.samples[frame] = values
        self.face_samples[frame] = face_values(self.session.scene)
        self.last_time = timestamp
        settings = self.session.scene.xr_ellen
        settings.recorded_frames = len(self.samples)
        settings.record_status = f"Recording: {len(self.samples)} samples / {(frame - 1) / self.fps:.2f} s"
        return None

    def bake(self):
        if not self.samples:
            raise RuntimeError("No motion received; nothing to save")
        scene, rig = self.session.scene, self.session.rig
        action = bpy.data.actions.new("XR Take " + datetime.now().strftime("%Y%m%d_%H%M%S"))
        action.use_fake_user = True
        rig.animation_data_create().action = action
        # Fix the unsent bones in their neutral preview pose as well.
        channels = {name: [[], []] for name in self.session.saved_pose}
        previous = {}
        for frame, values in sorted(self.samples.items()):
            for name, (position, rotation) in values.items():
                if name in previous and sum(a*b for a, b in zip(rotation, previous[name])) < 0:
                    rotation = tuple(-v for v in rotation)
                previous[name] = rotation
                channels[name][0].append((frame, position))
                channels[name][1].append((frame, rotation))
        for name, (positions, rotations) in channels.items():
            bone = rig.pose.bones[name]
            bone.rotation_mode = "QUATERNION"
            for prop, samples, default in (("location", positions, (0, 0, 0)),
                                           ("rotation_quaternion", rotations, (1, 0, 0, 0))):
                if not samples:
                    samples = [(1, default)]
                path = bone.path_from_id(prop)
                for index in range(len(default)):
                    curve = action.fcurve_ensure_for_datablock(rig, path, index=index)
                    curve.keyframe_points.add(len(samples))
                    curve.keyframe_points.foreach_set("co", [v for frame, value in samples for v in (frame, value[index])])
                    for key in curve.keyframe_points:
                        key.interpolation = "LINEAR"
                    curve.update()
        action["xr_source"] = self.source or "unknown"
        action["xr_source_start_seconds"] = self.origin
        action["xr_sample_count"] = len(self.samples)
        if scene.xr_ellen.contact_mouth or scene.xr_ellen.contact_head:
            action['xr_contact_configuration'] = json.dumps(contact_signature(scene.xr_ellen), ensure_ascii=False)
        scene.frame_start, scene.frame_end = 1, max(self.samples)
        scene.xr_ellen.take = action
        self.session.completed_face_take = self.bake_face(action.name + " Face")
        scene.xr_ellen.record_status = f"Recorded: {action.name} ({scene.frame_end} frames)"
        return action

    def bake_face(self, name):
        mesh = face_mesh(self.session.scene)
        if not mesh or not any(self.face_samples.values()):
            return None
        keys = mesh.data.shape_keys
        action = bpy.data.actions.new(name)
        action.use_fake_user = True
        keys.animation_data_create().action = action
        for key_name in FACE_NAMES:
            if key_name not in keys.key_blocks:
                continue
            samples = [(frame, values[key_name]) for frame, values in sorted(self.face_samples.items()) if key_name in values]
            curve = action.fcurve_ensure_for_datablock(keys, keys.key_blocks[key_name].path_from_id("value"))
            curve.keyframe_points.add(len(samples))
            curve.keyframe_points.foreach_set("co", [v for pair in samples for v in pair])
            for key in curve.keyframe_points:
                key.interpolation = "LINEAR"
            curve.update()
        self.session.scene.xr_ellen.face_take = action
        keys.animation_data.action = None
        return action


class Session:
    def __init__(self, scene):
        settings = scene.xr_ellen
        rig = settings.rig
        if rig is None or rig.type != "ARMATURE":
            raise RuntimeError("Choose Ellen's armature")
        root = Path(bpy.path.abspath(settings.xr_root))
        if not (root / "SystemAnimator_webkit.html").is_file():
            raise RuntimeError("XR Animator gadget folder is invalid")
        cls = bone_converter()
        converters = {bone.mmd_bone.name_j: (bone.name, cls(bone, settings.scale))
                      for bone in rig.pose.bones if getattr(bone, "mmd_bone", None) and bone.mmd_bone.name_j}
        if len(converters) < 20:
            raise RuntimeError("The target has no usable MMD bone names")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.socket.bind(("127.0.0.1", PORT))
            self.socket.setblocking(False)
        except Exception:
            self.socket.close()
            raise
        self.scene, self.rig, self.converters = scene, rig, converters
        self.flag = root / "TEMP/blender_bridge_enabled.json"
        self.token = secrets.token_hex(16)
        self.request = None
        self.saved_pose = {bone.name: (bone.matrix_basis.copy(), bone.rotation_mode) for bone in rig.pose.bones}
        self.action = rig.animation_data.action if rig.animation_data else None
        self.slot = rig.animation_data.action_slot if rig.animation_data else None
        self.tracks = [(track, track.mute) for track in rig.animation_data.nla_tracks] if rig.animation_data else []
        self.had_animation = rig.animation_data is not None
        self.was_fake = self.action.use_fake_user if self.action else False
        self.received = self.applied = self.rejected = 0
        self.last_received = self.last_write = self.last_report = 0
        self.packet = None
        self.root_origin = None
        self.root_source = None
        self.recording = None
        self.completed_take = None
        self.completed_face_take = None
        mesh = face_mesh(scene)
        if mesh:
            ensure_teeth_control(mesh)
            ensure_shadow_control(mesh)
        self.face_keys = mesh.data.shape_keys if mesh else None
        self.hand_names = {name: rig.pose.bones[value[0]] for name, value in converters.items()}
        bpy.context.view_layer.update()
        self.hand_anchor = mouth_anchor(rig, mesh, self.hand_names['頭']) if '頭' in self.hand_names else None
        self.crown_anchor = head_contact_anchor(rig, self.hand_names['頭']) if '頭' in self.hand_names else None
        keys = self.face_keys
        self.saved_face = {k.name: k.value for k in keys.key_blocks} if keys else {}
        self.face_action = keys.animation_data.action if keys and keys.animation_data else None
        self.face_slot = keys.animation_data.action_slot if keys and keys.animation_data else None
        self.face_tracks = [(t, t.mute) for t in keys.animation_data.nla_tracks] if keys and keys.animation_data else []
        self.face_detached = False
        self.smoothed_face = {}
        self.face_clock = None
        self.last_face_seen = 0
        self.report_path = Path(bpy.data.filepath).parent / "live_receiver_status.json" if bpy.data.filepath else None
        self.write_lease()
        if self.action:
            self.action.use_fake_user = True
            rig.animation_data.action = None
        for track, _ in self.tracks:
            track.mute = True
        for bone in rig.pose.bones:
            bone.rotation_mode = "QUATERNION"
            bone.matrix_basis = Matrix.Identity(4)

    def hand_correction_enabled(self, packet):
        settings = self.scene.xr_ellen
        return bool(contact_segments(settings, packet) or
                    (settings.mouth_hand_correction and settings.mouth_hand_strength > 0
                    and self.hand_anchor is not None and packet.get('mouth_hands')))

    def apply_pose(self, packet):
        matched = apply_packet(self.rig, self.converters, packet)
        if self.hand_correction_enabled(packet):
            bpy.context.view_layer.update()
            settings = self.scene.xr_ellen
            segments = contact_segments(settings, packet)
            reserved = {side for _, side, _ in segments}
            automatic = {s: row for s, row in (packet.get('mouth_hands') or {}).items() if s not in reserved}
            self.hand_weights = correct_mouth_hands(self.rig, self.hand_names, self.hand_anchor,
                automatic, settings.mouth_hand_strength if settings.mouth_hand_correction else 0,
                settings.mouth_hand_depth, settings.mouth_hand_height)
            self.hand_weights.update(correct_contact_hands(self.rig, self.hand_names, self.hand_anchor,
                self.crown_anchor, settings, packet, segments))
        else:
            self.hand_weights = {}
        return matched

    def detach_face(self):
        if self.face_detached or not self.face_keys:
            return
        if self.face_action:
            self.face_action.use_fake_user = True
        if self.face_keys.animation_data:
            self.face_keys.animation_data.action = None
        for track, _ in self.face_tracks:
            track.mute = True
        self.face_detached = True

    def apply_face(self, packet):
        settings = self.scene.xr_ellen
        if settings.face_mode == "OFF" or not self.face_keys:
            settings.face_status = "手动表情"
            return
        face = packet.get("face")
        active = bool(face and face.get("active"))
        now = time.monotonic()
        if active:
            self.last_face_seen = now
        elif not self.face_detached or now - self.last_face_seen < .35:
            settings.face_status = "等待 XR 面部数据"
            return
        if packet.get("paused"):
            return
        self.detach_face()
        values = tracked_face_weights(face or {}, settings) if active else dict.fromkeys(FACE_NAMES, 0.0)
        controlled = MOUTH_NAMES if settings.face_mode == "LIPS" else TRACKED_NAMES
        delta = min(.15, max(.001, now - self.face_clock)) if self.face_clock else .033
        factor = 1 - math.exp(-delta / max(.001, settings.face_smoothing))
        self.face_clock = now
        for name in controlled:
            key = self.face_keys.key_blocks.get(name)
            if key:
                value = self.smoothed_face.get(name, key.value)
                value += (values[name] - value) * factor
                self.smoothed_face[name] = value
                key.value = value
        settings.face_status = "XR 表情已连接" if active else "面部丢失，回到自然"
    def write_lease(self):
        self.flag.parent.mkdir(parents=True, exist_ok=True)
        value = {"version": 1, "enabled": True, "port": PORT, "token": self.token,
                 "updated_ms": int(time.time() * 1000), "request": self.request,
                 "face_mode": self.scene.xr_ellen.face_mode}
        temporary = self.flag.with_suffix(".tmp")
        temporary.write_text(json.dumps(value), encoding="utf-8")
        try:
            temporary.replace(self.flag)
        except PermissionError:
            # Windows can briefly deny replacement while XR reads the lease.
            # Retry on the next timer tick; do not discard the live pose/take.
            now = time.monotonic()
            if getattr(self, "lease_failure_since", None) is None:
                self.lease_failure_since = now
            if now - self.lease_failure_since > 3:
                raise
            return
        self.lease_failure_since = None
        self.last_write = time.monotonic()

    def command(self, kind, path=None):
        if self.recording and kind == "video":
            raise RuntimeError("Finish recording before changing the video")
        self.request = {"id": secrets.token_hex(8), "type": kind}
        if path is not None:
            self.request["path"] = str(path)
        self.write_lease()

    def placement_packet(self, packet):
        """Keep the room placement; XR's absolute tracking origin is not a stage position."""
        if self.scene.xr_ellen.motion_mode == "UPPER":
            # Include omitted lower-body channels to clear a previous full-body pose.
            bones = [bone for bone in packet["bones"] if bone["name"] not in FIXED_LOWER_BONES]
            bones.extend({"name": name, "p": [0, 0, 0], "q": [0, 0, 0, 1]}
                         for name in sorted(FIXED_LOWER_BONES) if name in self.converters)
            return dict(packet, bones=bones)
        if not self.scene.get("EllenHomeProject"):
            return packet
        source = (packet.get("source"), packet.get("request_id"))
        if source != self.root_source:
            self.root_source = source
            self.root_origin = None
        bones = []
        for bone in packet["bones"]:
            if bone["name"] == "全ての親":
                if self.root_origin is None:
                    self.root_origin = tuple(bone["p"])
                bone = dict(bone, p=[value - origin for value, origin
                                     in zip(bone["p"], self.root_origin)])
            bones.append(bone)
        return dict(packet, bones=bones)

    def begin_recording(self):
        if self.recording:
            raise RuntimeError("Already recording")
        if self.packet is None or time.monotonic() - self.last_received > 2:
            raise RuntimeError("Wait for live motion before recording")
        self.recording = Recording(self)
        self.scene.xr_ellen.recording = True
        self.scene.xr_ellen.recorded_frames = 0
        self.scene.xr_ellen.record_status = "Waiting for moving source frames"
        if self.packet.get("source") == "video":
            self.command("restart" if self.scene.xr_ellen.from_start else "play")
            self.recording.request_id = self.request["id"]

    def finish_recording(self):
        if self.recording is None:
            return self.completed_take
        recording, self.recording = self.recording, None
        self.scene.xr_ellen.recording = False
        if not recording.samples:
            self.scene.xr_ellen.record_status = "No frames recorded"
            return None
        self.completed_take = recording.bake()
        self.rig.animation_data.action = None
        return self.completed_take

    def tick(self):
        latest = None
        changed = False
        for _ in range(64):
            try:
                raw, address = self.socket.recvfrom(65535)
            except BlockingIOError:
                break
            if address[0] != "127.0.0.1":
                continue
            try:
                latest = self.placement_packet(validate_packet(raw, self.token))
                self.received += 1
                if self.recording:
                    self.apply_face(latest)
                    reason = self.recording.capture(latest)
                    if reason:
                        self.finish_recording()
            except (ValueError, TypeError, KeyError, OverflowError):
                self.rejected += 1
        now = time.monotonic()
        if latest is not None:
            self.packet = latest
            self.last_received = now
            # XR sends heartbeats while paused. Ignore sequence/send time, but still
            # apply a changed pose, source, seek or tracking setting immediately.
            paused_pose = (latest.get("source"), latest.get("request_id"), latest.get("time"),
                           latest["bones"], latest.get("face"), self.scene.xr_ellen.motion_mode,
                           self.scene.xr_ellen.face_mode, self.scene.xr_ellen.live_physics,
                           latest.get('mouth_hands'), self.scene.xr_ellen.mouth_hand_correction,
                           self.scene.xr_ellen.mouth_hand_strength, self.scene.xr_ellen.mouth_hand_depth,
                           self.scene.xr_ellen.mouth_hand_height, latest.get('source_video'),
                           contact_signature(self.scene.xr_ellen))
            duplicate = bool(latest.get("paused") and paused_pose == getattr(self, "paused_pose", None))
            self.paused_pose = paused_pose if latest.get("paused") else None
            if not duplicate:
                if not self.recording:
                    self.apply_face(latest)
                self.scene.xr_ellen.matched = self.apply_pose(latest)
                self.applied += 1
                self.scene.xr_ellen.source_time = latest.get("time") or 0
                self.scene.xr_ellen.status = ("Video paused" if latest.get("paused") else "Live video") if latest.get("source") == "video" else "Live camera"
                physics_updated = False
                if self.scene.get("EllenHomeProject") and self.scene.xr_ellen.live_physics:
                    import ellen_home_controls
                    physics_updated = ellen_home_controls.live_physics_step(self.scene, latest)
                elif self.scene.get("EllenHomeProject"):
                    import ellen_home_controls
                    ellen_home_controls._last_time = None
                if not physics_updated:
                    bpy.context.view_layer.update()
                changed = True
        elif now - self.last_received > 2:
            waiting = "Waiting for XR mocap data"
            if self.scene.xr_ellen.status != waiting:
                self.scene.xr_ellen.status = waiting
                changed = True
        if now - self.last_write > 1:
            self.write_lease()
        if self.report_path and now - self.last_report > 1:
            self.last_report = now
            self.scene.xr_ellen.received = self.received
            changed = True
            report = {"received": self.received, "applied": self.applied, "rejected": self.rejected,
                      "last_pose_age": now - self.last_received if self.last_received else None,
                      "matched": self.scene.xr_ellen.matched, "time": self.scene.xr_ellen.source_time,
                      "pose": {name: list(self.rig.pose.bones[name].rotation_quaternion)
                               for name in ["Head", "Arm_L", "Wrist_L", "IndexFinger1_L"] if name in self.rig.pose.bones},
                      "root_origin": self.root_origin,
                      "motion_mode": self.scene.xr_ellen.motion_mode,
                      "hand_weights": getattr(self, 'hand_weights', {}), "last_packet": self.packet}
            self.report_path.write_text(json.dumps(report, ensure_ascii=True), encoding="utf-8")
        if changed:
            redraw()
        return 1 / 30

    def close(self):
        self.finish_recording()
        self.socket.close()
        try:
            current = json.loads(self.flag.read_text(encoding="utf-8"))
            if current.get("token") == self.token:
                self.flag.write_text(json.dumps({"version": 1, "enabled": False}), encoding="utf-8")
        except (OSError, ValueError):
            pass
        for name, (basis, mode) in self.saved_pose.items():
            bone = self.rig.pose.bones.get(name)
            if bone:
                bone.rotation_mode = mode
                bone.matrix_basis = basis
        if self.face_keys and self.face_detached:
            for name, value in self.saved_face.items():
                self.face_keys.key_blocks[name].value = value
            self.face_keys.animation_data_create().action = self.face_action
            if self.face_action:
                self.face_keys.animation_data.action_slot = self.face_slot
            for track, mute in self.face_tracks:
                track.mute = mute
        if self.face_keys and self.completed_face_take:
            if self.face_action:
                self.face_action.use_fake_user = True
            self.face_keys.animation_data_create().action = self.completed_face_take
            for track, _ in self.face_tracks:
                track.mute = True
        if self.had_animation:
            self.rig.animation_data.action = self.action
            if self.action:
                self.rig.animation_data.action_slot = self.slot
                self.action.use_fake_user = self.was_fake
            for track, mute in self.tracks:
                track.mute = mute
        if self.completed_take:
            if self.action:
                self.action.use_fake_user = True
            self.rig.animation_data_create().action = self.completed_take
            for track, _ in self.tracks:
                track.mute = True
            for bone in self.rig.pose.bones:
                bone.rotation_mode = "QUATERNION"
            self.scene.frame_set(1)
            self.scene.xr_ellen.status = "Disconnected - recorded action active"
        else:
            self.scene.xr_ellen.status = "Disconnected - original action restored"
        self.scene.xr_ellen.received = 0
        redraw()


def tick():
    global _runtime
    if _runtime is None:
        return None
    try:
        interval = _runtime.tick()
        if _runtime.completed_take and _runtime.recording is None:
            stop()
            return None
        return interval
    except Exception as error:
        scene = _runtime.scene
        stop()
        scene.xr_ellen.status = "Error: " + str(error)
        return None


def start(scene):
    global _runtime
    if _runtime is not None:
        raise RuntimeError("A preview session is already connected")
    _runtime = Session(scene)
    scene.xr_ellen.status = "Waiting for XR mocap data"
    bpy.app.timers.register(tick, first_interval=0.1)


def stop():
    global _runtime
    if bpy.app.timers.is_registered(tick):
        bpy.app.timers.unregister(tick)
    if _runtime is not None:
        session, _runtime = _runtime, None
        session.close()


@persistent
def before_load(_):
    stop()


@persistent
def before_save(_):
    # Finish a take before disconnecting; temporary unrecorded poses are restored.
    stop()


class XRSettings(bpy.types.PropertyGroup):
    rig: PointerProperty(name="Ellen armature", type=bpy.types.Object, poll=lambda self, obj: obj.type == "ARMATURE")
    xr_root: StringProperty(name="XR folder", subtype="DIR_PATH")
    scale: FloatProperty(name="MMD scale", default=0.08, min=0.0001, max=10)
    status: StringProperty(default="Disconnected")
    received: IntProperty(default=0)
    matched: IntProperty(default=0)
    source_time: FloatProperty(default=0)
    from_start: BoolProperty(name="Record video from start", default=True)
    live_physics: BoolProperty(name="实时物理", default=True,
        description="仅控制 XR 实时预览的物理计算；关闭可减轻动捕负担，不改变离线动画的物理与碰撞设置")
    video_lips: BoolProperty(name="导入视频时启用嘴型", default=True,
        description="打开新视频时自动启用 XR 嘴型，保留手动眉眼；关闭后保持当前表情来源")
    mouth_hand_correction: BoolProperty(name="捂嘴位置校正（试验）", default=False,
        description="手指接近嘴部时按艾莲比例调整手臂；关闭即恢复原映射。单目视频可能误判接触意图")
    mouth_hand_strength: FloatProperty(name="校正强度", default=1, min=0, max=1)
    mouth_hand_depth: FloatProperty(name="前后距离保留", default=.2, min=0, max=1,
        description="越小手越靠近脸，越大越保留 XR 中的前后距离")
    mouth_hand_height: FloatProperty(name="手指上下微调", default=.025, min=-.1, max=.1, subtype='DISTANCE',
        description="相对嘴部的额外高度；正值上移，负值下移，不改变骨骼长度")
    contact_video: StringProperty(name="校正视频", subtype='FILE_PATH',
        description="仅作用于此视频；更换视频后需要重新设置时间段")
    contact_strength: FloatProperty(name="接触校正强度", default=1, min=0, max=1)
    contact_fade: FloatProperty(name="过渡秒数", default=.25, min=0, max=2)
    contact_mouth: BoolProperty(name="分段捂嘴", default=False)
    contact_mouth_side: EnumProperty(name="捂嘴用手", items=(('左', 'XR 左手', ''), ('右', 'XR 右手', '')), default='左')
    contact_mouth_start: FloatProperty(name="开始秒", default=0, min=0)
    contact_mouth_end: FloatProperty(name="结束秒", default=1, min=0)
    contact_mouth_height: FloatProperty(name="嘴部上下", default=.012, min=-.1, max=.1, subtype='DISTANCE')
    contact_mouth_depth: FloatProperty(name="离嘴距离", default=.045, min=.01, max=.2, subtype='DISTANCE')
    contact_head: BoolProperty(name="分段挠头", default=False)
    contact_head_side: EnumProperty(name="挠头用手", items=(('左', 'XR 左手', ''), ('右', 'XR 右手', '')), default='左')
    contact_head_start: FloatProperty(name="开始秒", default=0, min=0)
    contact_head_end: FloatProperty(name="结束秒", default=1, min=0)
    contact_head_height: FloatProperty(name="头顶上下", default=.005, min=-.1, max=.15, subtype='DISTANCE')
    contact_head_depth: FloatProperty(name="头顶前后", default=0, min=-.15, max=.15, subtype='DISTANCE')
    contact_head_side_offset: FloatProperty(name="头顶左右", default=.055, min=0, max=.15, subtype='DISTANCE')
    motion_mode: EnumProperty(name="动捕范围", items=(
        ("UPPER", "上半身（固定腿部）", "锁定整体位置、骨盆和腿部；跟随躯干、头部及手势"),
        ("FULL", "全身（允许移动）", "跟随全身动作及相对位移，适合走路和舞蹈")), default="UPPER")
    recording: BoolProperty(default=False, options={"SKIP_SAVE"})
    recorded_frames: IntProperty(default=0)
    record_status: StringProperty(default="")
    take: PointerProperty(name="Recorded action", type=bpy.types.Action)
    face_mesh: PointerProperty(name="面部网格", type=bpy.types.Object, poll=lambda self, obj: obj.type == "MESH" and obj.data.shape_keys is not None)
    face_mode: EnumProperty(name="表情来源", items=(
        ("OFF", "手动", "使用表情按钮和滑块"),
        ("LIPS", "XR 嘴型", "保留手动眼睛和眉毛，仅跟随嘴型"),
        ("FULL", "XR 全部表情", "跟随 XR 的眼睛、眉毛和嘴型")), default="OFF")
    mouth_gain: FloatProperty(name="嘴型幅度", default=.85, min=0, max=2)
    face_smoothing: FloatProperty(name="平滑时间", default=.06, min=.001, max=.5, subtype="TIME")
    face_status: StringProperty(default="手动表情")
    face_take: PointerProperty(name="Recorded face action", type=bpy.types.Action)


class XRRecord(bpy.types.Operator):
    bl_idname = "xr_ellen.record"
    bl_label = "Record Motion"

    def execute(self, context):
        try:
            if _runtime is None:
                raise RuntimeError("Connect XR and load motion first")
            _runtime.begin_recording()
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        return {"FINISHED"}


class XRFinishRecord(bpy.types.Operator):
    bl_idname = "xr_ellen.finish_record"
    bl_label = "Finish Recording"

    def execute(self, context):
        stop()
        return {"FINISHED"}


class XRSaveMotion(bpy.types.Operator, ExportHelper):
    bl_idname = "xr_ellen.save_motion"
    bl_label = "Save Motion .blend"
    filename_ext = ".blend"
    filter_glob: StringProperty(default="*.blend", options={"HIDDEN"})

    def invoke(self, context, event):
        try:
            if _runtime:
                _runtime.finish_recording()
            if context.scene.xr_ellen.take is None:
                raise RuntimeError("Click Record Motion and record some frames first")
            stop()
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        directory = Path(bpy.data.filepath).parent if bpy.data.filepath else Path.home()
        self.filepath = str(directory / ("Ellen_XR_Recorded_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".blend"))
        return ExportHelper.invoke(self, context, event)

    def execute(self, context):
        try:
            if _runtime:
                _runtime.finish_recording()
            if context.scene.xr_ellen.take is None:
                raise RuntimeError("No recorded action to save")
            stop()
            bpy.ops.wm.save_as_mainfile(filepath=bpy.path.ensure_ext(self.filepath, ".blend"))
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report({"INFO"}, "Recorded motion and scene saved")
        return {"FINISHED"}


class XRConnect(bpy.types.Operator):
    bl_idname = "xr_ellen.connect"
    bl_label = "Connect XR"

    def execute(self, context):
        try:
            start(context.scene)
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        return {"FINISHED"}


class XRDisconnect(bpy.types.Operator):
    bl_idname = "xr_ellen.disconnect"
    bl_label = "Disconnect / Restore"

    def execute(self, context):
        stop()
        return {"FINISHED"}


class XRVideo(bpy.types.Operator, ImportHelper):
    bl_idname = "xr_ellen.video"
    bl_label = "Open Video in XR"
    filename_ext = ".mp4"
    filter_glob: StringProperty(default="*.mp4;*.mov;*.mkv;*.webm;*.avi", options={"HIDDEN"})

    def execute(self, context):
        try:
            path = Path(self.filepath)
            if not path.is_file() or path.suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
                raise RuntimeError("Choose an existing video file")
            if context.scene.xr_ellen.video_lips and context.scene.xr_ellen.face_mode == "OFF":
                context.scene.xr_ellen.face_mode = "LIPS"
            if _runtime is None:
                start(context.scene)
            _runtime.command("video", path.resolve())
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        return {"FINISHED"}


class XRPlayback(bpy.types.Operator):
    bl_idname = "xr_ellen.playback"
    bl_label = "XR Video Playback"
    command: StringProperty(default="play")

    def execute(self, context):
        if _runtime is None:
            return {"CANCELLED"}
        _runtime.command(self.command)
        return {"FINISHED"}


class XRCamera(bpy.types.Operator):
    bl_idname = "xr_ellen.camera"
    bl_label = "Camera Preview"

    def execute(self, context):
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.spaces.active.region_3d.view_perspective = "CAMERA"
                area.spaces.active.overlay.show_overlays = False
        return {"FINISHED"}


class XRViewport(bpy.types.Operator):
    bl_idname = "xr_ellen.viewport"
    bl_label = "切换预览显示"
    bl_description = "仅切换视口显示，不更改最终渲染、材质、相机或碰撞配置"
    mode: EnumProperty(items=(("SOLID", "实体预览", ""), ("MATERIAL", "材质预览", "")))

    def execute(self, context):
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.spaces.active.shading.type = self.mode
        return {"FINISHED"}


class XRPanel(bpy.types.Panel):
    bl_idname = "XR_PT_ellen_preview"
    bl_label = "XR Animator Live Preview"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "XR Preview"

    @classmethod
    def poll(cls, context):
        return not context.scene.get('CharacterStudio')

    def draw(self, context):
        layout, settings = self.layout, context.scene.xr_ellen
        layout.label(text=settings.status, icon="LINKED" if _runtime else "UNLINKED")
        row = layout.row(align=True)
        row.operator("xr_ellen.connect", icon="LINKED") if _runtime is None else row.operator("xr_ellen.disconnect", icon="UNLINKED")
        layout.operator("xr_ellen.video", icon="FILE_MOVIE")
        row = layout.row(align=True)
        row.enabled = _runtime is not None
        row.operator("xr_ellen.playback", text="Play", icon="PLAY").command = "play"
        row.operator("xr_ellen.playback", text="Pause", icon="PAUSE").command = "pause"
        layout.operator("xr_ellen.camera", icon="VIEW_CAMERA")
        row = layout.row()
        row.enabled = not settings.recording
        row.prop(settings, "motion_mode")
        layout.prop(settings, "face_mode")
        layout.prop(settings, "mouth_hand_correction")
        if settings.mouth_hand_correction:
            layout.prop(settings, "mouth_hand_strength", slider=True)
            layout.prop(settings, "mouth_hand_depth", slider=True)
            layout.prop(settings, "mouth_hand_height")
        layout.prop(settings, 'contact_mouth')
        layout.prop(settings, 'contact_head')
        if settings.contact_mouth or settings.contact_head:
            layout.prop(settings, 'contact_video')
            layout.prop(settings, 'contact_strength', slider=True)
            layout.prop(settings, 'contact_fade')
            for kind in ('mouth', 'head'):
                if getattr(settings, 'contact_'+kind):
                    layout.label(text='捂嘴时间段' if kind == 'mouth' else '挠头时间段')
                    layout.prop(settings, 'contact_'+kind+'_side')
                    layout.prop(settings, 'contact_'+kind+'_start')
                    layout.prop(settings, 'contact_'+kind+'_end')
                    layout.prop(settings, 'contact_'+kind+'_height')
                    layout.prop(settings, 'contact_'+kind+'_depth')
                    if kind == 'head':
                        layout.prop(settings, 'contact_head_side_offset')
        if context.scene.get("EllenHomeProject"):
            layout.prop(settings, "live_physics")
        row = layout.row(align=True)
        row.operator("xr_ellen.viewport", text="实体", icon="SHADING_SOLID").mode = "SOLID"
        row.operator("xr_ellen.viewport", text="材质", icon="SHADING_TEXTURE").mode = "MATERIAL"
        layout.separator()
        row = layout.row()
        row.enabled = not settings.recording
        row.prop(settings, "from_start")
        row = layout.row()
        if settings.recording:
            row.operator("xr_ellen.finish_record", icon="PAUSE")
        else:
            row.enabled = _runtime is not None
            row.operator("xr_ellen.record", icon="REC")
        row = layout.row()
        row.enabled = settings.take is not None or settings.recording
        row.operator("xr_ellen.save_motion", icon="FILE_TICK")
        if settings.record_status:
            layout.label(text=settings.record_status)
        layout.label(text=f"Video: {settings.source_time:.2f} s")
        layout.label(text=f"Packets: {settings.received} | Bones: {settings.matched}")
        column = layout.column()
        column.enabled = _runtime is None
        column.prop(settings, "rig")
        column.prop(settings, "xr_root")
        column.prop(settings, "scale")


class XRFacePreset(bpy.types.Operator):
    bl_idname = "xr_ellen.face_preset"
    bl_label = "切换表情"
    bl_options = {"REGISTER", "UNDO"}
    preset: StringProperty(default="neutral")

    def execute(self, context):
        if context.scene.get('CharacterStudio'):
            return bpy.ops.ellen_expression.choose(preset=self.preset)
        mesh = face_mesh(context.scene)
        if not mesh or self.preset not in FACE_PRESETS:
            self.report({"ERROR"}, "未找到当前角色的面部形态键")
            return {"CANCELLED"}
        ensure_teeth_control(mesh)
        ensure_shadow_control(mesh)
        if BLUSH_CONTROL in FACE_PRESETS[self.preset][1]:
            ensure_blush_control(mesh)
        if DISGUST_CONTROL in FACE_PRESETS[self.preset][1]:
            ensure_disgust_control(mesh)
        if GRIMACE_CONTROL in FACE_PRESETS[self.preset][1]:
            ensure_disgust_grimace(mesh)
        if any(name in FACE_PRESETS[self.preset][1] for name in REFERENCE_CONTROLS):
            ensure_reference_disgust(mesh)
        if any(name in FACE_PRESETS[self.preset][1] for name in PAIR_CONTROLS):
            ensure_disgust_pair(mesh)
        if SIDE_GLANCE_CONTROL in FACE_PRESETS[self.preset][1]:
            ensure_reference_side_glance(mesh)
        if SOFT_SMILE_CONTROL in FACE_PRESETS[self.preset][1]:
            ensure_reference_soft_smile(mesh)
        if BLANK_BROWS_CONTROL in FACE_PRESETS[self.preset][1]:
            ensure_reference_blank_brows(mesh)
        if BELLE_WEARY_BROWS in FACE_PRESETS[self.preset][1]:
            ensure_belle_weary_brows(mesh)
        if _runtime:
            _runtime.detach_face()
            _runtime.smoothed_face.clear()
        else:
            keys = mesh.data.shape_keys
            if keys.animation_data:
                if keys.animation_data.action:
                    keys.animation_data.action.use_fake_user = True
                    keys.animation_data.action = None
                for track in keys.animation_data.nla_tracks:
                    track.mute = True
        keys = mesh.data.shape_keys.key_blocks
        weights = FACE_PRESETS[self.preset][1]
        for name in FACE_NAMES:
            if name in keys:
                keys[name].value = weights.get(name, 0)
        bpy.context.view_layer.update()
        redraw()
        return {"FINISHED"}


class XRFaceKeyframe(bpy.types.Operator):
    bl_idname = "xr_ellen.face_keyframe"
    bl_label = "插入表情关键帧"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        if _runtime:
            self.report({"ERROR"}, "实时预览请用 Record Motion；手动关键帧请先断开连接")
            return {"CANCELLED"}
        mesh = face_mesh(context.scene)
        if not mesh:
            return {"CANCELLED"}
        keys = mesh.data.shape_keys
        keys.animation_data_create()
        if context.scene.xr_ellen.face_take:
            keys.animation_data.action = context.scene.xr_ellen.face_take
        for name in FACE_NAMES:
            key = keys.key_blocks.get(name)
            if key:
                key.keyframe_insert("value", frame=context.scene.frame_current, group="Ellen Face")
        context.scene.xr_ellen.face_take = keys.animation_data.action
        keys.animation_data.action.use_fake_user = True
        return {"FINISHED"}


class XRFacePanel(bpy.types.Panel):
    bl_idname = "XR_PT_ellen_face"
    bl_label = "角色表情"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "XR Preview"

    @classmethod
    def poll(cls, context):
        return not context.scene.get('CharacterStudio')

    def draw(self, context):
        layout, settings = self.layout, context.scene.xr_ellen
        layout.prop(settings, "face_mesh", text="控制角色")
        layout.prop(settings, "face_mode", expand=True)
        layout.prop(settings, "video_lips")
        layout.label(text=settings.face_status)
        if settings.face_mode != "OFF":
            layout.prop(settings, "mouth_gain", slider=True)
            layout.prop(settings, "face_smoothing")
        mesh = face_mesh(context.scene)
        if not mesh:
            layout.prop(settings, "face_mesh")
            return
        column = layout.column()
        column.enabled = _runtime is None or settings.face_mode != "FULL"
        row = None
        for index, (name, (label, _)) in enumerate(FACE_PRESETS.items()):
            if index % 2 == 0:
                row = column.row(align=True)
            row.operator("xr_ellen.face_preset", text=label).preset = name
        row = layout.row()
        row.enabled = _runtime is None
        row.operator("xr_ellen.face_keyframe", icon="KEY_HLT")
        keys = mesh.data.shape_keys.key_blocks
        for index, (name, label) in enumerate(FACE_CONTROLS):
            if index in (0, 8, 14):
                layout.separator()
            key = keys.get(name)
            if key:
                row = layout.row()
                row.enabled = _runtime is None or name == SHADOW_CONTROL or settings.face_mode == "OFF" or (settings.face_mode == "LIPS" and index < 14)
                row.prop(key, "value", text=label, slider=True)


CLASSES = (XRSettings, XRConnect, XRDisconnect, XRVideo, XRPlayback, XRCamera, XRViewport,
           XRRecord, XRFinishRecord, XRSaveMotion, XRPanel, XRFacePreset, XRFaceKeyframe, XRFacePanel)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.xr_ellen = PointerProperty(type=XRSettings)
    bpy.app.handlers.load_pre.append(before_load)
    bpy.app.handlers.save_pre.append(before_save)


def unregister():
    stop()
    if before_load in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.remove(before_load)
    if before_save in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.remove(before_save)
    del bpy.types.Scene.xr_ellen
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)

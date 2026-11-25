"""One scene profile for Ellen's SAM, controlled takes and final renders.

Hair/ribbons retain the project's configured Wiggle springs and colliders.
Skirt/tail use the dedicated fixed-length solver. Both are sampled by the same
runtime; a render uses a temporary baked action and restores its source.
"""
import importlib.util
import importlib
import hashlib
import json
import math
import sys
from array import array
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import bpy
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, FloatProperty, PointerProperty
from mathutils import Matrix, Vector

_owners, _busy = set(), set()
_preview, _renders = {}, {}
_history, _pending = {}, {}
VERSION = 8
MAX_PREVIEW_FRAMES = 256
SYNC_SEEK_FRAMES = 8


def rig_for(scene):
    for name in ('sam_ellen', 'character_director', 'xr_ellen'):
        rig = getattr(getattr(scene, name, None), 'rig', None)
        if rig and rig.type == 'ARMATURE' and (rig.get('ellen_physics_layout') or
                (rig.pose.bones.get('Q_0_0') and rig.pose.bones.get('W_0_1'))):
            return rig


def manages(scene):
    return hasattr(scene, 'ellen_physics') and rig_for(scene) is not None


def director():
    module = next((m for m in tuple(sys.modules.values()) if m and getattr(m, '__name__', '').endswith('character_director')
                   and hasattr(m, 'SecondaryBake')), None)
    # Import helpers without enabling the separate Character Director addon.
    return module or importlib.import_module('character_director')


def bone_binding(rig):
    """Physics needs leg transforms, not Director's enabled-IK validation."""
    by_jp = {getattr(getattr(b, 'mmd_bone', None), 'name_j', ''): b for b in rig.pose.bones}
    result = {}
    for role, (jp, en) in director().SEMANTICS.items():
        b = by_jp.get(jp) or rig.pose.bones.get(en) or rig.pose.bones.get(jp)
        if b: result[role] = b
    return result


def active_control(scene):
    try:
        session = director().active_session()
        return session is not None and session.scene == scene
    except (StopIteration, AttributeError):
        return False


def changed(self, context):
    clear_preview(self.id_data)


class EllenPhysicsSettings(bpy.types.PropertyGroup):
    enabled: BoolProperty(name='启用人物物理', default=True, update=changed)
    preview: BoolProperty(name='时间轴预览物理', default=True, update=changed)
    during_control: BoolProperty(name='试走 / 录制时启用物理', default=False, update=changed)
    during_render: BoolProperty(name='渲染时启用物理（自动补算）', default=True, update=changed)
    advanced: BoolProperty(name='高级设置', default=False, options={'SKIP_SAVE'})
    hair_enabled: BoolProperty(name='头发（含刘海）', default=True, update=changed)
    ribbons_enabled: BoolProperty(name='衣带', default=True, update=changed)
    skirt_enabled: BoolProperty(name='裙摆防穿', default=True, update=changed)
    tail_enabled: BoolProperty(name='鲨鱼尾巴', default=True, update=changed)
    skirt_clearance: FloatProperty(name='裙摆防穿间距', default=.012, min=0, max=.05, subtype='DISTANCE', update=changed)
    skirt_softness: FloatProperty(name='裙摆柔软度', default=.7, min=.3, max=2, update=changed)
    rear_skirt_lift: FloatProperty(name='后裙摆最大上扬', default=.018, min=0, max=.06, subtype='DISTANCE', update=changed)
    tail_inertia: FloatProperty(name='尾巴惯性', default=1, min=.3, max=2, update=changed)
    tail_droop: FloatProperty(name='尾巴静止下垂', default=math.radians(7), min=0, max=math.radians(20), subtype='ANGLE', update=changed)


def migrate(scene):
    p = scene.ellen_physics
    if scene.get('EllenPhysicsProfileVersion', 0) >= VERSION:
        return p
    previous = getattr(scene, 'character_director', None)
    if previous:
        for name in ('skirt_clearance', 'skirt_softness', 'tail_inertia'):
            if name in previous:
                setattr(p, name, previous[name])
    # Existing hair/ribbon spring, collider and safety constraint settings stay
    # on the model's original bones; this profile controls their shared use.
    scene['EllenPhysicsProfileVersion'] = VERSION
    return p


def fingerprint(scene):
    p = migrate(scene)
    values = {n: getattr(p, n) for n in ('hair_enabled', 'ribbons_enabled', 'skirt_enabled',
        'tail_enabled', 'skirt_clearance', 'skirt_softness', 'rear_skirt_lift', 'tail_inertia', 'tail_droop')}
    values['solver_version'] = VERSION
    rig = rig_for(scene)
    if rig and rig.get('ellen_physics_layout'):
        values['layout'] = rig['ellen_physics_layout']
    return json.dumps(values, sort_keys=True)


def baked(scene, rig):
    action = rig.animation_data.action if rig.animation_data else None
    if not action:
        return False
    try:
        info = json.loads(action.get('ellen_physics_bake', 'null'))
        return info and info['profile'] == fingerprint(scene) and info['start'] <= scene.frame_current <= info['end']
    except (ValueError, KeyError, TypeError):
        return False


class Runtime:
    def __init__(self, session):
        self.session, self.scene, self.rig = session, session.scene, session.rig
        self.p = migrate(self.scene)
        path = Path(__file__).with_name('character_director_secondary.py')
        spec = importlib.util.spec_from_file_location('ellen_secondary_runtime', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        class Adapter:
            p = self.p
            def __getattr__(self, name): return getattr(session, name)
        self.solver = module.SecondaryMotion(Adapter())
        layout = json.loads(self.rig.get('ellen_physics_layout', '{}'))
        self.ribbon_names = set(layout.get('ribbons', []))
        self.legacy = [b for b in self.rig.pose.bones if getattr(b, 'wiggle_enable', False)
            and b.name not in self.solver.names and not b.name.startswith(('Q_', 'W_')) and not b.wiggle_mute
            and (b.wiggle_tail or (b.wiggle_head and not b.bone.use_connect))
            and (self.p.ribbons_enabled if b.name.startswith('D_') or b.name in self.ribbon_names else self.p.hair_enabled)]
        self.legacy.sort(key=lambda b: len(b.parent_recursive))
        self.bones = self.solver.bones+self.legacy
        self.names = {b.name for b in self.bones}
        self.initialized = False
        self.input_pose = {}
        self.recoveries = 0

    def __getattr__(self, name): return getattr(self.solver, name)

    def reset_legacy(self, wiggle):
        for b in self.legacy:
            wiggle.reset_bone(b)
            # Wiggle's reset leaves sticky collider contacts alive. A fresh
            # bake must not inherit the previous render's contact points.
            b.wiggle.collision_ob = b.wiggle.collision_ob_head = None
            b.wiggle.collision_point = b.wiggle.collision_point_head = Vector()

    def step(self, dt):
        self.input_pose = {b.name: b.matrix_basis.copy() for b in self.bones}
        if self.rig.wiggle_mute or self.rig.wiggle_freeze:
            return
        if self.legacy:
            import wiggle_2 as wiggle
            if not dt and self.session.mode != 'NEW':
                self.reset_legacy(wiggle)
                self.solver.step(0)
                self.initialized = True
                return
            # Re-simulate from the animated body, never from old baked hair.
            for b in self.bones:
                b.matrix_basis = Matrix.Identity(4)
            self.scene.view_layers[0].update()
            if not self.initialized:
                self.reset_legacy(wiggle)
            if dt:
                state = self.scene.wiggle
                state.dt = dt
                dg = bpy.context.evaluated_depsgraph_get()
                for b in self.legacy:
                    b.wiggle.collision_normal = b.wiggle.collision_normal_head = Vector()
                    wiggle.move(b, dg)
                for i in range(state.iterations):
                    for b in self.legacy: wiggle.constrain(b, state.iterations-1-i, dg)
                for b in self.legacy:
                    wiggle.update_matrix(b, True)
                    b.wiggle.velocity = collision_velocity(b.wiggle.position-b.wiggle.position_last,
                        b.wiggle.velocity, b.wiggle.collision_normal, b.wiggle_bounce)
                    b.wiggle.velocity_head = collision_velocity(b.wiggle.position_head-b.wiggle.position_last_head,
                        b.wiggle.velocity_head, b.wiggle.collision_normal_head, b.wiggle_bounce_head)
                    b.wiggle.position_last = b.wiggle.position
                    b.wiggle.position_last_head = b.wiggle.position_head
                if any(not math.isfinite(v) for b in self.legacy for row in b.matrix_basis for v in row):
                    # The original chain solver can become singular at a
                    # coincident parent/tip during a sharp reversal. Discard
                    # that frame's cache before it can corrupt the whole rig.
                    for b in self.legacy: b.matrix_basis = Matrix.Identity(4)
                    self.scene.view_layers[0].update()
                    self.reset_legacy(wiggle)
                    self.recoveries += 1
            self.scene.wiggle.lastframe = self.scene.frame_current
        self.solver.step(dt)
        self.initialized = True

    def close(self, committed):
        self.solver.close(committed)


def collision_velocity(displacement, velocity, normal, bounce):
    # Match Wiggle's original final velocity update, including restitution.
    reflected = velocity.reflect(normal).project(normal)*bounce if normal.length_squared else Vector()
    return displacement+reflected


def preview_allowed(scene):
    if not manages(scene) or scene.as_pointer() in _busy or active_control(scene):
        return False
    rig = rig_for(scene)
    if rig.wiggle_mute or rig.wiggle_freeze: return False
    p = migrate(scene)
    sam = getattr(scene, 'sam_ellen', None)
    bridge = sys.modules.get('sam_ellen_bridge')
    work = getattr(bridge, '_work', None)
    processing = bool(work and work.get('scene') == scene)
    return p.enabled and p.preview and not processing and not getattr(sam, 'busy', False) and not baked(scene, rig_for(scene))


def clear_preview(scene):
    key = scene.as_pointer()
    _history.pop(key, None)
    _pending.pop(key, None)
    current = _preview.pop(key, None)
    if current:
        runtime = current[0]
        for name, basis in runtime.input_pose.items():
            runtime.rig.pose.bones[name].matrix_basis = basis
        runtime.close(False)


def new_runtime(scene):
    rig = rig_for(scene)
    s = SimpleNamespace(scene=scene, rig=rig, p=migrate(scene), bones=bone_binding(rig),
        mode='NEW', saved_pose={b.name: (b.matrix_basis.copy(), b.rotation_mode) for b in rig.pose.bones},
        current_rig_world=rig.matrix_world.copy(), scale=sum(rig.matrix_world.to_scale())/3)
    return Runtime(s)


def action_signature(scene, rig):
    """Check authored keys on seeks, including edits to the existing Action."""
    digest = hashlib.blake2b(digest_size=16)
    digest.update(fingerprint(scene).encode())
    digest.update(repr((scene.frame_start, scene.render.fps, scene.render.fps_base,
                        tuple(scene.gravity), scene.wiggle.iterations)).encode())
    objects = [rig]
    parent = rig.parent
    while parent:
        objects.append(parent)
        parent = parent.parent
    for ob in scene.objects:
        if ob.type == 'MESH' and 'Collider' in ob.name: objects.append(ob)
    seen = set()
    for ob in objects:
        ad = ob.animation_data
        actions = ([ad.action] if ad and ad.action else [])
        if ad:
            actions += [st.action for tr in ad.nla_tracks if not tr.mute for st in tr.strips if st.action]
            digest.update(repr([(tr.name, tr.mute, [(st.frame_start, st.frame_end, st.influence, st.scale, st.repeat)
                for st in tr.strips]) for tr in ad.nla_tracks]).encode())
        for action in actions:
            if action.as_pointer() in seen: continue
            seen.add(action.as_pointer())
            digest.update(str(action.as_pointer()).encode())
            for fc in director().curves(action):
                digest.update(repr((fc.data_path, fc.array_index, fc.mute, fc.extrapolation)).encode())
                for attr in ('co', 'handle_left', 'handle_right'):
                    values = array('f', [0.])*(2*len(fc.keyframe_points))
                    fc.keyframe_points.foreach_get(attr, values)
                    digest.update(values.tobytes())
                digest.update(bytes(k.bl_rna.properties['interpolation'].enum_items[k.interpolation].value
                                    for k in fc.keyframe_points))
    for b in rig.pose.bones:
        if b.wiggle_enable and not b.name.startswith(('Q_', 'W_')):
            digest.update(repr((b.name, b.wiggle_mute, b.wiggle_head, b.wiggle_tail,
                b.wiggle_stiff, b.wiggle_damp, b.wiggle_gravity, b.wiggle_bounce,
                b.wiggle_radius, b.wiggle_friction)).encode())
    return digest.digest()


def checkpoint(runtime):
    legacy = {}
    for b in runtime.legacy:
        values = {}
        for prop in b.wiggle.bl_rna.properties:
            n = prop.identifier
            if n in {'rna_type', 'name'} or prop.is_readonly: continue
            v = getattr(b.wiggle, n)
            if isinstance(v, Matrix): v = tuple(x for row in v.transposed() for x in row)
            elif getattr(prop, 'is_array', False): v = tuple(v)
            values[n] = v
        legacy[b.name] = values
    return dict(poses={b.name: (b.matrix_basis.copy(), b.rotation_mode) for b in runtime.bones},
        inputs={n:m.copy() for n,m in runtime.input_pose.items()}, legacy=legacy,
        states={n:([v.copy() for v in points], [v.copy() for v in velocities])
                for n,(points,velocities) in runtime.solver.states.items()})


def restore_checkpoint(runtime, saved):
    for n,(basis,mode) in saved['poses'].items():
        b = runtime.rig.pose.bones[n]
        b.rotation_mode, b.matrix_basis = mode, basis
    for n,values in saved['legacy'].items():
        for prop,v in values.items(): setattr(runtime.rig.pose.bones[n].wiggle, prop, v)
    runtime.input_pose = {n:m.copy() for n,m in saved['inputs'].items()}
    runtime.solver.states = {n:([v.copy() for v in points], [v.copy() for v in velocities])
                            for n,(points,velocities) in saved['states'].items()}
    runtime.initialized = True


def restore_input(runtime):
    for n,m in runtime.input_pose.items(): runtime.rig.pose.bones[n].matrix_basis = m


def remember(scene, runtime, stamp):
    frames = _history[scene.as_pointer()]['frames']
    frames[stamp] = checkpoint(runtime)
    frames.move_to_end(stamp)
    while len(frames) > MAX_PREVIEW_FRAMES: frames.popitem(last=False)
    _preview[scene.as_pointer()] = runtime, stamp


def advance_preview(scene, target, limit):
    """Replay authored intermediate frames with handlers guarded; never invent motion."""
    key = scene.as_pointer()
    current = _preview.get(key)
    runtime, at = current if current else (None, min(scene.frame_start, math.floor(target))-1)
    start = min(scene.frame_start, math.floor(target))
    frames = _history[key]['frames']
    if at > target or (at < target and any(at < f <= target for f in frames)):
        candidates = [f for f in frames if f <= target]
        if candidates:
            at = max(candidates)
            if runtime is None: runtime = new_runtime(scene)
            restore_checkpoint(runtime, frames[at])
        else:
            if runtime: runtime.close(False)
            runtime, at = None, start-1
    done = 0
    try:
        while at < target and done < limit:
            if runtime: restore_input(runtime)
            next_frame = min(target, math.floor(at)+1)
            scene.frame_set(math.floor(next_frame), subframe=next_frame-math.floor(next_frame))
            scene.view_layers[0].update()
            if runtime is None: runtime = new_runtime(scene)
            runtime.session.current_rig_world = runtime.rig.matrix_world.copy()
            runtime.step(0 if at < start else (next_frame-at)*scene.render.fps_base/scene.render.fps)
            scene.view_layers[0].update()
            at = next_frame
            remember(scene, runtime, at)
            done += 1
    finally:
        if runtime: restore_input(runtime)
        scene.frame_set(math.floor(target), subframe=target-math.floor(target))
        scene.view_layers[0].update()
        if runtime and at == target:
            restore_checkpoint(runtime, frames[target])
            scene.view_layers[0].update()
    return at == target


def preview_tick():
    for key, scene in tuple(_pending.items()):
        try:
            if key not in _history or not preview_allowed(scene):
                _pending.pop(key, None)
                continue
            target = scene.frame_current+scene.frame_subframe
            _busy.add(key)
            with bpy.context.temp_override(scene=scene, view_layer=scene.view_layers[0]):
                finished = advance_preview(scene, target, 2)
            if finished: _pending.pop(key, None)
            for screen in bpy.data.screens:
                for area in screen.areas:
                    if area.type == 'VIEW_3D': area.tag_redraw()
        except (ReferenceError, RuntimeError, ValueError) as e:
            _pending.pop(key, None)
            clear_preview(scene)
            print('ELLEN_PHYSICS_PREVIEW_ERROR', str(e), flush=True)
        finally:
            _busy.discard(key)
    return .01 if _pending else None


def preview_step(scene, dt=None):
    key = scene.as_pointer()
    current = _preview.get(key)
    stamp = scene.frame_current+scene.frame_subframe
    if dt is not None:
        # Live control supplies its own clock and must never replay the timeline.
        _history.pop(key, None)
        _pending.pop(key, None)
        runtime = current[0] if current else new_runtime(scene)
        runtime.session.current_rig_world = runtime.rig.matrix_world.copy()
        runtime.step(dt if current else 0)
        _preview[key] = runtime, stamp
        return True
    if key not in _history or not current or stamp != current[1]+1:
        signature = action_signature(scene, rig_for(scene))
        if key not in _history or _history[key]['signature'] != signature:
            clear_preview(scene)
            current = None
            _history[key] = dict(signature=signature, frames=OrderedDict())
    frames = _history[key]['frames']
    if stamp in frames:
        runtime = current[0] if current else new_runtime(scene)
        restore_checkpoint(runtime, frames[stamp])
        _preview[key] = runtime, stamp
        _pending.pop(key, None)
        return True
    if current and stamp == current[1]+1:
        runtime = current[0]
        runtime.session.current_rig_world = runtime.rig.matrix_world.copy()
        runtime.step(scene.render.fps_base/scene.render.fps)
        remember(scene, runtime, stamp)
        _pending.pop(key, None)
        return True
    gap = stamp-(current[1] if current and current[1] < stamp else min(scene.frame_start, math.floor(stamp))-1)
    limit = int(gap)+1 if bpy.app.background else SYNC_SEEK_FRAMES
    if advance_preview(scene, stamp, limit):
        _pending.pop(key, None)
        return True
    _pending[key] = scene
    if not bpy.app.timers.is_registered(preview_tick): bpy.app.timers.register(preview_tick, first_interval=.01)
    return False


@persistent
def frame_pre(scene, graph=None):
    if scene.as_pointer() in _busy: return
    runtime = _preview.get(scene.as_pointer())
    if runtime:
        for name, basis in runtime[0].input_pose.items():
            runtime[0].rig.pose.bones[name].matrix_basis = basis
        if not preview_allowed(scene): clear_preview(scene)


@persistent
def frame_post(scene, graph=None):
    if graph is not None and graph.mode == 'RENDER': return
    if not preview_allowed(scene): return
    key = scene.as_pointer()
    _busy.add(key)
    try:
        preview_step(scene)
        scene.view_layers[0].update()
    finally:
        _busy.discard(key)


@persistent
def render_init(scene, graph=None):
    if not manages(scene): return
    clear_preview(scene)
    p = migrate(scene)
    if not p.enabled or not p.during_render: return
    rig = rig_for(scene)
    if rig.wiggle_mute or rig.wiggle_freeze: return
    key = scene.as_pointer()
    if key in _renders: return
    if active_control(scene):
        raise RuntimeError('请先结束试走或录制，再渲染')
    cd, rig = director(), rig_for(scene)
    try:
        action = rig.animation_data.action if rig.animation_data else None
        info = json.loads(action.get('ellen_physics_bake', 'null')) if action else None
        if (info and info['profile'] == fingerprint(scene)
                and info['start'] <= min(scene.frame_start, scene.frame_current)
                and info['end'] >= max(scene.frame_end, scene.frame_current)):
            return
    except (ValueError, TypeError, KeyError):
        pass
    control = getattr(scene, 'character_director', None)
    saved = dict(rig=rig, animation=cd.animation_state(rig), frame=scene.frame_current,
        mutes={b.name: b.wiggle_mute for b in rig.pose.bones}, last=control.last_body if control else None,
        status=control.status if control else '', poses={b.name: (b.matrix_basis.copy(), b.rotation_mode) for b in rig.pose.bones},
        lock=scene.render.use_lock_interface, temp=None, input_temp=None, had_animation=bool(rig.animation_data))
    _renders[key] = saved
    _busy.add(key)
    scene.render.use_lock_interface = True
    baker = None
    try:
        if not action:
            # Static studios can still have animated parents or physics-only
            # motion. Seed a temporary empty source, then restore the absence
            # of an action along with every original pose after rendering.
            saved['input_temp'] = bpy.data.actions.new('人物物理 · 临时静态输入')
            rig.animation_data_create().action = saved['input_temp']
        baker = cd.SecondaryBake(scene, rig=rig)
        # Include the current frame for single-frame renders outside the range.
        baker.start = min(baker.start, saved['frame'])
        baker.end = max(baker.end, saved['frame'])
        baker.at = baker.start
        while not baker.step(): pass
        baker.close(True)
        saved['temp'] = rig.animation_data.action
        if control:
            control.last_body = saved['last']
            control.status = saved['status']
        scene.frame_set(saved['frame'])
        scene.view_layers[0].update()
        print('ELLEN_RENDER_PHYSICS_READY', baker.start, baker.end, len(baker.secondary.bones), flush=True)
    except Exception:
        if baker and not baker.closed: baker.close(False)
        render_finish(scene)
        raise


@persistent
def render_finish(scene, graph=None):
    key = scene.as_pointer()
    saved = _renders.pop(key, None)
    if not saved: return
    try:
        rig, cd = saved['rig'], director()
        cd.restore_animation(rig, saved['animation'])
        for name, (basis, mode) in saved['poses'].items():
            b = rig.pose.bones[name]
            b.rotation_mode, b.matrix_basis = mode, basis
            b.wiggle_mute = saved['mutes'][name]
        control = getattr(scene, 'character_director', None)
        if control:
            control.last_body = saved['last']
            control.status = saved['status']
        if saved['temp']:
            bpy.data.actions.remove(saved['temp'])
        if saved.get('input_temp'):
            bpy.data.actions.remove(saved['input_temp'])
        if not saved.get('had_animation', True):
            rig.animation_data_clear()
        scene.frame_set(saved['frame'])
        scene.render.use_lock_interface = saved['lock']
        _preview.pop(key, None)
    finally:
        _busy.discard(key)


@persistent
def load_post(_):
    _preview.clear()
    _history.clear()
    _pending.clear()
    _busy.clear()
    _renders.clear()
    for scene in bpy.data.scenes:
        if manages(scene): migrate(scene)


def draw(layout, scene):
    p = migrate(scene)
    box = layout.box()
    box.prop(p, 'enabled', text='人物物理')
    col = box.column()
    col.enabled = p.enabled
    row = col.grid_flow(row_major=True, columns=2, even_columns=True)
    layout = json.loads(rig_for(scene).get('ellen_physics_layout', '{}'))
    for name in ('hair_enabled', 'ribbons_enabled', 'skirt_enabled', 'tail_enabled'):
        if name == 'ribbons_enabled' and layout and not layout.get('ribbons'): continue
        row.prop(p, name)
    key = scene.as_pointer()
    if key in _pending:
        at = _preview.get(key, (None, scene.frame_start-1))[1]
        col.label(text=f'正在补算物理预览：{int(at)} / {scene.frame_current} 帧', icon='TIME')
    box.prop(p, 'advanced', emboss=False, icon='TRIA_DOWN' if p.advanced else 'TRIA_RIGHT')
    if not p.advanced: return
    col = box.column()
    col.enabled = p.enabled
    for name in ('preview', 'during_control', 'during_render'): col.prop(p, name)
    if p.skirt_enabled:
        for name in ('skirt_clearance', 'skirt_softness', 'rear_skirt_lift'): col.prop(p, name)
    if p.tail_enabled:
        for name in ('tail_inertia', 'tail_droop'): col.prop(p, name)
    col.operator('ellen_physics.refresh', text='刷新物理预览', icon='FILE_REFRESH')
    col.operator('ellen_physics.bake', text='更新整段物理关键帧', icon='PHYSICS')


class ELLENPHYSICS_OT_refresh(bpy.types.Operator):
    bl_idname = 'ellen_physics.refresh'
    bl_label = '刷新物理预览'
    bl_description = '修改动作、碰撞体或弹簧后，清除旧缓存并重新计算当前帧；保留原动作'

    def execute(self, context):
        scene = context.scene
        clear_preview(scene)
        scene.frame_set(scene.frame_current, subframe=scene.frame_subframe)
        return {'FINISHED'}


class ELLENPHYSICS_OT_bake(bpy.types.Operator):
    bl_idname = 'ellen_physics.bake'
    bl_label = '更新整段物理关键帧'
    bl_description = '为当前动作另建含物理的动作，原动作保留；播放和渲染可直接读取物理关键帧'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        clear_preview(context.scene)
        baker = None
        try:
            baker = director().SecondaryBake(context.scene, rig=rig_for(context.scene))
            while not baker.step(): pass
            baker.close(True)
        except Exception as e:
            if baker and not baker.closed: baker.close(False)
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        return {'FINISHED'}


def register(owner='director'):
    _owners.add(owner)
    if hasattr(bpy.types.Scene, 'ellen_physics'): return
    bpy.utils.register_class(EllenPhysicsSettings)
    bpy.utils.register_class(ELLENPHYSICS_OT_refresh)
    bpy.utils.register_class(ELLENPHYSICS_OT_bake)
    bpy.types.Scene.ellen_physics = PointerProperty(type=EllenPhysicsSettings)
    for handlers, callback in ((bpy.app.handlers.frame_change_pre, frame_pre),
        (bpy.app.handlers.frame_change_post, frame_post), (bpy.app.handlers.render_init, render_init),
        (bpy.app.handlers.render_complete, render_finish), (bpy.app.handlers.render_cancel, render_finish),
        (bpy.app.handlers.load_post, load_post)):
        handlers.append(callback)


def unregister(owner='director'):
    _owners.discard(owner)
    if _owners or not hasattr(bpy.types.Scene, 'ellen_physics'): return
    # Blender unregisters addons in an arbitrary order on exit. Wiggle's
    # dynamic RNA classes may already be gone; never run its property update
    # callbacks from here. Discard only our Python cache during teardown.
    for handlers, callback in ((bpy.app.handlers.frame_change_pre, frame_pre),
        (bpy.app.handlers.frame_change_post, frame_post), (bpy.app.handlers.render_init, render_init),
        (bpy.app.handlers.render_complete, render_finish), (bpy.app.handlers.render_cancel, render_finish),
        (bpy.app.handlers.load_post, load_post)):
        if callback in handlers: handlers.remove(callback)
    _preview.clear()
    _history.clear()
    _pending.clear()
    if bpy.app.timers.is_registered(preview_tick): bpy.app.timers.unregister(preview_tick)
    del bpy.types.Scene.ellen_physics
    bpy.utils.unregister_class(ELLENPHYSICS_OT_bake)
    bpy.utils.unregister_class(ELLENPHYSICS_OT_refresh)
    bpy.utils.unregister_class(EllenPhysicsSettings)

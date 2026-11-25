"""Owned native constraints: reversible FK state, missing spine binding and height."""
import bpy, json, uuid
from mathutils import Matrix,Quaternion,Euler
from . import compat
from .math_core import rest,direction
from .mapping import SWITCHES

PREFIX = 'MMRB_'


def create_collection(scene, cfg):
    cfg.session_id = uuid.uuid4().hex[:12]
    c = bpy.data.collections.new('MMR重定向辅助_'+cfg.session_id)
    scene.collection.children.link(c)
    c['mmrb_session'] = cfg.session_id
    cfg.helpers = c
    return c


def helper(cfg, role, parent=None):
    ob = bpy.data.objects.new(PREFIX+role+'_'+cfg.session_id[:6], None)
    cfg.helpers.objects.link(ob)
    ob['mmrb_role'] = role; ob['mmrb_session'] = cfg.session_id
    ob.empty_display_size = .015; ob.hide_render = True
    ob.rotation_mode = 'QUATERNION'
    if parent:
        ob.parent = parent
    return ob


def constraint(owner, kind, target, name, bone=''):
    c = owner.constraints.new(kind); c.name = PREFIX+name
    c.target = target; c.subtarget = bone
    c.owner_space = 'WORLD'; c.target_space = 'WORLD'
    return c


def snapshot(cfg, names):
    rig = cfg.target
    ad = rig.animation_data
    cfg.original_action = ad.action if ad else None
    state = {'slot':compat.slot_id(rig), 'basis':{}, 'properties':{}, 'muted':[],
             'nla':[(t.name,t.mute) for t in ad.nla_tracks] if ad else [],
             'use_nla':ad.use_nla if ad else True, 'switch_drivers':[],
             'blend_type':ad.action_blend_type if ad else 'REPLACE',
             'influence':ad.action_influence if ad else 1.,
             'extrapolation':ad.action_extrapolation if ad else 'HOLD'}
    for n in names:
        p = rig.pose.bones.get(n)
        if p:
            state['basis'][n] = {'matrix':[list(row) for row in p.matrix_basis], 'mode':p.rotation_mode}
    for n in SWITCHES:
        pb = rig.pose.bones.get(n)
        if pb:
            state['properties'][n] = {k:pb[k] for k in ['IK_FK','IK_Stretch'] if k in pb}
    cfg.helpers['snapshot'] = json.dumps(state,ensure_ascii=False)
    return state


def write_state(cfg, state):
    cfg.helpers['snapshot'] = json.dumps(state,ensure_ascii=False)


def read_state(cfg):
    return json.loads(cfg.helpers.get('snapshot','{}')) if cfg.helpers else {}


def source_dependency(cfg, enabled=True):
    if not cfg.helpers or not cfg.target:return
    name=cfg.helpers.get('source_dependency','')
    con=cfg.target.constraints.get(name) if name else None
    if enabled and cfg.source:
        if con is None:
            con=constraint(cfg.target,'COPY_TRANSFORMS',cfg.source,'SourceDependency_'+cfg.session_id)
            con.influence=0.
            con.subtarget=next((e.source for e in cfg.bone_map if e.target=='spine_fk'),'')
            cfg.helpers['source_dependency']=con.name
        con.mute=False
    elif con:con.mute=True


def mute_constraint(cfg, ob, pb, con, state):
    state['muted'].append(('RIG' if ob == cfg.target else 'MMD', pb.name, con.name, con.mute))
    con.mute = True


def force_fk(cfg, state):
    rig = cfg.target
    ad = rig.animation_data_create()
    for n in SWITCHES:
        pb = rig.pose.bones.get(n)
        if pb is None or 'IK_FK' not in pb:
            raise ValueError('找不到 MMR 的 IK/FK 属性：'+n)
        for prop, value in [('IK_FK',1.), ('IK_Stretch',0.)]:
            if prop not in pb:
                continue
            path = pb.path_from_id()+'["'+prop+'"]'
            fc = next((fc for fc in ad.drivers if fc.data_path == path),None)
            if fc is not None:
                raise ValueError(n+' 的 '+prop+' 已有自定义驱动器，请先移除该属性的驱动器。')
            pb[prop] = value
            fc = rig.driver_add(path)
            fc.driver.type = 'SCRIPTED'; fc.driver.expression = str(value)
            state['switch_drivers'].append(path)
            write_state(cfg,state)
    write_state(cfg,state)


def release_fk_drivers(cfg):
    state = read_state(cfg)
    for path in state.get('switch_drivers',[]):
        try: cfg.target.driver_remove(path)
        except (TypeError,RuntimeError): pass


def mute_old_layers(cfg, state, source_mode):
    if source_mode:
        for p in cfg.target.pose.bones:
            if p.name not in cfg.mapped_targets:
                continue
            for c in p.constraints:
                if c.name.startswith(('BAC_', 'RGBZH_Retarget', 'TorsoDelta')):
                    mute_constraint(cfg,cfg.target,p,c,state)
    if cfg.deform:
        for p in cfg.deform.pose.bones:
            # MMR already drives these MMD bones. Leaving the old MMD leg IK
            # active re-solves the chain and makes baked output depend on the
            # number/order of depsgraph evaluations (especially bent knees).
            bound=any(c.type=='COPY_TRANSFORMS' and c.target==cfg.target and not c.mute for c in p.constraints)
            for c in p.constraints:
                if c.name == 'NeckHeight_Adjust' or (source_mode and bound and (c.type=='IK' or c.name=='mmd_ik_limit_override')):
                    mute_constraint(cfg,cfg.deform,p,c,state)
    write_state(cfg,state)


def ensure_spine(cfg, deform, state):
    arm,rig = cfg.deform,cfg.target
    dn = deform.get('spine_fk.002')
    if not arm or not dn or 'DEF-spine.002' not in rig.pose.bones:
        return
    pb = arm.pose.bones[dn]
    # Already bound in the accepted project: keep that binding.
    if any(c.type == 'COPY_TRANSFORMS' and c.target == rig and not c.mute and c.influence > .99 for c in pb.constraints):
        return
    base = helper(cfg,'Spine2_base')
    constraint(base,'COPY_TRANSFORMS',rig,'Spine2_follow','DEF-spine.002')
    offset = helper(cfg,'Spine2_offset',base)
    offset.matrix_basis = rest(rig,'DEF-spine.002').inverted() @ rest(arm,dn)
    constraint(pb,'COPY_TRANSFORMS',offset,'Spine2_binding')


def build_height(scene, cfg, deform):
    rig,arm = cfg.target,cfg.deform
    names = [deform.get(n) for n in ['spine_fk.003','neck','head']]
    if not arm or not all(names):
        return False
    bindings = []
    for n in names:
        c = next((c for c in arm.pose.bones[n].constraints if c.type == 'COPY_TRANSFORMS' and c.target == rig and c.subtarget),None)
        if not c:
            return False
        bindings.append(c.subtarget)
    keyh = 'MMRB_颈头抬升厘米_'+cfg.session_id
    keyr = 'MMRB_颈根占比_'+cfg.session_id
    cfg.helpers['height_keys'] = json.dumps([keyh,keyr])
    scene[keyh],scene[keyr] = cfg.height_cm,cfg.height_root_ratio
    scene.id_properties_ui(keyh).update(min=0.,max=5.,description='沿胸颈骨链增加的高度，单位厘米。0 恢复原比例。')
    scene.id_properties_ui(keyr).update(min=0.,max=1.)
    units = max(scene.unit_settings.scale_length,1e-8)
    def drive_scale(obj, length, neck=False):
        fc = obj.driver_add('scale',1)
        for name,key in [('h',keyh),('r',keyr)]:
            v = fc.driver.variables.new(); v.name = name; v.type = 'SINGLE_PROP'
            v.targets[0].id_type = 'SCENE'; v.targets[0].id = scene; v.targets[0].data_path = '["'+key+'"]'
        fraction = '(1-r)' if neck else 'r'
        fc.driver.type = 'SCRIPTED'
        fc.driver.expression = f'1+h*0.01*{fraction}/{length*units:.12f}'
    chest = helper(cfg,'Height_chest_base')
    constraint(chest,'COPY_TRANSFORMS',rig,'Height_base',bindings[0])
    stretch = helper(cfg,'Height_chest_stretch',chest)
    length = arm.data.bones[names[0]].length
    drive_scale(stretch,length*rest(rig,bindings[0]).to_scale().y)
    neckroot = helper(cfg,'Height_neck_root',stretch); neckroot.location.y = length
    neck = helper(cfg,'Height_neck_base')
    constraint(neck,'COPY_LOCATION',neckroot,'Height_location')
    constraint(neck,'COPY_ROTATION',rig,'Height_rotation',bindings[1])
    constraint(neck,'COPY_SCALE',rig,'Height_scale',bindings[1])
    nstretch = helper(cfg,'Height_neck_stretch',neck)
    nlength = arm.data.bones[names[1]].length
    drive_scale(nstretch,nlength*rest(rig,bindings[1]).to_scale().y,True)
    headroot = helper(cfg,'Height_head_root',nstretch); headroot.location.y = nlength
    head = helper(cfg,'Height_head_base')
    constraint(head,'COPY_LOCATION',headroot,'Height_location')
    constraint(head,'COPY_ROTATION',rig,'Height_rotation',bindings[2])
    constraint(head,'COPY_SCALE',rig,'Height_scale',bindings[2])
    head_adjust = helper(cfg,'Height_head_adjust',head)
    adjust_keys={}
    for role,ob in [('neck',nstretch),('head',head_adjust)]:
        keys=['MMRB_'+role+'_q'+str(i)+'_'+cfg.session_id for i in range(4)]
        adjust_keys[role]=keys
        for i,key in enumerate(keys):
            scene[key]=1. if i==0 else 0.
            fc=ob.driver_add('rotation_quaternion',i);fc.driver.type='AVERAGE'
            v=fc.driver.variables.new();v.name='q';v.type='SINGLE_PROP'
            v.targets[0].id_type='SCENE';v.targets[0].id=scene;v.targets[0].data_path='["'+key+'"]'
    cfg.helpers['adjustment_keys']=json.dumps(adjust_keys)
    calibration={}
    for role,n in [('neck',names[1]),('head',names[2])]:
        sn=next((e.source for e in cfg.bone_map if e.target==role),'')
        calibration[role]=list(direction(arm,n).rotation_difference(direction(cfg.source,sn))) if cfg.source and sn in cfg.source.pose.bones else list(Quaternion())
    cfg.helpers['axis_calibration']=json.dumps(calibration)
    for n,ob in zip(names,[stretch,nstretch,head_adjust]):
        constraint(arm.pose.bones[n],'COPY_TRANSFORMS',ob,'Height_adjust')
    return True


def sync_height(scene,cfg):
    if not cfg.helpers:
        return
    keys = json.loads(cfg.helpers.get('height_keys','[]'))
    changed=False
    if keys and (scene.get(keys[0]) != cfg.height_cm or scene.get(keys[1]) != cfg.height_root_ratio):
        scene[keys[0]],scene[keys[1]] = cfg.height_cm,cfg.height_root_ratio
        changed=True
    adjustments=json.loads(cfg.helpers.get('adjustment_keys','{}'))
    calibration=json.loads(cfg.helpers.get('axis_calibration','{}'))
    for role,keys in adjustments.items():
        q=Quaternion()
        if cfg.mode=='EXISTING' and cfg.deform:
            from .mapping import deform_names
            n=deform_names(cfg.deform).get(role)
            if n:
                qr=rest(cfg.deform,n).to_quaternion();aim=Quaternion()
                if role in calibration:
                    aim=Quaternion().slerp(Quaternion(calibration[role]),cfg.head_axis if role=='head' else cfg.neck_axis)
                q=qr.inverted()@aim@Euler(cfg.head_angles if role=='head' else cfg.neck_angles,'XYZ').to_quaternion()@qr
        for key,val in zip(keys,q):
            if scene.get(key)!=val:scene[key]=val;changed=True
    if changed:scene.update_tag()


def restore_animation(cfg,state):
    rig = cfg.target
    if rig:
        for n,data in state.get('basis',{}).items():
            if n in rig.pose.bones:
                p = rig.pose.bones[n]; p.rotation_mode = data['mode']; p.matrix_basis = Matrix(data['matrix'])
        for n,values in state.get('properties',{}).items():
            if n in rig.pose.bones:
                for key,val in values.items():rig.pose.bones[n][key] = val
        compat.assign_action(rig,cfg.original_action,state.get('slot',''))
        if rig.animation_data:
            rig.animation_data.use_nla = state.get('use_nla',True)
            rig.animation_data.action_blend_type = state.get('blend_type','REPLACE')
            rig.animation_data.action_influence = state.get('influence',1.)
            rig.animation_data.action_extrapolation = state.get('extrapolation','HOLD')
            for name,mute in state.get('nla',[]):
                track = rig.animation_data.nla_tracks.get(name)
                if track:track.mute = mute


def restore(scene,cfg):
    state = read_state(cfg)
    rig = cfg.target
    release_fk_drivers(cfg)
    dependency=cfg.helpers.get('source_dependency','') if cfg.helpers else ''
    if rig and dependency in rig.constraints:rig.constraints.remove(rig.constraints[dependency])
    for ob in [rig,cfg.deform]:
        if not ob:
            continue
        for pb in ob.pose.bones:
            for c in list(pb.constraints):
                if c.name.startswith(PREFIX) and c.target and c.target.get('mmrb_session') == cfg.session_id:
                    pb.constraints.remove(c)
    for role,bone,name,mute in state.get('muted',[]):
        ob = rig if role == 'RIG' else cfg.deform
        if ob and bone in ob.pose.bones and name in ob.pose.bones[bone].constraints:
            ob.pose.bones[bone].constraints[name].mute = mute
    restore_animation(cfg,state)
    coll = cfg.helpers
    if coll:
        for key in json.loads(coll.get('height_keys','[]')):
            if key in scene:del scene[key]
        for keys in json.loads(coll.get('adjustment_keys','{}')).values():
            for key in keys:
                if key in scene:del scene[key]
        for ob in list(coll.objects):
            if ob.get('mmrb_session') == cfg.session_id:
                bpy.data.objects.remove(ob,do_unlink=True)
        if not coll.objects:
            bpy.data.collections.remove(coll)
    cfg.helpers = None; cfg.original_action = None; cfg.session_id = ''; cfg.status = 'NONE'
    scene.frame_set(scene.frame_current)

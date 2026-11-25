"""Per-control offsets layered after retargeting, without changing rig topology."""
import bpy
from mathutils import Euler,Vector,Quaternion
from . import layers

PREFIX='MMRB_Adjust_'


def nonzero(values):return any(abs(v)>1e-9 for v in values)


def constraints(cfg):
    if not cfg.target or not cfg.helpers:return []
    return [c for p in cfg.target.pose.bones for c in p.constraints
            if c.name.startswith(PREFIX) and c.target and c.target.get('mmrb_session')==cfg.session_id]


def active_names(cfg):
    names={e.target for e in cfg.bone_map if e.target in cfg.target.pose.bones and (nonzero(e.rotation_offset) or nonzero(e.location_offset))}
    if cfg.mode=='EXISTING' and nonzero(cfg.root_offset):names.add('root')
    return names


def sync_height_translation(cfg,known,name,loc,rot,space):
    """The height layer anchors the neck to the chest endpoint. Carry its
    explicit control translation through that anchor as a native delta.
    Head translation already reaches the MMD chain via MMR's neck stretch.
    """
    if name!='neck' or not cfg.height_available:return
    base=known.get('Height_'+name+'_base')
    if not base:return
    role='ADJ_HEIGHT_'+name;delta=known.get(role)
    if not nonzero(loc) and not delta:return
    axis=known.get(role+'_axes')
    if not axis:
        axis=layers.helper(cfg,role+'_axes');axis.hide_set(True)
        layers.constraint(axis,'COPY_ROTATION',cfg.target,'Adjust_axes',name)
        layers.constraint(axis,'COPY_SCALE',cfg.target,'Adjust_scale',name)
    if not delta:delta=layers.helper(cfg,role,axis);delta.hide_set(True)
    for c in axis.constraints:c.mute=space=='WORLD'
    delta.location=rot.inverted()@loc if space=='LOCAL' else loc
    con=next((c for c in base.constraints if c.target==delta),None)
    if not con:
        con=layers.constraint(base,'COPY_LOCATION',delta,'Adjust_height_location');con.use_offset=True
    con.mute=not nonzero(loc)


def sync(cfg):
    if not cfg.helpers or not cfg.target:return
    known={o.get('mmrb_role'):o for o in cfg.helpers.objects}
    specs=[(e.target,Vector(e.location_offset),Euler(e.rotation_offset,'XYZ').to_quaternion(),e.offset_space)
           for e in cfg.bone_map if e.target in cfg.target.pose.bones]
    if cfg.mode=='EXISTING':specs.append(('root',Vector(cfg.root_offset),Quaternion(),'WORLD'))
    for name,loc,rot,space in specs:
        sync_height_translation(cfg,known,name,loc,rot,space)
        role='ADJ_'+name;ob=known.get(role)
        enabled=nonzero(loc) or abs(rot.w)<1.-1e-12 or nonzero(tuple(rot)[1:])
        if not enabled and not ob:continue
        pb=cfg.target.pose.bones[name]
        if not ob:
            ob=layers.helper(cfg,role);ob.hide_set(True)
        con=next((c for c in pb.constraints if c.name.startswith(PREFIX) and c.target==ob),None)
        if not con:con=layers.constraint(pb,'COPY_TRANSFORMS',ob,'Adjust_'+name)
        con.mix_mode='AFTER_FULL' if space=='LOCAL' else 'BEFORE_SPLIT'
        # A connected FK bone cannot store an independent head translation in
        # ordinary location keys. Keep only that translation as a native layer
        # after baking; its rotation is already in the new Action.
        if cfg.status=='BAKED':
            enabled=pb.bone.use_connect and nonzero(loc)
            if space=='LOCAL':loc=rot.inverted()@loc
            rot=Quaternion()
        ob.location=loc;ob.rotation_quaternion=rot;ob.scale=(1,1,1)
        con.mute=not enabled


def suspend(cfg):
    state=[(c,c.mute) for c in constraints(cfg)]
    for c,_ in state:c.mute=True
    return state


def restore_suspended(state):
    for c,mute in state:c.mute=mute

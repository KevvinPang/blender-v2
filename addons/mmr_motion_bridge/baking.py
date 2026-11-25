import bpy
from . import compat, layers, runtime, mapping, adjustments
from mathutils import Matrix
from .math_core import pose


def set_retarget_constraints(cfg, muted):
    for pb in cfg.target.pose.bones:
        for c in pb.constraints:
            if c.name.startswith('MMRB_RT_') and c.target and c.target.get('mmrb_session')==cfg.session_id:
                c.mute=muted


def bake(scene,cfg=None):
    if cfg is None:cfg=scene.mmrb
    if cfg.status not in {'LIVE','PAUSED'}:
        raise ValueError('请先建立实时预览。')
    if cfg.frame_end<cfg.frame_start:raise ValueError('烘焙帧范围无效。')
    rig=cfg.target
    previous_action=rig.animation_data.action if rig.animation_data else None
    previous_slot=compat.slot_id(rig)
    ad=rig.animation_data_create()
    old_anim=(ad.use_nla,ad.action_blend_type,ad.action_influence,ad.action_extrapolation,[(t.name,t.mute) for t in ad.nla_tracks])
    previous_status=cfg.status;current_frame=scene.frame_current
    names={e.target for e in cfg.bone_map if e.target in rig.pose.bones}
    names.update(n for n in ['root','torso','hips','chest'] if n in rig.pose.bones)
    if cfg.mode=='EXISTING':
        names.update(p.name for p in rig.pose.bones if p.custom_shape and not p.name.startswith(('ORG-','MCH-','DEF-')))
        names.update(n for n in mapping.SWITCHES if n in rig.pose.bones)
    # Copy existing animation only in adjustment mode. Source mode writes a new
    # action without importing unrelated old IK/FK curves or NLA channels.
    destination=previous_action.copy() if cfg.mode=='EXISTING' and previous_action else None
    cfg.status='LIVE'
    checkpoints=sorted(set([cfg.frame_start,(cfg.frame_start+cfg.frame_end)//2,cfg.frame_end]))
    expected={}
    created=None
    try:
        for f in checkpoints:
            scene.frame_set(f);runtime.update(scene,True,cfg)
            expected[f]={n:pose(rig,n).copy() for n in names}
        # Existing MMR controls are inputs to native IK/constraints. Bake their
        # evaluated local channels, not the solved output of an IK chain twice.
        created=compat.bake_controls(rig,names,cfg.frame_start,cfg.frame_end,cfg.bake_step,destination,custom_props=cfg.mode=='EXISTING',visual=cfg.mode=='SOURCE',visual_overrides=adjustments.active_names(cfg) if cfg.mode=='EXISTING' else ())
        created.name=cfg.action_name or 'MMR_重定向_已烘焙';created.use_fake_user=True
        cfg.status='BAKED'
        layers.source_dependency(cfg,False)
        adjustments.sync(cfg)
        set_retarget_constraints(cfg,True)
        layers.release_fk_drivers(cfg)
        ad.use_nla=False;ad.action_blend_type='REPLACE';ad.action_influence=1.
        for t in ad.nla_tracks:t.mute=True
        if cfg.mode=='SOURCE':
            for n in mapping.SWITCHES:
                for prop,value in [('IK_FK',1.),('IK_Stretch',0.)]:
                    if prop in rig.pose.bones[n]:
                        compat.add_constant_key(rig,n,prop,value,cfg.frame_start,cfg.frame_end)
        error=0.
        # At step > 1 interpolation between sampled frames is intentionally
        # approximate; compare only the baked endpoints in that mode.
        for f in checkpoints if cfg.bake_step==1 else [checkpoints[0],checkpoints[-1]]:
            scene.frame_set(f);bpy.context.view_layer.update()
            for n,m in expected[f].items():
                now=pose(rig,n)
                error=max(error,max(abs(now[r][c]-m[r][c]) for r in range(4) for c in range(4)))
        if error>2e-4:
            raise RuntimeError(f'烘焙校验不一致（{error:.6g}），已恢复实时预览。')
        cfg.last_baked_action=created
        cfg.last_message=f'烘焙完成：{created.name}；原动作保留。高度参数仍可调。'
        cfg.last_bake_error=error
        return created
    except Exception:
        compat.assign_action(rig,previous_action,previous_slot)
        ad.use_nla,ad.action_blend_type,ad.action_influence,ad.action_extrapolation=old_anim[:4]
        for name,mute in old_anim[4]:
            if name in ad.nla_tracks:ad.nla_tracks[name].mute=mute
        cfg.status=previous_status
        if cfg.mode=='SOURCE':layers.source_dependency(cfg)
        adjustments.sync(cfg)
        set_retarget_constraints(cfg,False)
        if cfg.mode=='SOURCE' and cfg.auto_fk:
            state=layers.read_state(cfg)
            # Remove any remaining owned overrides before reinstating them.
            layers.release_fk_drivers(cfg);state['switch_drivers']=[]
            layers.force_fk(cfg,state)
        if created and created!=previous_action:
            bpy.data.actions.remove(created)
        elif destination and destination!=previous_action:
            bpy.data.actions.remove(destination)
        raise
    finally:
        scene.frame_set(current_frame);bpy.context.view_layer.update()


def resume(scene,cfg=None):
    if cfg is None:cfg=scene.mmrb
    if cfg.status=='BAKED':
        state=layers.read_state(cfg)
        if cfg.mode=='SOURCE':layers.source_dependency(cfg)
        if cfg.mode=='SOURCE':
            # Start again from the uncorrected FK channels. Otherwise baked
            # per-control translations would be added for a second time.
            compat.assign_action(cfg.target,None)
            for n in state.get('basis',{}):
                if n in cfg.target.pose.bones:cfg.target.pose.bones[n].matrix_basis=Matrix.Identity(4)
            state['switch_drivers']=[]
            if cfg.auto_fk:layers.force_fk(cfg,state)
            set_retarget_constraints(cfg,False)
        else:
            layers.restore_animation(cfg,state)
        cfg.status='LIVE';runtime.update(scene,True,cfg)
    elif cfg.status=='PAUSED':
        cfg.status='LIVE';runtime.update(scene,True,cfg)
    elif cfg.status=='LIVE':
        cfg.status='PAUSED'

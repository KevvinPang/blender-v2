import bpy, math, json
import numpy as np
from contextlib import contextmanager
from mathutils import Vector, Matrix, Quaternion, Euler
from bpy.app.handlers import persistent
from . import layers, compat, adjustments, mapping as maps
from .math_core import rest, pose, length, direction, world, calibration, paired_arms, solve_two_bone

SESSIONS = {}
BUSY = False
FOCUS = None


def settings(scene):
    if hasattr(scene,'mmrb'):
        yield scene.mmrb
    group=getattr(scene,'mmrb_multi',None)
    if group is not None:
        yield from group.pairs


def in_range(scene,cfg):
    return not cfg.source_layout or cfg.frame_start <= scene.frame_current_final <= cfg.frame_end


@contextmanager
def focus(cfg):
    """Bake one actor without solving every other actor at each sample."""
    global FOCUS
    saved=FOCUS
    FOCUS=cfg.session_id
    try:yield
    finally:FOCUS=saved


def validate(scene,cfg,allowed_driver_paths=()):
    if not maps.is_mmr(cfg.target):
        raise ValueError('请选择已由 MMR 生成的控制骨架（含 spine_fk、torso 等控制骨）。')
    if cfg.target.library:
        raise ValueError('目标为链接骨架，请先建立可编辑的库覆盖。')
    if cfg.target.name not in scene.objects:
        raise ValueError('目标控制骨架不在当前场景。')
    if cfg.mode == 'SOURCE':
        if not cfg.source or cfg.source == cfg.target or cfg.source.type != 'ARMATURE':
            raise ValueError('请选择独立的源动作骨架。')
        if cfg.source.name not in scene.objects:
            raise ValueError('源动作骨架不在当前场景。')
        pairs = {e.target:e.source for e in cfg.bone_map}
        missing = [n for n in maps.REQUIRED if n not in cfg.target.pose.bones or pairs.get(n,'') not in cfg.source.pose.bones]
        if missing:
            raise ValueError('必需骨骼映射缺失：'+', '.join(missing[:8]))
        if cfg.frame_end < cfg.frame_start:
            raise ValueError('结束帧必须不小于起始帧。')
        if not cfg.auto_fk and any(float(cfg.target.pose.bones[n].get('IK_FK',0.)) < .999 for n in maps.SWITCHES):
            raise ValueError('目标仍在 IK 状态，请启用「自动切换 FK」。')
        ad = cfg.target.animation_data
        if cfg.auto_fk and ad:
            switch_paths = {cfg.target.pose.bones[n].path_from_id()+'["'+p+'"]' for n in maps.SWITCHES for p in ['IK_FK','IK_Stretch']}
            if any(fc.data_path in switch_paths and fc.data_path not in allowed_driver_paths for fc in ad.drivers):
                raise ValueError('IK/FK 属性存在用户自定义驱动器，请先停用该驱动器。')
    for obj in [cfg.target,cfg.source] if cfg.mode == 'SOURCE' else [cfg.target]:
        scale = world(obj).to_scale()
        if min(scale) <= 0 or max(scale)/min(scale) > 1.0001:
            raise ValueError(obj.name+' 存在非均匀或负缩放，请先应用缩放。')
    if cfg.deform and cfg.deform == cfg.target:
        raise ValueError('MMD 变形骨架与 MMR 控制骨架应分别选择。')


class Session:
    def __init__(self,scene,cfg):
        self.scene,self.cfg,self.src,self.rig = scene,cfg,cfg.source,cfg.target
        self.map = {e.target:e.source for e in cfg.bone_map if cfg.source and e.source in cfg.source.pose.bones and e.target in cfg.target.pose.bones}
        self.helpers = {o['mmrb_role'][3:]:o for o in cfg.helpers.objects if o.get('mmrb_role','').startswith('RT_')}
        self.deform = maps.deform_names(cfg.deform)
        self.cache = json.loads(cfg.helpers.get('contacts','{}'))
        self.sole = {}
        if cfg.mode == 'SOURCE':
            self.offsets = calibration(self.src,self.rig,self.map,cfg.deform,self.deform)
            self.scale = sum(length(self.rig,n+'.L') for n in ['thigh_fk','shin_fk'])/sum(length(self.src,self.map[n+'.L']) for n in ['thigh_fk','shin_fk'])
            self.source_rest = rest(self.src,self.map['spine_fk']).translation
            self.target_rest = rest(self.rig,'torso').translation
            self.first = Vector(json.loads(cfg.helpers['first_hip']))
            self.anchor = self.target_rest.copy()
            if cfg.source_layout:self.anchor.x,self.anchor.y=self.first.x,self.first.y
            ratio=1. if cfg.source_layout else self.scale
            self.foot_offsets = {s:rest(self.rig,'foot_fk.'+s).translation-self.target_rest-(rest(self.src,self.map['foot_fk.'+s]).translation-self.source_rest)*ratio for s in 'LR'}
            self.read_soles()

    def read_soles(self):
        cfg = self.cfg
        points = {s:[] for s in 'LR'}
        if cfg.deform:
            for ob in self.scene.objects:
                if ob.type != 'MESH' or not any(m.type == 'ARMATURE' and m.object == cfg.deform for m in ob.modifiers):
                    continue
                groups = {s:{g.index for g in ob.vertex_groups if g.name.endswith('.'+s) and g.name.startswith(('足首','足先','つま先'))} for s in 'LR'}
                for v in ob.data.vertices:
                    for s in 'LR':
                        if sum(g.weight for g in v.groups if g.group in groups[s]) > .5:
                            points[s].append(world(ob)@v.co)
        for s in 'LR':
            inv = rest(self.rig,'foot_fk.'+s).inverted()
            pts = points[s]
            if pts:
                z = min(p.z for p in pts)
                pts = [p for p in pts if p.z < z+.025/max(self.scene.unit_settings.scale_length,1e-8)]
                if len(pts)>500:pts=pts[::max(1,len(pts)//500)]
                self.sole[s] = np.asarray([inv@p for p in pts],float)*rest(self.rig,'foot_fk.'+s).to_scale().x
            else:
                toe = 'toe.'+s
                p = rest(self.rig,toe).translation if toe in self.rig.pose.bones else rest(self.rig,'foot_fk.'+s).translation
                ankle = rest(self.rig,'foot_fk.'+s).translation
                z = min(p.z,ankle.z)-.025
                self.sole[s] = np.asarray([inv@Vector((p.x,p.y,z)),inv@Vector((ankle.x,ankle.y,z))])*rest(self.rig,'foot_fk.'+s).to_scale().x

    def analyze_contacts(self):
        cfg,scene = self.cfg,self.scene
        if cfg.mode != 'SOURCE':return
        positions=[]; hips=[]; footq=[]
        old=scene.frame_current
        wm=bpy.context.window_manager;wm.progress_begin(cfg.frame_start,cfg.frame_end+1)
        try:
            for f in range(cfg.frame_start,cfg.frame_end+1):
                scene.frame_set(f)
                positions.append([list(pose(self.src,self.map['foot_fk.'+s]).translation) for s in 'LR'])
                hips.append(list(pose(self.src,self.map['spine_fk']).to_quaternion()@rest(self.src,self.map['spine_fk']).to_quaternion().inverted()))
                footq.append([list(pose(self.src,self.map['foot_fk.'+s]).to_quaternion()) for s in 'LR'])
                if f%50==0:wm.progress_update(f)
        finally:
            scene.frame_set(old);wm.progress_end()
        arr=np.asarray(positions);fps=scene.render.fps/scene.render.fps_base
        floors=np.percentile(arr[:,:,2],3,axis=0)
        speed=np.linalg.norm(np.gradient(arr[:,:,:2],axis=0),axis=-1)*fps if len(arr)>1 else np.zeros((1,2))
        meters=max(scene.unit_settings.scale_length,1e-8)
        mask=(arr[:,:,2]<floors[None,:]+cfg.contact_height_cm*.01/meters)&(speed<cfg.contact_speed_cm*.01/meters)
        segments=[]
        for side in range(2):
            start=None
            for i in range(len(arr)+1):
                contact=i<len(arr) and bool(mask[i,side])
                if contact and start is None:start=i
                if not contact and start is not None:
                    if i-start>=max(4,round(fps*.15)):segments.append([side,start,i])
                    start=None
        self.cache={'start':cfg.frame_start,'positions':positions,'hips':hips,'foot_quats':footq,'floors':floors.tolist(),'segments':segments}
        cfg.helpers['contacts']=json.dumps(self.cache,separators=(',',':'))
        cfg.contact_count=len(segments)

    def raw_goal(self,s,source_position,hipdelta,footquat,scale,heading):
        cfg=self.cfg;i='LR'.index(s)
        delta=Vector(source_position)-Vector((self.first.x,self.first.y,self.source_rest.z))
        goal=self.anchor+heading@(delta*scale+hipdelta@self.foot_offsets[s])+Vector(cfg.root_offset)
        if cfg.ground and self.cache:
            clearance=max(0,source_position[2]-self.cache['floors'][i])*scale
            rotated=self.sole[s]@np.asarray(footquat.to_matrix()).T
            goal.z=cfg.floor_height+cfg.root_offset[2]+clearance-float(rotated[:,2].min())
        return goal

    def update(self):
        cfg=self.cfg
        layers.sync_height(self.scene,cfg)
        adjustments.sync(cfg)
        if cfg.mode != 'SOURCE':
            bpy.context.view_layer.update();return
        held=adjustments.suspend(cfg)
        try:self.update_source()
        finally:
            adjustments.restore_suspended(held)
            if held:bpy.context.view_layer.update()

    def update_source(self):
        cfg=self.cfg
        heading=Quaternion((0,0,1),cfg.heading)
        scale=(1. if cfg.source_layout else self.scale)*cfg.motion_scale
        quats={n:heading@pose(self.src,sn).to_quaternion()@self.offsets[n] for n,sn in self.map.items()}
        for n,angles in [('head',cfg.head_angles),('neck',cfg.neck_angles),('spine_fk.003',cfg.chest_angles)]:
            qrest=rest(self.rig,n).to_quaternion()
            aim=Quaternion()
            if n in {'head','neck'}:
                target=direction(cfg.deform,self.deform[n]) if n in self.deform else direction(self.rig,n)
                correction=target.rotation_difference(direction(self.src,self.map[n]))
                aim=Quaternion().slerp(correction,cfg.head_axis if n=='head' else cfg.neck_axis)
            quats[n]=quats[n]@qrest.inverted()@aim@Euler(angles,'XYZ').to_quaternion()@qrest
        for n in quats:
            if n.startswith(('thumb.','f_')) and cfg.finger_strength<1:
                hand='hand_fk.'+n.rsplit('.',1)[-1]
                neutral=quats[hand]@rest(self.rig,hand).to_quaternion().inverted()@rest(self.rig,n).to_quaternion()
                quats[n]=neutral.slerp(quats[n],cfg.finger_strength)
        pos=pose(self.src,self.map['spine_fk']).translation
        delta=Vector((pos.x-self.first.x,pos.y-self.first.y,pos.z-self.source_rest.z))
        position=self.anchor+heading@(delta*scale)+Vector(cfg.root_offset)
        hipdelta=pose(self.src,self.map['spine_fk']).to_quaternion()@rest(self.src,self.map['spine_fk']).to_quaternion().inverted()
        forward=(heading@hipdelta)@Vector((1,0,0))
        rootq=Quaternion((0,0,1),math.atan2(forward.y,forward.x))@rest(self.rig,'root').to_quaternion()
        self.helpers['root'].rotation_quaternion=rootq
        self.helpers['root'].location=rest(self.rig,'root').translation+position-self.target_rest
        self.helpers['torso'].location=position
        for n,q in quats.items():self.helpers[n].rotation_quaternion=q
        bpy.context.view_layer.update()
        if cfg.leg_position:
            goals={};frame=self.scene.frame_current
            for s in 'LR':
                goal=self.raw_goal(s,pose(self.src,self.map['foot_fk.'+s]).translation,hipdelta,quats['foot_fk.'+s],scale,heading)
                if cfg.foot_lock and self.cache:
                    ix=frame-self.cache['start'];side='LR'.index(s)
                    seg=next((v for v in self.cache['segments'] if v[0]==side and v[1]<=ix<v[2]),None)
                    if seg:
                        _,a,b=seg;anchor=min(a+2,b-1)
                        aq=heading@Quaternion(self.cache['foot_quats'][anchor][side])@self.offsets['foot_fk.'+s]
                        ag=self.raw_goal(s,self.cache['positions'][anchor][side],Quaternion(self.cache['hips'][anchor]),aq,scale,heading)
                        sole=Vector(self.sole[s].mean(axis=0))
                        shift=(ag+aq@sole)-(goal+quats['foot_fk.'+s]@sole)
                        w=max(0,min(1,(ix-a)/4,(b-1-ix)/4));w=w*w*(3-2*w)
                        goal.x+=shift.x*w;goal.y+=shift.y*w
                goals[s]=goal
            drop=0.
            for s,g in goals.items():
                hip=pose(self.rig,'thigh_fk.'+s).translation
                reach=sum(length(self.rig,n+'.'+s) for n in ['thigh_fk','shin_fk'])*.9999
                xy=(Vector((hip.x,hip.y))-Vector((g.x,g.y))).length
                if xy<reach:drop=min(drop,g.z+math.sqrt(max(0,reach*reach-xy*xy))-hip.z)
            self.helpers['root'].location.z+=drop;self.helpers['torso'].location.z+=drop
            bpy.context.view_layer.update()
            for s,goal in goals.items():
                a,b='thigh_fk.'+s,'shin_fk.'+s
                pole=heading@(pose(self.src,self.map[b]).translation-pose(self.src,self.map[a]).translation)
                q1,q2,error=solve_two_bone(pose(self.rig,a).translation,goal,pole,length(self.rig,a),length(self.rig,b),quats[a],quats[b])
                self.helpers[a].rotation_quaternion=q1;self.helpers[b].rotation_quaternion=q2
        bpy.context.view_layer.update()
        if cfg.hand_pair_strength>0:
            for n,q in paired_arms(self.src,self.rig,self.map,quats,cfg.hand_pair_strength,heading).items():
                self.helpers[n].rotation_quaternion=q
        bpy.context.view_layer.update()


def get_session(scene,cfg=None):
    if cfg is None:cfg=scene.mmrb
    if not cfg.helpers or not cfg.session_id:return None
    value=SESSIONS.get(cfg.session_id)
    if value is None:
        if cfg.mode=='SOURCE':
            layers.source_dependency(cfg);bpy.context.view_layer.update()
        value=SESSIONS[cfg.session_id]=Session(scene,cfg)
    return value


def start(scene,cfg=None):
    global BUSY
    if cfg is None:
        cfg=scene.mmrb
        if any(c.source_layout and c.status!='NONE' for c in settings(scene)):
            raise ValueError('请先拆除多人重定向。')
    if cfg.status!='NONE':raise ValueError('请先拆除当前重定向，或使用恢复实时预览。')
    validate(scene,cfg)
    BUSY=True;oldframe=scene.frame_current
    try:
        layers.create_collection(scene,cfg)
        names=list(maps.DEFAULT)+['root','torso','hips','chest']
        state=layers.snapshot(cfg,names)
        layers.source_dependency(cfg);bpy.context.view_layer.update()
        if cfg.mode=='SOURCE':
            # Record the first animated hip before adding any handlers/constraints.
            scene.frame_set(cfg.frame_start)
            pairs={e.target:e.source for e in cfg.bone_map}
            cfg.helpers['first_hip']=json.dumps(list(pose(cfg.source,pairs['spine_fk']).translation))
            if cfg.auto_fk:layers.force_fk(cfg,state)
            compat.assign_action(cfg.target,None)
            for track in cfg.target.animation_data.nla_tracks:track.mute=True
            cfg.target.animation_data.use_nla=False
            for n in names:
                p=cfg.target.pose.bones.get(n)
                if p:p.matrix_basis=Matrix.Identity(4)
        layers.mute_old_layers(cfg,state,cfg.mode=='SOURCE')
        deform=maps.deform_names(cfg.deform)
        layers.ensure_spine(cfg,deform,state)
        if cfg.mode=='SOURCE':
            pairs={e.target:e.source for e in cfg.bone_map if e.source in cfg.source.pose.bones and e.target in cfg.target.pose.bones}
            for n in list(pairs)+['root','torso']:
                ob=layers.helper(cfg,'RT_'+n)
                ob.scale=rest(cfg.target,n).to_scale()
                pb=cfg.target.pose.bones[n];pb.rotation_mode='QUATERNION'
                kind='COPY_TRANSFORMS' if n=='root' else ('COPY_LOCATION' if n=='torso' else 'COPY_ROTATION')
                layers.constraint(pb,kind,ob,'RT_'+n)
        cfg.height_available=layers.build_height(scene,cfg,deform)
        session=SESSIONS[cfg.session_id]=Session(scene,cfg)
        if cfg.mode=='SOURCE' and cfg.leg_position and (cfg.ground or cfg.foot_lock):session.analyze_contacts()
        cfg.status='LIVE';scene.frame_set(oldframe)
        if in_range(scene,cfg):session.update()
        for ob in cfg.helpers.objects:ob.hide_set(True)
        cfg.last_message='实时预览已建立；尚未烘焙。'
    except Exception:
        if cfg.helpers:
            sid=cfg.session_id
            layers.restore(scene,cfg);SESSIONS.pop(sid,None)
        raise
    finally:BUSY=False


def update(scene,force=False,cfg=None):
    global BUSY
    if BUSY or not hasattr(scene,'mmrb'):return
    BUSY=True
    try:
        for current in settings(scene) if cfg is None else (cfg,):
            if current.status=='NONE' or (FOCUS is not None and current.session_id!=FOCUS):continue
            try:
                layers.sync_height(scene,current)
                if (current.status=='LIVE' or (force and current.status=='PAUSED')) and in_range(scene,current):
                    session=get_session(scene,current)
                    if session:session.update()
            except (ValueError,KeyError,ReferenceError,RuntimeError) as e:
                current.last_message='预览已暂停：'+str(e);current.status='PAUSED'
    finally:BUSY=False


def remove(scene,cfg=None):
    global BUSY
    if cfg is None:cfg=scene.mmrb
    sid=cfg.session_id;BUSY=True
    try:layers.restore(scene,cfg);SESSIONS.pop(sid,None)
    finally:BUSY=False


@persistent
def frame_changed(scene,depsgraph=None):
    update(scene)


@persistent
def load_begin(_):
    global BUSY
    BUSY=True


def rebuild_loaded():
    global BUSY
    BUSY=True
    scenes=[]
    try:
        if hasattr(bpy.data,'scenes'):
            scenes=[scene for scene in bpy.data.scenes if any(c.status=='LIVE' for c in settings(scene))]
            # matrix_world may still be an unevaluated identity in load_post.
            # Evaluate object transforms before caching reference calibration.
            for scene in scenes:
                for cfg in settings(scene):
                    if cfg.status=='LIVE' and cfg.mode=='SOURCE':layers.source_dependency(cfg)
                scene.frame_set(scene.frame_current)
            if scenes:bpy.context.view_layer.update()
        SESSIONS.clear()
    finally:BUSY=False
    for scene in scenes:update(scene)
    return None


@persistent
def load_finished(_):
    SESSIONS.clear()
    # File loading temporarily exposes the previous UI/depsgraph. Defer GUI
    # rebuilding until Blender has completed the window/scene switch.
    if bpy.app.background and hasattr(bpy.data,'scenes'):
        rebuild_loaded()
    elif not bpy.app.timers.is_registered(rebuild_loaded):
        bpy.app.timers.register(rebuild_loaded,first_interval=.1)


def register_handlers():
    for seq,func in [(bpy.app.handlers.frame_change_post,frame_changed),(bpy.app.handlers.load_pre,load_begin),(bpy.app.handlers.load_post,load_finished),(bpy.app.handlers.undo_post,load_finished),(bpy.app.handlers.redo_post,load_finished)]:
        for old in list(seq):
            if getattr(old,'__module__','')==__name__ and old.__name__==func.__name__:seq.remove(old)
        seq.append(func)


def unregister_handlers():
    for seq,func in [(bpy.app.handlers.frame_change_post,frame_changed),(bpy.app.handlers.load_pre,load_begin),(bpy.app.handlers.load_post,load_finished),(bpy.app.handlers.undo_post,load_finished),(bpy.app.handlers.redo_post,load_finished)]:
        if func in seq:seq.remove(func)
    if bpy.app.timers.is_registered(rebuild_loaded):bpy.app.timers.unregister(rebuild_loaded)
    SESSIONS.clear()

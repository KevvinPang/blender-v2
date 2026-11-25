"""One final owner for image evidence, torso clearance and continuous arms.

Choose complete fixed-length chains over time. A safe key pose is insufficient:
the edges between poses must also be safe. Fingers are never written here.
"""
import math
import numpy as np
from mathutils import Quaternion, Vector


class PathDisconnected(ValueError):
    def __init__(self, index):
        self.index = index
        super().__init__('手臂安全轨迹无法连续连接（样本 %d），已保留原动作；请检查接触预设或观测数据' % (index+1))


def _rotate(q, v):
    q=np.asarray(q);v=np.broadcast_to(np.asarray(v),q.shape[:-1]+(3,))
    t=2*np.cross(q[...,1:],v)
    return v+q[...,:1]*t+np.cross(q[...,1:],t)


def _quats(elbows, wrists, a):
    from sam_ellen_arm_chain import frame_rotation
    u=elbows-np.asarray(a);v=wrists-elbows
    return np.array([tuple(frame_rotation(x,np.cross(x,y))) for x,y in zip(u,v)])


def _angle(first,last):
    return 2*np.arccos(np.clip(np.abs(first@last.T),0,1))


def _circles(a, goals, upper, lower, phases):
    goals=np.asarray(goals);axis=goals-a;d=np.linalg.norm(axis,axis=1)
    good=(d>.025)&(d<upper+lower-1e-5)
    goals=goals[good];d=d[good];axis=axis[good]/d[:,None]
    seed=np.tile([0.,1.,0.],(len(axis),1));seed[abs(axis[:,1])>.9]=[1.,0.,0.]
    x=np.cross(axis,seed);x/=np.maximum(1e-8,np.linalg.norm(x,axis=1))[:,None]
    y=np.cross(axis,x);along=(d*d+upper*upper-lower*lower)/(2*d)
    radius=np.sqrt(np.maximum(0,upper*upper-along*along))
    bends=x[:,None,:]*np.cos(phases)[None,:,None]+y[:,None,:]*np.sin(phases)[None,:,None]
    es=a+axis[:,None,:]*along[:,None,None]+bends*radius[:,None,None]
    ws=np.broadcast_to(goals[:,None,:],es.shape)
    return es.reshape(-1,3),ws.reshape(-1,3)


def _pose_cost(es,ws,row,goal,camera,data,side,measured,mirror,upper,lower,index):
    from sam_ellen_screen import project
    from sam_ellen_screen_fit import projected
    reach=upper+lower;aa=row['a']
    cp=np.c_[ws,np.ones(len(ws))]@np.asarray(camera[0]).T
    ep=np.c_[es,np.ones(len(es))]@np.asarray(camera[0]).T
    wp=projected(cp[:,:3],camera);elp=projected(ep[:,:3],camera)
    target=np.array(project(Vector(goal),camera))
    cost=np.sum((ws-goal)**2,axis=1)/reach**2
    # Inferred fallback points are useful for joining a brief dropout, but
    # their detector confidence is not joint-localization confidence. Keep
    # them from owning the image fit or overpowering the temporal path.
    image_weight=float(np.clip(row.get('tracking_reliability',.7)/.65,.12,1.))
    if row.get('inferred_hand',False):image_weight=min(image_weight,.28)
    if row.get('semantic_weight',0):
        cost+=80*row['semantic_weight']*np.sum((ws-goal)**2,axis=1)/reach**2
    if row.get('palm_evidence',False):
        # Choose the forearm WITH its observed palm, before the anatomical
        # wrist bound. Otherwise the final clamp destroys a correct gesture.
        from sam_ellen_arm_policy import wrist_limit
        forward=np.array(row['palm']@Vector((0,1,0)))
        bends=np.arccos(np.clip((ws-es)@forward/lower,-1,1))
        flex=np.degrees(np.arccos(np.clip(np.sum((es-aa)*(ws-es),axis=1)/(upper*lower),-1,1)))
        limits=np.radians(np.clip(45.+.75*(145.-flex),45.,60.))
        cost+=24*np.maximum(0,bends-limits)**2
    if row.get('close_range'):
        cost*=8.
        forward=np.array(row['palm']@Vector((0,1,0)))
        bends=np.arccos(np.clip((ws-es)@forward/lower,-1,1))
        cost+=25*np.maximum(0,bends-math.radians(58))**2
    if row['visible']:
        forward=np.asarray(row['body']@Vector((0,-1,0)))
        behind=np.maximum(0,reach*.16-(ws-aa)@forward)
        cost+=120*(1-row.get('semantic_weight',0))*(behind/reach)**2
        cost+=180*image_weight*np.sum((wp-target)**2,axis=1)
        layout=data.get('_sam_gesture_layout')
        if row.get('observation') is not None:
            from sam_ellen_shot import render_points
            shot=layout['shot'] if layout else None
            # Evaluate the whole posed hand, not only its wrist. This is a
            # solver cost before path selection, never an after-the-fact clamp.
            offsets=row['offsets']@np.asarray(row['palm'].to_matrix()).T
            forward=np.asarray(row['palm']@Vector((0,1,0)))
            forearm=(ws-es)/lower
            angle=np.arccos(np.clip(forearm@forward,-1,1))
            flex=np.degrees(np.arccos(np.clip(np.sum((es-aa)*(ws-es),axis=1)/(upper*lower),-1,1)))
            limit=np.radians(np.clip(45.+.75*(145.-flex),45.,60.))
            turn=np.maximum(0,angle-limit)
            axis=np.cross(np.broadcast_to(forward,forearm.shape),forearm)
            axis/=np.maximum(1e-9,np.linalg.norm(axis,axis=1))[:,None]
            rotated=offsets[None,:,:]*np.cos(turn)[:,None,None]+np.cross(axis[:,None,:],offsets[None,:,:])*np.sin(turn)[:,None,None]+axis[:,None,:]*(axis@offsets.T)[:,:,None]*(1-np.cos(turn))[:,None,None]
            points=ws[:,None,:]+rotated
            if row.get('spread_palm_weight',0):
                from sam_ellen_palm_orientation import palm_basis
                normal=palm_basis(offsets)[:,2]*(-1 if side=='L' else 1)
                bounded_normal=normal*np.cos(turn)[:,None]+np.cross(axis,normal)*np.sin(turn)[:,None]+axis*(axis@normal)[:,None]*(1-np.cos(turn))[:,None]
                up=np.asarray(row['body']@Vector((0,0,1)))
                # Evaluate the final bounded palm together with the arm. A
                # later wrist clamp must not silently undo a palm-up gesture.
                cost+=250*row['spread_palm_weight']*(1-np.clip(bounded_normal@up,-1,1))**2
            if len(row['palm_env']):
                from sam_ellen_body_clearance import _normals
                clearance=(points@_normals.T)[:,:,None,:]-row['palm_env'][None,None,:,:]
                clearance=clearance.max(3).min(2)
                cost+=10000*np.mean(np.maximum(0,.006-clearance)**2,axis=1)/reach**2
            cp_hand=np.c_[points.reshape(-1,3),np.ones(points.shape[0]*points.shape[1])]@np.asarray(camera[0]).T
            image=render_points(projected(cp_hand[:,:3],camera),shot).reshape(len(ws),21,2)
            mask=row['observation']['mask']
            desired=row['observation']['points'].copy();desired[:,0]*=camera[2]/data['screen']['aspect']
            if mirror:desired[:,0]=camera[2]-desired[:,0]
            desired=render_points(desired,shot)
            if row.get('semantic_anchor') is not None:
                from sam_ellen_semantic import contact_image_target
                anchor=render_points(project(Vector(row['semantic_anchor']),camera),shot)
                desired=contact_image_target(desired,row['semantic_descriptor'],anchor)
            cost+=600*image_weight*np.mean(np.sum((image[:,mask]-desired[None,mask])**2,axis=2),axis=1)
            if row.get('mouth_goal') is not None:
                from sam_ellen_semantic import hand_anchor
                attachment=hand_anchor(image,row['semantic_descriptor'])
                # Use the same phased target as the intact-hand fit. Pulling
                # directly to the held-contact point during entry/release
                # creates a second, discontinuous objective and a visible snap.
                delta=attachment-hand_anchor(desired,row['semantic_descriptor'])
                contact_weight=float(row.get('semantic_weight',0.))
                # Express contact error in face-width units, independent of
                # resolution/crop. It must agree with the image target above.
                width=max(.025,float(row.get('mouth_width',.1)))
                cost+=80*contact_weight*np.sum(delta*delta,axis=1)/(width*width)
            bounds=np.maximum(0,.025-image)+np.maximum(0,image-np.array([camera[2]-.025,.975]))
            cost+=100*image_weight*np.mean(np.sum(bounds[:,mask]**2,axis=2),axis=1)
        reference=max(.06,-(camera[0]@Vector(goal)).z)
        cost+=5*np.log(np.maximum(.025,-cp[:,2])/reference)**2
        pose=data['screen'].get('pose_arms',[])
        observation=pose[index].get(side) if len(pose)==len(data['frames']) else None
        if observation and observation.get('confidence',0)>=.5 and observation.get('elbow_observed'):
            from sam_ellen_gesture_layout import points as layout_points
            expected=layout_points(observation['points'],data,index,camera[2],mirror)
            direction=expected[1]-expected[2];direction/=max(1e-8,np.linalg.norm(direction))
            actual=elp-wp;actual/=np.maximum(1e-8,np.linalg.norm(actual,axis=1))[:,None]
            cost+=16*observation['confidence']*np.arccos(np.clip(actual@direction,-1,1))**2
            if measured:cost+=30*np.sum((elp-expected[1])**2,axis=1)
    else:cost+=18*np.sum((ws-goal)**2,axis=1)/reach**2
    from sam_ellen_upper_posture import posture_cost,reachable_posture
    posture=reachable_posture(row['upper_posture'],aa,goal,row['body'],upper,lower,
                              row.get('semantic_weight',0.))
    cost+=posture_cost(es,aa,row['body'],upper,camera,posture,mirror)
    cost+=.06*np.sum((es-row['e'])**2,axis=1)/reach**2
    return cost


def _path(sets, times, rows, upper, lower):
    """Dynamic programming on anatomically valid poses AND collision-free edges."""
    from sam_ellen_body_clearance import chains_safe
    from sam_ellen_motion_evidence import edge_limits
    costs=sets[0]['cost'].copy();parents=[]
    for i in range(1,len(sets)):
        old,new=sets[i-1],sets[i];dt=max(1e-6,times[i]-times[i-1])
        turn=_angle(old['q'],new['q'])
        flex=np.abs(old['flex'][:,None]-new['flex'][None,:])
        # Limit both links; an unchanged wrist must not conceal a hinge flip.
        lo=_rotate(old['q'],[0,1,0])*np.cos(old['flex'])[:,None]+_rotate(old['q'],[0,0,1])*np.sin(old['flex'])[:,None]
        ln=_rotate(new['q'],[0,1,0])*np.cos(new['flex'])[:,None]+_rotate(new['q'],[0,0,1])*np.sin(new['flex'])[:,None]
        lower_turn=np.arccos(np.clip(lo@ln.T,-1,1))
        rate,flex_rate=edge_limits(rows[i-1],rows[i])
        valid=(turn<=math.radians(rate*dt))&(lower_turn<=math.radians(rate*dt))&(flex<=math.radians(flex_rate*dt))
        unchanged=rows[i].get('locked',False) and rows[i-1].get('locked',False)
        if unchanged:valid[:]=True
        # Check the moving torso and arm at multiple interior points. This
        # precedes path selection; no independent projection may follow it.
        oi,ni=np.nonzero(valid)
        if len(oi) and not unchanged:
            q0=old['q'][oi];q1=new['q'][ni].copy();q1[np.sum(q0*q1,axis=1)<0]*=-1
            for t in (.25,.5,.75):
                q=q0*(1-t)+q1*t;q/=np.linalg.norm(q,axis=1)[:,None]
                f=old['flex'][oi]*(1-t)+new['flex'][ni]*t
                a=rows[i-1]['a']*(1-t)+rows[i]['a']*t
                e=a+_rotate(q,[0,1,0])*upper
                w=e+(_rotate(q,[0,1,0])*np.cos(f)[:,None]+_rotate(q,[0,0,1])*np.sin(f)[:,None])*lower
                env=rows[i-1]['env']*(1-t)+rows[i]['env']*t
                safe=chains_safe(a,e,w,env,upper,lower,margin=.0015)
                valid[oi[~safe],ni[~safe]]=False
        fast=max(rows[i-1].get('fast_motion',0.),rows[i].get('fast_motion',0.))
        transition=(.24*(turn/max(.08,dt*5))**2+.12*(flex/max(.08,dt*5))**2)/(1+2*fast)
        transition+=.4*np.sum((old['w'][:,None,:]-new['w'][None,:,:])**2,axis=2)/(upper+lower)**2
        score=costs[:,None]+transition
        score[~valid]=np.inf
        parent=np.argmin(score,axis=0);costs=new['cost']+score[parent,np.arange(len(parent))]
        if not np.isfinite(costs).any():
            print('ARM_PATH_DISCONNECTED',i+1,'minimum_turn',float(turn.min()),'edges',int(valid.sum()),
                  'prior_reachable',int(np.isfinite(score).any(1).sum()),'tail_turn',float(turn[-1,-1]),flush=True)
            raise PathDisconnected(i)
        parents.append(parent)
    index=int(np.argmin(costs));result=[index]
    for parent in reversed(parents):index=int(parent[index]);result.append(index)
    return list(reversed(result))


def connected_path(sets,times,rows,upper,lower,report=None):
    """Densify only a disconnected neighborhood; never waive safe edges."""
    from sam_ellen_body_clearance import chains_safe
    for radius in (0,6,12,24,36):
        try:
            return _path(sets,times,rows,upper,lower)
        except PathDisconnected as error:
            if radius == 36:
                raise
            center=error.index
            width=(6,12,24,36)[(0,6,12,24).index(radius)]
            lo=max(0,center-width);hi=min(len(rows)-1,center+width)
            left=np.argsort(sets[lo]['cost'])[:3];right=np.argsort(sets[hi]['cost'])[:3]
            added=0
            for i in range(lo+1,hi):
                if rows[i].get('locked',False):continue
                alpha=float((times[i]-times[lo])/(times[hi]-times[lo]))
                qs=[];fs=[]
                for a in left:
                    q0=rows[lo]['body'].inverted()@Quaternion(sets[lo]['q'][a])
                    for b in right:
                        q1=rows[hi]['body'].inverted()@Quaternion(sets[hi]['q'][b])
                        qs.append(tuple(rows[i]['body']@q0.slerp(q1,alpha)))
                        fs.append(sets[lo]['flex'][a]*(1-alpha)+sets[hi]['flex'][b]*alpha)
                q=np.array(qs);f=np.array(fs)
                u=_rotate(q,[0,1,0]);e=rows[i]['a']+u*upper
                w=e+(u*np.cos(f)[:,None]+_rotate(q,[0,0,1])*np.sin(f)[:,None])*lower
                safe=chains_safe(rows[i]['a'],e,w,rows[i]['env'],upper,lower,margin=.003)
                safe&=(f>=math.radians(15))&(f<=math.radians(180-rows[i].get('minimum_elbow',35.)))
                if not safe.any():continue
                # Preserve each frame's image objective: this is an optional
                # corridor, not a forced replacement or a time shift.
                base=float(np.min(sets[i]['cost']))
                objective=rows[i].get('pose_objective')
                cost=objective(e,w) if objective else base+2.+20*np.sum((w-rows[i]['w'])**2,axis=1)/(upper+lower)**2
                for key,value in dict(q=q,e=e,w=w,flex=f,cost=cost).items():
                    sets[i][key]=np.concatenate((sets[i][key],value[safe]))
                added+=int(safe.sum())
            if report is not None:
                report.append({'model':'local-safe-corridor-v1','sample':center+1,
                               'frames':[lo+1,hi+1],'added_candidates':added})


def _bridge_bank(sets, rows, visible, locked):
    """Include exact protected endpoints in the shared transition corridor.

    Protected poses have only one graph node. Sampling only editable observed
    frames omitted their hinge planes, leaving otherwise safe gestures without
    a legal entry/exit edge. Carry those endpoints into neighboring editable
    frames; the normal pose, speed and swept-collision checks still apply.
    """
    from sam_ellen_arm_chain import frame_rotation
    visible,locked=np.asarray(visible,bool),np.asarray(locked,bool)
    observed=np.flatnonzero(visible&~locked)
    selected=observed[np.unique(np.linspace(0,len(observed)-1,min(8,len(observed))).astype(int))] if len(observed) else []
    boundaries=np.flatnonzero(locked&(np.r_[True,~locked[:-1]]|np.r_[~locked[1:],True]))
    bank=[]
    for j in np.unique(np.r_[selected,boundaries]).astype(int):
        best=int(np.argmin(sets[j]['cost']))
        goalq=rows[j]['body'].inverted()@Quaternion(sets[j]['q'][best])
        goalf=float(sets[j]['flex'][best])
        ie,iw=rows[j]['idle_e'],rows[j]['idle_w'];aa=rows[j]['a']
        idleq=rows[j]['body'].inverted()@frame_rotation(Vector(ie-aa),Vector(np.cross(ie-aa,iw-ie)))
        idlef=float(Vector(ie-aa).angle(Vector(iw-ie)))
        for t in np.linspace(0,1,13):bank.append((idleq.slerp(goalq,float(t)),idlef*(1-t)+goalf*t))
    return bank


def solve_trajectory(scratch,rig,channels,data,cameras,options,protected=None,
                     allowed=None,sides=('L','R'),report=None):
    from sam_ellen_arm_visibility import observed_frames,neutral_pose,screen_hand,hand_landmark_mask
    from sam_ellen_arm_chain import frame_rotation,anatomical_rotations,bounded_wrist_rotation
    from sam_ellen_body_clearance import torso_geometry,torso_envelope,chains_safe,arm_clearance,observed_elbows
    from sam_ellen_arm_policy import PREFIXES,wrist_limit,supported_elbow_minimum
    from sam_ellen_screen import project,unproject
    from sam_ellen_screen_fit import hand_points
    from sam_ellen_motion_continuity import translation_goal
    from sam_ellen_retarget import set_global_rotation
    times=np.array([r['time'] for r in data['frames']]);n=len(times)
    if n<2 or not cameras:return
    original={name:list(values) for name,values in channels.items()}
    geometry=torso_geometry(rig);palm_geometry=torso_geometry(rig,include_bust=True);view=scratch.view_layers[0]
    from sam_ellen_palm_orientation import orientation_observations,fit_rotation
    from sam_ellen_palm_evidence import spread_palms_evidence
    spread_evidence=spread_palms_evidence(data)
    if report is not None:
        report.append({'model':'bilateral-low-open-palm-context-v1',
            'frames':(np.flatnonzero(spread_evidence)+1).tolist(),
            'evidence':'soft palm-up preference from sustained bilateral low open palms; inferred, not measured 3D',
            'exclusions':'raised hands / visible backs / unilateral gesture / body contact'})
    from sam_ellen_semantic import goal as semantic_goal,activation as contact_activation
    from sam_ellen_image_geometry import head_geometry
    semantic_geometry={'torso':geometry,'head':head_geometry(rig)} if options.get('semantic_contacts') else {}
    allowed=np.ones(n,bool) if allowed is None else np.asarray(allowed,bool)
    chest=rig.pose.bones.get('UpperBody2') or rig.pose.bones.get('UpperBody1')
    def load(i):
        for name,values in original.items():rig.pose.bones[name].rotation_quaternion=values[i][1]
        view.update()
    for side in sides:
        if not all(p+side in channels for p in PREFIXES):continue
        visible=observed_frames(data,side)
        locked=(np.asarray(protected[side],bool) if protected and side in protected else np.zeros(n,bool))|~allowed
        if locked.all():continue
        measured=observed_elbows(data,side)
        from sam_ellen_upper_posture import posture_evidence
        upper_evidence=posture_evidence(data,side)
        from sam_ellen_motion_evidence import motion_evidence
        evidence=motion_evidence(data,side)
        palm_observations=orientation_observations(data,side)
        tracked=data.get('screen',{}).get('_tracked_hands',{}).get(side,[])
        path_visible=visible.copy()
        if len(tracked)==n:
            path_visible|=np.array([bool(h and h.get('inferred')) for h in tracked])
        a,e,w=[rig.pose.bones[p+'_'+side] for p in ('Arm','Elbow','Wrist')]
        rows=[];goals=[]
        for i in range(n):
            load(i);aa,ee,ww=[b.head.copy() for b in (a,e,w)]
            body=chest.matrix.to_quaternion()@chest.bone.matrix_local.to_quaternion().inverted()
            palm=w.matrix.to_quaternion().copy();roll=rig.pose.bones['HandTwist_'+side].rotation_quaternion.copy()
            offsets=np.array([palm.inverted()@(p-ww) for p in hand_points(rig,side)])
            h=screen_hand(data['screen'],i,side,include_inferred=True);goal=ww.copy()
            inferred=bool(h and h.get('inferred'))
            descriptor=data.get('_sam_semantic',{}).get(side,[None]*n)[i]
            mouth_goal=None;mouth_width=None;semantic_anchor=None
            if descriptor and descriptor['region']=='mouth':
                from sam_ellen_retarget import mouth_landmark
                mouth_bone,mouth_local=mouth_landmark(rig)
                mouth_point=rig.pose.bones[mouth_bone].matrix@mouth_local
                mouth_goal=np.asarray(project(mouth_point,cameras[i]))
                mouth_width=float((project(rig.pose.bones['Eye_L'].head,cameras[i])-
                                   project(rig.pose.bones['Eye_R'].head,cameras[i])).length)
                if data.get('_sam_gesture_layout'):
                    from sam_ellen_shot import render_points
                    mouth_goal=render_points(mouth_goal,data['_sam_gesture_layout']['shot'])
                    mouth_width*=data['_sam_gesture_layout']['shot']['scale']
            palm_evidence=False;palm_candidates=[];obs=None
            if path_visible[i] and not locked[i] and h:
                from sam_ellen_gesture_layout import points as layout_points
                adapted=layout_points(h['points'],data,i,cameras[i][2],options.get('mirror',False))
                if options.get('mirror'):adapted[:,0]=cameras[i][2]-adapted[:,0]
                adapted[:,0]*=data['screen']['aspect']/cameras[i][2]
                obs={'points':adapted,'mask':hand_landmark_mask(h)}
                depth_hand=None if inferred else palm_observations[i]
                if options.get('unified_arms') and depth_hand is not None:
                    target=obs['points'].copy();target[:,0]*=cameras[i][2]/data['screen']['aspect']
                    if options.get('mirror'):target[:,0]=cameras[i][2]-target[:,0]
                    palm,details,palm_candidates=fit_rotation(offsets,ww,palm,depth_hand['points'],target,obs['mask'],
                        cameras[i],ww-ee,180.,options.get('mirror',False),ambiguous=True,return_candidates=True,
                        normal_hint=(np.asarray(cameras[i][0].to_3x3()@(body@Vector((0,0,1))))*(-1 if side=='L' else 1),spread_evidence[i]) if spread_evidence[i] else None)
                    palm_evidence=bool(details)
                goal=translation_goal(offsets,palm,ww,obs,cameras[i],data['screen']['aspect'],options.get('mirror',False))
                if descriptor:
                    posed=[ww+palm@Vector(offset) for offset in offsets]
                    region_goal,anchor=semantic_goal(rig,side,descriptor,posed,semantic_geometry)
                    semantic_anchor=np.array(anchor)
                    goal=goal.lerp(region_goal,contact_activation(descriptor))
            close_rows=data.get('_sam_close_targets',{}).get(side,[])
            close=close_rows[i] if len(close_rows)==n and not locked[i] and visible[i] else None
            if descriptor and descriptor['weight']>.2:close=None
            if close:
                blend=float(np.clip(close.get('weight',1.),0.,1.))
                goal=goal.lerp(cameras[i][0].inverted()@Vector(close['wrist']),blend)
                palm=palm.slerp(cameras[i][0].to_quaternion().inverted()@Quaternion(close['rotation']),blend)
            goals.append(np.array(goal))
            idle=neutral_pose(rig,side,view)
            for name,q in idle.items():
                # The shoulder and fingers belong to the source take.
                if name.startswith(PREFIXES):rig.pose.bones[name].rotation_quaternion=q
            view.update();ie,iw=e.head.copy(),w.head.copy()
            env=torso_envelope(rig,geometry);palm_env=torso_envelope(rig,palm_geometry)
            # Find a nearby natural pose outside this character's shoulder
            # socket, rather than exempting half its upper arm from collision.
            natural=[]
            for spread in (.24,.38,.55,.75):
                for forward in (.06,.20,.38,.60):
                    se=aa+body@Vector(((1 if side=='L' else -1)*spread,-forward,-1)).normalized()*(ee-aa).length
                    sw=se+body@Vector(((1 if side=='L' else -1)*spread*.65,-(forward+.60),-1)).normalized()*(ww-ee).length
                    natural.append((se,sw))
            ns=chains_safe(np.array(aa),np.array([x[0] for x in natural]),np.array([x[1] for x in natural]),env,(ee-aa).length,(ww-ee).length,margin=.003)
            feasible=[natural[k] for k in np.flatnonzero(ns)]
            if feasible:ie,iw=min(feasible,key=lambda p:(p[0]-ie).length_squared+(p[1]-iw).length_squared)
            rows.append({**evidence[i],'tracking_reliability':h.get('tracking_reliability',.7) if h else .05,'palm_candidates':palm_candidates,'offsets':offsets,'observation':obs,
                'semantic_descriptor':descriptor,'semantic_anchor':semantic_anchor,
                'mouth_goal':mouth_goal,'mouth_width':mouth_width,'close_target':close,
                'palm_evidence':palm_evidence,'inferred_hand':inferred,
                'spread_palm_weight':float(spread_evidence[i]),'semantic_weight':descriptor['weight'] if descriptor else 0.,
                'close_range':bool(close),'a':np.array(aa),'e':np.array(ee),'w':np.array(ww),'idle_e':np.array(ie),'idle_w':np.array(iw),
                'upper_posture':upper_evidence[i],'palm_env':palm_env,'body':body,'palm':palm,'roll':roll,'visible':bool(path_visible[i]),'locked':bool(locked[i]),'env':env,
                'minimum_elbow':supported_elbow_minimum(data,side,i),
                'natural':[(np.array(x),np.array(y)) for x,y in feasible]})
            if i%100==0:yield side,i,n*3
        upper=np.linalg.norm(rows[0]['e']-rows[0]['a']);lower=np.linalg.norm(rows[0]['w']-rows[0]['e']);reach=upper+lower
        if options.get('unified_arms'):
            from sam_ellen_palm_orientation import continuous_candidates
            chosen=continuous_candidates(times,[r['palm_candidates'] for r in rows],[r.get('fast_motion',0.) for r in rows])
            for i,palm in chosen.items():
                row=rows[i]
                if row['locked']:continue
                row['palm']=palm
                if row['observation'] is None:continue
                load(i);ww=Vector(row['w']);obs=row['observation'];offsets=row['offsets']
                goal=translation_goal(offsets,palm,ww,obs,cameras[i],data['screen']['aspect'],options.get('mirror',False))
                descriptor=row['semantic_descriptor']
                if descriptor:
                    posed=[ww+palm@Vector(offset) for offset in offsets]
                    region_goal,anchor=semantic_goal(rig,side,descriptor,posed,semantic_geometry)
                    row['semantic_anchor']=np.array(anchor)
                    goal=goal.lerp(region_goal,contact_activation(descriptor))
                close=row['close_target']
                if close:
                    blend=float(np.clip(close.get('weight',1.),0.,1.))
                    goal=goal.lerp(cameras[i][0].inverted()@Vector(close['wrist']),blend)
                    palm=palm.slerp(cameras[i][0].to_quaternion().inverted()@Quaternion(close['rotation']),blend)
                row['palm']=palm;goals[i]=np.array(goal)
        valid=np.flatnonzero(path_visible|locked)
        # An observed downward departure carries on towards that image edge.
        # Long losses then converge to a lowered arm without holding at the face.
        for i in range(n):
            if path_visible[i] or locked[i]:continue
            left=valid[valid<i];right=valid[valid>i]
            if len(left) and len(right) and times[right[0]]-times[left[-1]]<=.24:
                j,k=int(left[-1]),int(right[0]);t=(times[i]-times[j])/(times[k]-times[j]);goals[i]=goals[j]*(1-t)+goals[k]*t
            elif len(left):
                j=int(left[-1]);dt=times[i]-times[j];gain=float(np.clip((dt-.10)/.32,0,1));gain=gain*gain*(3-2*gain)
                # Move relative to the torso, not the old world-space socket.
                ref=goals[j]+rows[i]['a']-rows[j]['a']
                if j>0 and path_visible[j-1] and dt<.20:
                    velocity=(goals[j]-goals[j-1])/max(1e-6,times[j]-times[j-1])
                    velocity*=min(1.,1.8/max(1e-8,np.linalg.norm(velocity)))
                    ref=ref+velocity*min(dt,.12)*(1-gain)
                goals[i]=ref*(1-gain)+rows[i]['idle_w']*gain
            elif len(right) and times[right[0]]-times[i]<.30:
                j=int(right[0]);t=float(np.clip(1-(times[j]-times[i])/.30,0,1));t=t*t*(3-2*t)
                goals[i]=rows[i]['idle_w']*(1-t)+(goals[j]+rows[i]['a']-rows[j]['a'])*t
            else:goals[i]=rows[i]['idle_w'].copy()
        from sam_ellen_position_trajectory import fit_goals
        goals,position_report=fit_goals(times,goals,cameras,rows)
        if report is not None:report.append(dict(position_report,side=side))
        palm_valid=np.array([i for i,r in enumerate(rows) if r['palm_evidence'] or r['locked']],int)
        for i,row in enumerate(rows):
            if row['locked'] or row['palm_evidence']:continue
            left=palm_valid[palm_valid<i];right=palm_valid[palm_valid>i]
            if len(left) and len(right) and times[right[0]]-times[left[-1]]<=.35:
                j,k=int(left[-1]),int(right[0]);row['palm']=rows[j]['palm'].slerp(rows[k]['palm'],float((times[i]-times[j])/(times[k]-times[j])))
            elif len(left):
                # Location visibility is not an orientation observation. Do
                # not revive a noisy SAM palm on a partially detected hand.
                j=int(left[-1]);row['palm']=row['body']@rows[j]['body'].inverted()@rows[j]['palm']
        sets=[]
        for i,row in enumerate(rows):
            aa=row['a'];camera=cameras[i]
            row['pose_objective']=lambda es,ws,i=i,row=row,camera=camera:_pose_cost(es,ws,row,goals[i],camera,data,side,bool(measured[i]),options.get('mirror',False),upper,lower,i)
            if locked[i]:
                es=row['e'][None,:];ws=row['w'][None,:]
            else:
                goal=goals[i];xy=project(Vector(goal),camera);depth=max(.06,-(camera[0]@Vector(goal)).z)
                shoulder=-(camera[0]@Vector(aa)).z
                zs=np.unique(np.r_[depth,np.linspace(max(.06,shoulder-reach),shoulder+reach,17)])
                gs=np.array([unproject(xy,float(z),camera) for z in zs])
                # Natural poses remain graph candidates. They are expensive
                # when observed, but guarantee a conservative loss fallback.
                bridge=[row['idle_w']*(1-t)+goal*t for t in (.2,.4,.6,.8)]
                gs=np.vstack((gs,row['idle_w'],row['w'],bridge))
                es,ws=_circles(aa,gs,upper,lower,np.arange(64)*math.tau/64)
                elbow_bank=[]
                for spread in (.04,.12,.24,.38):
                    for forward in (-.04,.08,.22,.38):
                        direction=row['body']@Vector(((1 if side=='L' else -1)*spread,-forward,-1))
                        elbow_bank.append(aa+np.asarray(direction.normalized())*upper)
                ne=np.asarray(elbow_bank);nv=gs[None,:,:]-ne[:,None,:]
                nv/=np.maximum(1e-8,np.linalg.norm(nv,axis=2))[:,:,None]
                nw=ne[:,None,:]+nv*lower
                es=np.vstack((es,np.broadcast_to(ne[:,None,:],nw.shape).reshape(-1,3)))
                ws=np.vstack((ws,nw.reshape(-1,3)))
                es=np.vstack((es,row['e'],row['idle_e']));ws=np.vstack((ws,row['w'],row['idle_w']))
                if row['natural']:
                    es=np.vstack((es,[p[0] for p in row['natural']]))
                    ws=np.vstack((ws,[p[1] for p in row['natural']]))
            u=es-aa;v=ws-es
            flex=np.arccos(np.clip(np.sum(u*v,axis=1)/(upper*lower),-1,1))
            safe=chains_safe(aa,es,ws,row['env'],upper,lower,margin=.003)
            safe&=(flex>=math.radians(15)-1e-5)&(flex<=math.radians(180-row['minimum_elbow'])+1e-5)
            if locked[i]:safe[:]=True
            es,ws,flex=es[safe],ws[safe],flex[safe]
            if not len(es):raise ValueError('模型肩窝附近没有安全手臂解；已保留原动作（%s，第 %d 个样本）'%(side,i+1))
            q=_quats(es,ws,aa)
            cost=_pose_cost(es,ws,row,goals[i],camera,data,side,bool(measured[i]),options.get('mirror',False),upper,lower,i)
            # Retain diverse depths/phases, plus the safe idle, so a local
            # image minimum cannot remove the only continuous route.
            order=np.argsort(cost);keep=[]
            for k in order:
                if not keep or np.min(np.linalg.norm(es[keep]-es[k],axis=1)+np.linalg.norm(ws[keep]-ws[k],axis=1))>reach*.025:
                    keep.append(int(k))
                if len(keep)>=28:break
            idle_id=int(np.argmin(np.sum((es-row['idle_e'])**2+(ws-row['idle_w'])**2,axis=1)))
            if idle_id not in keep:keep.append(idle_id)
            if not locked[i]:
                for t in (.2,.4,.6,.8,1.):
                    target_w=row['idle_w']*(1-t)+goals[i]*t
                    distance=np.sum((ws-target_w)**2,axis=1)/reach**2
                    index=int(np.argmin(distance*100+cost*.025))
                    if index not in keep:keep.append(index)
                for ne,nw in row['natural']:
                    index=int(np.argmin(np.sum((es-ne)**2+(ws-nw)**2,axis=1)))
                    if index not in keep:keep.append(index)
            sets.append({'e':es[keep],'w':ws[keep],'q':q[keep],'flex':flex[keep],'cost':cost[keep]})
            if i%50==0:yield side,n+i,n*3
        # Shared torso-relative bridge poses make the graph connected across
        # abrupt detection starts. Pruning each frame independently otherwise
        # retains only idle and target clusters, with no legal route between.
        bank=_bridge_bank(sets,rows,visible,locked)
        for i,row in enumerate(rows):
            if locked[i] or not bank:continue
            q=np.array([tuple(row['body']@x[0]) for x in bank]);flex=np.array([x[1] for x in bank])
            es=row['a']+_rotate(q,[0,1,0])*upper
            ws=es+(_rotate(q,[0,1,0])*np.cos(flex)[:,None]+_rotate(q,[0,0,1])*np.sin(flex)[:,None])*lower
            safe=chains_safe(row['a'],es,ws,row['env'],upper,lower,margin=.003)
            safe&=(flex>=math.radians(15)-1e-5)&(flex<=math.radians(180-row['minimum_elbow'])+1e-5)
            es,ws,q,flex=es[safe],ws[safe],q[safe],flex[safe]
            if not len(es):continue
            costs=_pose_cost(es,ws,row,goals[i],cameras[i],data,side,bool(measured[i]),options.get('mirror',False),upper,lower,i)
            extra={'e':es,'w':ws,'q':q,'flex':flex,'cost':costs}
            for key,values in extra.items():sets[i][key]=np.concatenate((sets[i][key],values))
        from sam_ellen_arm_candidates import transport_neighbors
        transported=transport_neighbors(sets,rows,upper,lower)
        if report is not None:report.append({'model':'neighbor-pose-corridor-v1','side':side,'candidates':transported})
        path=connected_path(sets,times,rows,upper,lower,report)
        from sam_ellen_arm_curve import fit_curve,fit_palm_curve
        fitted_q,fitted_f,curve_report=fit_curve(times,rows,
            np.array([s['q'][k] for s,k in zip(sets,path)]),
            np.array([s['flex'][k] for s,k in zip(sets,path)]),upper,lower)
        if report is not None:report.append(dict(curve_report,side=side))
        directions=[Quaternion(q)@Vector((0,1,0))*math.cos(float(f))+
                    (Quaternion(q)@Vector((0,0,1)))*math.sin(float(f))
                    for q,f in zip(fitted_q,fitted_f)]
        palms,palm_report=fit_palm_curve(times,[r['palm'] for r in rows],directions,
            [wrist_limit(180-math.degrees(float(f))) for f in fitted_f],locked,
            [r.get('fast_motion',0.) for r in rows],
            [r['tracking_reliability'] if r['palm_evidence'] else .1 for r in rows])
        if report is not None:report.append(dict(palm_report,side=side))
        changes=[];hinge={}
        for i,k in enumerate(path):
            if locked[i]:hinge={};continue
            load(i);row=rows[i];aa=Vector(row['a']);q=Quaternion(fitted_q[i]);flex=float(fitted_f[i])
            u=q@Vector((0,1,0));v=u*math.cos(flex)+(q@Vector((0,0,1)))*math.sin(flex)
            ee=aa+u*upper;ww=ee+v*lower
            palm=palms[i]
            anatomical_rotations(a,e,w,aa,ee,ww,palm,set_global_rotation,view,hinge)
            held=w.matrix.to_quaternion().copy()
            rig.pose.bones['HandTwist_'+side].rotation_quaternion=row['roll'];view.update();set_global_rotation(w,held);view.update()
            for prefix in PREFIXES:
                name=prefix+side;channels[name][i]=channels[name][i][0],tuple(rig.pose.bones[name].rotation_quaternion)
            changes.append({'frame':i+1,'observed':bool(visible[i]),'elbow_measured':bool(measured[i]),
                'clearance_m':arm_clearance(a.head,e.head,w.head,row['env']),
                'wrist_shift_m':float(np.linalg.norm(np.array(w.head)-row['w']))})
            descriptor=row.get('semantic_descriptor') or {}
            if descriptor:
                changes[-1]['contact_fit']={
                    'region':descriptor['region'],
                    'activation':contact_activation(descriptor),
                    'wrist_goal_error_m':float((w.head-Vector(goals[i])).length),
                }
            if i%80==0:yield side,2*n+i,n*3
        if report is not None:report.append({'model':'coupled-image-clearance-trajectory-v1','side':side,'changes':changes,
            'frames':n,'protected_frames':(np.flatnonzero(locked)+1).tolist(),
            'constraints':'fixed lengths / observed elbow / whole upper arm / speed / quarter-frame collision edges'})

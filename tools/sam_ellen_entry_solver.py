"""Jointly join hidden preparation, partial visibility and measured hand motion.

Runs inside the final visibility owner. Only entrance windows are changed;
the two boundary poses and all later tracked/protected poses stay anchored.
"""
import math
import numpy as np
from mathutils import Vector,Quaternion


def fit_translation(local,rotation,origin,observation,camera,aspect,mirror=False):
    """Fit translation AND scale at the wrist rotation that will actually run."""
    from sam_ellen_screen_fit import projected
    target=np.asarray(observation['points'],float).copy();target[:,0]*=camera[2]/aspect
    if mirror:target[:,0]=camera[2]-target[:,0]
    mask=np.asarray(observation['mask'],bool);weights=mask.astype(float)
    weights[[0,5,9,13,17]]*=2.;weights/=max(1e-8,weights.sum())
    cp=np.asarray(camera[0]@origin);depth=max(.08,-cp[2]);initial=depth
    offsets=np.asarray(local)@np.asarray(camera[0].to_3x3()@rotation.to_matrix()).T
    value=np.array([cp[0],cp[1],math.log(depth)])
    def residual(v):
        p=np.r_[v[:2],-math.exp(float(v[2]))]+offsets
        r=((projected(p,camera)-target)*np.sqrt(weights[:,None])).ravel()
        return np.r_[r,.008*(v[2]-math.log(initial))]
    for _ in range(22):
        r=residual(value);jac=np.stack([(residual(value+np.eye(3)[j]*1e-4)-r)/1e-4 for j in range(3)],axis=1)
        step=np.linalg.solve(jac.T@jac+np.eye(3)*.0001,-jac.T@r)
        step[:2]*=min(1.,.06/max(1e-8,np.linalg.norm(step[:2])));step[2]=np.clip(step[2],-.15,.15)
        changed=False
        for gain in (1.,.5,.2):
            candidate=value+gain*step;candidate[2]=np.clip(candidate[2],math.log(.08),math.log(max(.12,initial*3)))
            if residual(candidate)@residual(candidate)<r@r:value=candidate;changed=True;break
        if not changed or np.linalg.norm(step)<1e-5:break
    return camera[0].inverted()@Vector((float(value[0]),float(value[1]),-math.exp(float(value[2]))))


def solve_entries(scratch,rig,channels,data,side,cameras,allowed,protected,floors,mirror=False,image_weights=None,report=None):
    from sam_ellen_entry import long_entries,incoming_edge,hand_at
    from sam_ellen_arm_visibility import arm_names,hand_landmark_mask,neutral_pose
    from sam_ellen_screen_fit import hand_points
    from sam_ellen_screen import project,unproject
    from sam_ellen_arm_policy import camera_reachable_goal,front_circle,transport_bend,wrist_limit
    from sam_ellen_arm_chain import anatomical_rotations,torso_front_plane,bounded_wrist_rotation
    from sam_ellen_retarget import set_global_rotation
    from sam_ellen_temporal import trajectory,quaternions
    times=np.array([r['time'] for r in data['frames']]);n=len(times);view=scratch.view_layers[0]
    names=arm_names(rig,side);original={name:list(rows) for name,rows in channels.items()}
    a,e,w=[rig.pose.bones[name+'_'+side] for name in ('Arm','Elbow','Wrist')]
    cache=data['screen'].get('hand_entry',{});episodes={ep['start']:ep for ep in cache.get('sides',{}).get(side,[])}
    policy=data.get('_sam_arm_policy',[]);diagnostics=data.setdefault('_sam_entry_report',[])
    def load(i):
        for name,rows in original.items():rig.pose.bones[name].rotation_quaternion=rows[i][1]
        view.update()
    for start in long_entries(data,side):
        if not start or not allowed[start] or protected[start]:continue
        ep=episodes.get(start,{});flow={int(k):v for k,v in ep.get('observations',{}).items()}
        edge,edge_source=(ep['edge'],ep['edge_source']) if ep else incoming_edge(data,side,start)
        if mirror and edge<2:edge=1-edge
        regions={int(k):v for k,v in ep.get('regions',{}).items()}
        first=min([start]+list(flow)+list(regions));end=min(n-1,start+8)
        # Never bridge across a manual interval, a reliable tracking end, or a
        # low-strength custom interval using a full-strength entrance repair.
        while end>start and (not allowed[start:end+1].all() or protected[start:end+1].any() or any(hand_at(data['screen'],i,side) is None for i in range(start,end+1))):end-=1
        if end-start<3:continue
        begin=max(0,int(np.searchsorted(times,times[first]-.42)))
        while begin<first and (not allowed[begin:first].all() or protected[begin:first].any()):begin+=1
        if first-begin<2:continue
        # Copy the first supported native finger pose into hidden preparation;
        # join back to each captured finger over only the first two frames.
        seed=min(end,start+2);ids=np.arange(begin,end+1);qhands=[];origins=[];bends=[];locals=[];lengths=[]
        for i in ids:
            load(int(i));origins.append(w.head.copy());qhands.append(w.matrix.to_quaternion().copy())
            axis=(w.head-a.head).normalized();bend=e.head-a.head;bend-=axis*bend.dot(axis);bends.append(bend.normalized())
            lengths.append(((e.head-a.head).length,(w.head-e.head).length))
        qseed=qhands[seed-begin]
        for k,i in enumerate(ids):
            if i<seed:
                alpha=float(np.clip((times[i]-times[begin])/(times[seed]-times[begin]),0,1));alpha=alpha*alpha*(3-2*alpha)
                qhands[k]=qhands[0].slerp(qseed,alpha)
        qhands=quaternions(times[ids],qhands,.04,weights=np.r_[30.,np.ones(len(ids)-2),30.])
        observations={};first_hand=hand_at(data['screen'],start,side)
        # Weak extrapolation is confined to the approach to the frame edge. It
        # does not pretend that hidden points have been independently detected.
        follow=[]
        for i in range(start,min(end+1,start+4)):
            h=hand_at(data['screen'],i,side)
            if h:follow.append((times[i],np.asarray(h['points'],float)))
        t=np.array([v[0] for v in follow]);p=np.array([v[1] for v in follow]);center=p[:,[5,9,13,17]].mean(1)
        delta=np.median(np.diff(center,axis=0)/np.diff(t)[:,None],axis=0) if len(t)>1 else np.zeros(2)
        # A kinematic extrapolation can prepare a hidden hand, but cannot move
        # the first-visible time without video support.
        for i in range(first,end+1):
            h=hand_at(data['screen'],i,side)
            if h:observations[i]={'points':np.asarray(h['points'],float),'mask':hand_landmark_mask(h),'weight':1.,'source':'measured'}
            elif i in flow:
                v=flow[i];observations[i]={'points':np.asarray(v['points'],float),'mask':np.asarray(v['in_frame'],bool),'weight':.45,'source':v['source']}
            else:
                points=np.asarray(first_hand['points'])+delta*(times[i]-times[start]);mask=(points[:,0]>=0)&(points[:,0]<=data['screen']['aspect'])&(points[:,1]>=0)&(points[:,1]<=1)
                observations[i]={'points':points,'mask':mask,'weight':.15,'source':'bounded-entry-continuation'}
        goals={};poses={};stage=[]
        for i in range(first,end+1):
            k=i-begin;load(i);rotation=qhands[k];native_q=w.matrix.to_quaternion();origin=w.head.copy()
            for name in names:
                if name in original and name.startswith(('Thumb','IndexFinger','MiddleFinger','RingFinger','LittleFinger')) and i<seed:
                    alpha=float(np.clip((times[i]-times[begin])/(times[first]-times[begin]),0,1));alpha=alpha*alpha*(3-2*alpha)
                    rig.pose.bones[name].rotation_quaternion=Quaternion(original[name][begin][1]).slerp(Quaternion(original[name][seed][1]),alpha)
            view.update();native_q=w.matrix.to_quaternion();origin=w.head.copy()
            local=np.array([native_q.inverted()@(p-origin) for p in hand_points(rig,side)])
            goal=fit_translation(local,rotation,origin,observations[i],cameras[i],data['screen']['aspect'],mirror)
            goal=camera_reachable_goal(a.head,goal,*lengths[k],floors[i],cameras[i],torso_front_plane(rig))
            elbow,goal,bend=front_circle(a.head,goal,*lengths[k],bends[k],torso_front_plane(rig),floors[i])
            # Fit again after anatomy clamps the wrist. Translation must use
            # the same rotation as the final palm, not an unreachable proposal.
            for _ in range(3):
                rotation=bounded_wrist_rotation(rotation,goal-elbow,wrist_limit(180-math.degrees((elbow-a.head).angle(goal-elbow))))
                target=fit_translation(local,rotation,goal,observations[i],cameras[i],data['screen']['aspect'],mirror)
                goal=camera_reachable_goal(a.head,target,*lengths[k],floors[i],cameras[i],torso_front_plane(rig))
                elbow,goal,bend=front_circle(a.head,goal,*lengths[k],bend,torso_front_plane(rig),floors[i])
            if i in regions and i not in flow:
                # A verified small patch constrains only its visible top and
                # side extent. Avoid enlarging it into a complete guessed hand.
                top=regions[i]['bounds'][1];image=np.array([project(goal+rotation@Vector(p),cameras[i]) for p in local]);xy=project(goal,cameras[i]);depth=max(.08,-(cameras[i][0]@goal).z)
                xy.y+=float(top-image[:,1].min());goal=unproject(xy,depth,cameras[i])
                goal=camera_reachable_goal(a.head,goal,*lengths[k],35.,cameras[i],torso_front_plane(rig))
            # Rejoin the existing stable trajectory over its last three frames.
            blend=float(np.clip((i-(end-3))/3,0,1));blend=blend*blend*(3-2*blend)
            goal=goal.lerp(origins[k],blend);rotation=rotation.slerp(qhands[k],blend)
            goals[i]=goal;poses[i]=rotation
            stage.append({'frame':i+1,'source':observations[i]['source'],'fitted_wrist':list(project(goal,cameras[i]))})
        # Join to the visible trajectory with its incoming screen velocity.
        # The first and last original poses are fixed anchors.
        anchor=goals[first];anchor_xy=project(anchor,cameras[first]);anchor_depth=max(.08,-(cameras[first][0]@anchor).z)
        load(begin);idle=w.head.copy();old_xy=project(idle,cameras[begin]);old_depth=max(.08,-(cameras[begin][0]@idle).z)
        for i in range(begin,first):
            t=float((times[i]-times[begin])/(times[first]-times[begin]));ease=t*t*(3-2*t)
            xy=old_xy.lerp(anchor_xy,ease);depth=math.exp((1-ease)*math.log(old_depth)+ease*math.log(anchor_depth))
            goals[i]=unproject(xy,depth,cameras[i]);poses[i]=qhands[i-begin]
        # Fit acceleration jointly in camera coordinates; measured entry points
        # outweigh weak hidden continuation. No filtering of final IK angles.
        cp=np.array([cameras[i][0]@goals[i] for i in ids]);z=np.maximum(.08,-cp[:,2]);values=np.c_[cp[:,:2]/z[:,None],np.log(z)]
        measured_weight=64. if first<start else 8.
        weights=np.array([measured_weight if i in observations and observations[i]['source']=='measured' else observations[i]['weight']*8 if i in observations else .3 for i in ids]);weights[[0,-1]]=10000.
        values=trajectory(times[ids],values,.05,weights)
        hinge={};last_axis=None;last_bend=None
        for k,i in enumerate(ids):
            if i==begin or i==end:continue
            load(int(i));idle_pose={name:Quaternion(original[name][begin][1]) for name in names if name in original}
            for name in names:
                if name not in original or not name.startswith(('Thumb','IndexFinger','MiddleFinger','RingFinger','LittleFinger')):continue
                if i<seed:
                    alpha=float(np.clip((times[i]-times[begin])/(times[first]-times[begin]),0,1));alpha=alpha*alpha*(3-2*alpha)
                    rig.pose.bones[name].rotation_quaternion=idle_pose[name].slerp(Quaternion(original[name][seed][1]),alpha)
            view.update();upper,lower=lengths[k];v=values[k];depth=math.exp(v[2]);goal=cameras[i][0].inverted()@Vector((float(v[0]*depth),float(v[1]*depth),-depth))
            if i<first:
                # Enforce pre-entry visibility on the actual native hand,
                # whose extent differs from the source hand's crop.
                q=poses[i];native=w.matrix.to_quaternion();origin=w.head.copy()
                offsets=[q@(native.inverted()@(p-origin)) for p in hand_points(rig,side)]
                projected=[project(goal+o,cameras[i]) for o in offsets];xy=project(goal,cameras[i])
                extent=[max(p.x-xy.x for p in projected),max(xy.x-p.x for p in projected),max(p.y-xy.y for p in projected),max(xy.y-p.y for p in projected)]
                if edge==0:xy.x=min(xy.x,-extent[0]-.004)
                elif edge==1:xy.x=max(xy.x,cameras[i][2]+extent[1]+.004)
                elif edge==2:xy.y=min(xy.y,-extent[2]-.004)
                else:xy.y=max(xy.y,1+extent[3]+.004)
                goal=unproject(xy,depth,cameras[i])
            floor=float(floors[i]) if i>=first else 35.
            goal=camera_reachable_goal(a.head,goal,upper,lower,floor,cameras[i],torso_front_plane(rig))
            axis=(goal-a.head).normalized();bend=transport_bend(bends[k],None,axis)
            if last_bend is not None:
                prior=transport_bend(last_bend,last_axis,axis);turn=math.atan2(axis.dot(prior.cross(bend)),prior.dot(bend));limit=math.radians(360*(times[i]-times[i-1]))
                bend=Quaternion(axis,float(np.clip(turn,-limit,limit)))@prior
            elbow,goal,bend=front_circle(a.head,goal,upper,lower,bend,torso_front_plane(rig),floor,last_bend)
            rotation=bounded_wrist_rotation(poses[i],goal-elbow,wrist_limit(180-math.degrees((elbow-a.head).angle(goal-elbow))))
            anatomical_rotations(a,e,w,a.head.copy(),elbow,goal,rotation,set_global_rotation,view,hinge)
            # The anatomy-limited palm can have a different extent than its
            # proposal. Close the crop against the actually executed hand.
            if i<first:
                for _ in range(6):
                    hp=np.array([project(p,cameras[i]) for p in hand_points(rig,side)]);xy=project(w.head,cameras[i]);depth=max(.08,-(cameras[i][0]@w.head).z)
                    correction=[-float(hp[:,0].max())-.006,float(cameras[i][2]-hp[:,0].min())+.006,-float(hp[:,1].max())-.006,float(1-hp[:,1].min())+.006][edge]
                    if (edge in (0,2) and correction>=0) or (edge in (1,3) and correction<=0):break
                    if edge<2:xy.x+=correction
                    else:xy.y+=correction
                    target=unproject(xy,depth,cameras[i]);target=camera_reachable_goal(a.head,target,upper,lower,35.,cameras[i],torso_front_plane(rig))
                    elbow,target,bend=front_circle(a.head,target,upper,lower,bend,torso_front_plane(rig),35.,last_bend)
                    rotation=bounded_wrist_rotation(rotation,target-elbow,60.)
                    anatomical_rotations(a,e,w,a.head.copy(),elbow,target,rotation,set_global_rotation,view,hinge)
            last_axis=(w.head-a.head).normalized();last_bend=e.head-a.head;last_bend-=last_axis*last_bend.dot(last_axis);last_bend.normalize()
            # Match velocity as well as pose at the hidden preparation boundary.
            # Original and solved hands are still outside the crop here.
            ramp=float(np.clip((times[i]-times[begin])/.16,0,1));ramp=ramp*ramp*(3-2*ramp)
            amount=(float(image_weights[i]) if image_weights is not None else 1.)*ramp
            for name in names:
                if name in channels:
                    q=Quaternion(original[name][i][1]).slerp(rig.pose.bones[name].rotation_quaternion,amount)
                    channels[name][i]=channels[name][i][0],tuple(q)
            if len(policy)==n and side in policy[i]:
                policy[i][side].update(entry_continuous=True,entry_edge=edge,entry_source=observations.get(i,{}).get('source','hidden-preparation'))
            if report is not None and i<len(report):report[i]['entry_continuous']=True
        diagnostics.append({'side':side,'start':start+1,'first_visible':first+1,'begin':begin+1,'end':end+1,
            'edge':edge,'edge_source':edge_source,'video_flow_frames':[i+1 for i in flow],'stages':stage})

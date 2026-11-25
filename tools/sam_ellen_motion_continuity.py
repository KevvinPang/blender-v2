"""Final evidence-constrained motion trajectories, after all image pose searches."""
import math
import numpy as np
from mathutils import Vector,Quaternion


def groups(mask):
    edges=np.diff(np.r_[False,np.asarray(mask,bool),False].astype(int))
    return zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))


def fill_rotation_losses(times,values,observed,max_gap=.85):
    """Bridge missing estimates in the same pose space, never switch sources."""
    result=[Quaternion(q).normalized() for q in values];valid=np.flatnonzero(observed)
    if not len(valid):return result
    for lo,hi in groups(~np.asarray(observed,bool)):
        left=lo-1 if lo else None;right=hi if hi<len(times) else None
        if left is not None and right is not None and times[right]-times[left]<=max_gap:
            for i in range(lo,hi):result[i]=result[left].slerp(result[right],float((times[i]-times[left])/(times[right]-times[left])))
        else:
            # Long losses have no justified target; retain the last trustworthy
            # pose rather than releasing abruptly to an unrelated SAM estimate.
            for i in range(lo,hi):
                if left is not None:result[i]=result[left].copy()
                elif right is not None:result[i]=result[right].copy()
    return result


def repair_body_losses(scratch,rig,channels,data,report):
    """Repair old baked fallback poses as well as future public bake output."""
    from sam_ellen_temporal import quaternions
    times=np.array([r['time'] for r in data['frames']]);visible=np.array([bool(r.get('face')) for r in data['screen']['frames']])
    if visible.all() or not visible.any():return
    # Local spine / neck channels share the same missing interval. Repairing
    # only the head in world space leaves the shoulder sockets teleporting.
    # Face loss is not evidence of a missing torso. Never freeze captured spine
    # motion just because the performer turned their face away.
    names=[n for n in ('Neck','Head') if n in channels]
    missing=np.flatnonzero(~visible);distance=np.min(abs(times[:,None]-times[missing][None,:]),axis=1)
    envelope=np.clip(1-distance/.13,0,1);envelope=envelope*envelope*(3-2*envelope)
    for name in names:
        values=[Quaternion(q) for _,q in channels[name]]
        bridged=fill_rotation_losses(times,values,visible)
        filtered=quaternions(times,bridged,.045)
        channels[name]=[(f,tuple(values[i].slerp(filtered[i],float(envelope[i])))) for i,(f,_) in enumerate(channels[name])]
    report.append({'model':'continuous-body-loss-bridge-v1','missing_frames':(missing+1).tolist(),'channels':names})


def translation_goal(offsets,rotation,wrist,observation,camera,aspect,mirror=False):
    """Fit image translation at captured depth; elbow / depth cannot switch."""
    from sam_ellen_screen_fit import projected
    from sam_ellen_screen import project,unproject
    target=observation['points'].copy();target[:,0]*=camera[2]/aspect
    if mirror:target[:,0]=camera[2]-target[:,0]
    support=np.asarray(observation.get('joint_weights',observation['mask']),float)*observation['mask']
    weights=support
    if weights.sum()<1e-6:return wrist.copy()
    weights/=weights.sum();size=max(.025,float(np.sqrt(np.sum(np.sum((target-target.mean(0))**2,axis=1)*weights))))
    cp=np.asarray(camera[0]@wrist);depth=max(.07,float(-cp[2]));inverse=camera[0].inverted()
    transformed=np.asarray(camera[0].to_3x3()@rotation.to_matrix())
    local=np.asarray(offsets)@transformed.T
    xy=np.asarray(project(wrist,camera))
    p=np.asarray(camera[0]@unproject(Vector(xy),depth,camera))
    value=p[:2].copy()
    def residual(x):
        origin=np.r_[x[:2],-depth]
        points=origin+local;image=projected(points,camera)
        return ((image-target)*np.sqrt(weights[:,None])/size).ravel()
    for _ in range(18):
        r=residual(value);jac=np.stack([(residual(value+np.eye(2)[j]*1e-5)-r)/1e-5 for j in range(2)],axis=1)
        step=np.linalg.solve(jac.T@jac+np.eye(2)*.004,-jac.T@r)
        step*=min(1.,.04/max(1e-8,float(np.linalg.norm(step))))
        improved=False
        for gain in (1.,.5,.2):
            trial=value+gain*step
            rr=residual(trial)
            if rr@rr<r@r:value=trial;improved=True;break
        if not improved or np.linalg.norm(step)<1e-6:break
    return inverse@Vector((float(value[0]),float(value[1]),-depth))


def reachable_goal(origin,goal,upper,lower,min_elbow=35.,max_elbow=165.):
    """Project to the limb's reachable shell without switching depth branches."""
    lo=math.sqrt(upper*upper+lower*lower-2*upper*lower*math.cos(math.radians(min_elbow)))+1e-6
    hi=math.sqrt(upper*upper+lower*lower-2*upper*lower*math.cos(math.radians(max_elbow)))-1e-6
    delta=goal-origin;length=delta.length
    return origin+delta.normalized()*float(np.clip(length,lo,hi)) if length>1e-8 else goal.copy()


def compatible_bend(axis,distance,along,radius,lower,bend,hand,limit=60.):
    """Choose the closest elbow-circle point compatible with the observed palm."""
    forward=hand@Vector((0,1,0));across=forward-axis*forward.dot(axis)
    if across.length<1e-7 or radius<1e-7:return bend
    target=-across.normalized()
    required=(math.cos(math.radians(limit))*lower-forward.dot(axis)*(distance-along))/(radius*across.length)
    if required<=-1:return bend
    maximum=math.acos(float(np.clip(required,-1,1)))
    angle=math.atan2(axis.dot(bend.cross(target)),float(np.clip(bend.dot(target),-1,1)))
    if abs(angle)<=maximum:return bend
    return (Quaternion(axis,math.copysign(abs(angle)-maximum,angle))@bend).normalized()


def stabilize_motion(scratch,rig,channels,data,options,cameras,report,repair_losses=False,reference_channels=None,protected=None):
    """Fit image trajectories before the final observed-arm transition pass."""
    if not options or options.get('space')!='FRAME' or options.get('strength',1.)<=0:return
    from sam_ellen_arm_policy import activation as interval_activation,excluded,elbow_floor,front_circle,transport_bend,PREFIXES
    from sam_ellen_arm_chain import frame_rotation,direction_evidence
    from sam_ellen_image_chain import fit_direction
    from sam_ellen_screen_fit import landmark_observations,hand_points
    from sam_ellen_screen import project
    from sam_ellen_arm_chain import anatomical_rotations,bounded_wrist_rotation,front_safe_positions,torso_front_plane
    from sam_ellen_retarget import set_global_rotation
    from sam_ellen_temporal import trajectory,quaternions
    from sam_punch_evidence import detect
    times=np.array([r['time'] for r in data['frames']]);relative=times-times[0];view=scratch.view_layers[0]
    activation=interval_activation(times,options);allowed=activation>0
    # Preserve the captured world-space hands before repairing the torso. Arm
    # channels were solved relative to that old torso; applying them under the
    # repaired torso would move a near-camera hand by hundreds of image pixels.
    original={n:list(rows) for n,rows in channels.items()}
    reference_channels=reference_channels or original
    policy=data.setdefault('_sam_arm_policy',[{} for _ in times])
    if repair_losses and options.get('start',0)==0 and options.get('end',0)==0:repair_body_losses(scratch,rig,channels,data,report)
    body_names={'UpperBody','UpperBody1','UpperBody2','Neck','Head'}
    for side in (('L','R') if options['side']=='BOTH' else (options['side'],)):
        if excluded(options,side):continue
        obs=landmark_observations(data['screen'],side,space='FRAME');segments,_=detect(data['screen'],side)
        directions=direction_evidence(data,side,options,independent_only=True)
        punch=np.array([any(s['start']-.12<=t<=s['end']+.18 for s in segments) for t in relative])
        positions=[];hands=[];bends=[];lengths=[];offsets=[];priors=[]
        def load(i,repaired=False):
            for n,rows in original.items():rig.pose.bones[n].rotation_quaternion=(channels[n] if repaired and n in body_names else rows)[i][1]
            view.update()
        for i in range(len(times)):
            load(i);a,e,w=[rig.pose.bones[n+'_'+side] for n in ('Arm','Elbow','Wrist')]
            u=e.head-a.head;v=w.head-e.head;q=w.matrix.to_quaternion()
            chest=rig.pose.bones['UpperBody2'].matrix.to_quaternion()
            hands.append(cameras[i][0].to_quaternion()@q)
            axis=(w.head-a.head).normalized();bend=e.head-a.head;bend-=axis*bend.dot(axis)
            # Filter complete hinge frames on SO(3). Averaging opposite bend
            # vectors in R3 can pass through zero and select the other branch.
            bends.append(tuple(chest.inverted()@frame_rotation(u,u.cross(v))))
            positions.append(w.head.copy());lengths.append((u.length,v.length))
            offsets.append(np.array([q.inverted()@(p-w.head) for p in hand_points(rig,side)]))
            for prefix in PREFIXES:
                name=prefix+side
                if name in reference_channels:rig.pose.bones[name].rotation_quaternion=reference_channels[name][i][1]
            view.update()
            ra,re,rw=[rig.pose.bones[n+'_'+side].head.copy() for n in ('Arm','Elbow','Wrist')]
            rb=re-ra;rx=(rw-ra).normalized();rb-=rx*rb.dot(rx)
            priors.append((ra,re,rw,rx,rb.normalized(),elbow_floor(ra,re,rw)))
        # Hand orientation is filtered in camera space, independently of the
        # arbitrary axial rotation chosen for the elbow plane.
        handq=quaternions(times,hands,.065)
        bendv=quaternions(times,bends,.085)
        measured=np.array([bool(o is not None and (not o.get('held') or o.get('interpolated'))) for o in obs],dtype=bool)&allowed
        for i,o in enumerate(obs):
            if measured[i] and not punch[i]:
                q=cameras[i][0].to_quaternion().inverted()@handq[i]
                fitted=translation_goal(offsets[i],q,positions[i],o,cameras[i],data['screen']['aspect'],options.get('mirror',False))
                # This is a continuity pass, not another unconstrained pose
                # search. A crop/reacquisition can otherwise pull a correctly
                # placed near-camera hand far outside its reachable image ray.
                displacement=(project(fitted,cameras[i])-project(positions[i],cameras[i])).length
                fitted=positions[i].lerp(fitted,min(1.,.016/max(displacement,1e-8)))
                positions[i]=fitted
            if i%50==0:yield i+1,len(times)*3
        cp=np.array([cameras[i][0]@positions[i] for i in range(len(times))]);depth=np.maximum(.04,-cp[:,2]);rays=cp[:,:2]/depth[:,None]
        speed=np.r_[0,np.linalg.norm(np.diff(rays,axis=0),axis=1)/np.maximum(1e-6,np.diff(times))]
        weights=1+3*np.clip(trajectory(times,speed,.06)/.8,0,1)
        # A visible hand crossing the frame boundary is a timing anchor. Heavy
        # smoothing there can delay its entrance or push observed fingertips
        # outside the image even though the overall median fit improves.
        edge=np.zeros(len(times))
        for i,o in enumerate(obs):
            if not measured[i]:continue
            xy=project(positions[i],cameras[i]);margin=min(xy.x,cameras[i][2]-xy.x,xy.y,1-xy.y)
            edge[i]=float(np.clip((.08-margin)/.08,0,1))
        weights+=20*edge
        # Symmetric time fitting follows source fast motion without causal lag;
        # weak monocular depth and the elbow plane receive stronger filtering.
        rays=trajectory(times,rays,.045,weights,robust=.035)
        # Near-fist depth already fits the skinned outline. Keep that measured
        # scale as a strong anchor instead of broadening its approach/retract.
        depth_weights=weights*(1+15*punch)
        depth=np.exp(trajectory(times,np.log(depth),.06,depth_weights,robust=.12))
        cp=np.c_[rays*depth[:,None],-depth]
        # An image direction leaves a depth ambiguity. Solve it over time,
        # before choosing the final elbow circle, so equally good image fits
        # cannot switch depth branches on adjacent stationary-hand frames.
        image_offset=np.zeros(len(times));image_weight=np.ones(len(times))
        image_previous=None;image_previous_time=None
        for i in range(len(times)):
            if directions[i] is None or not measured[i]:continue
            if image_previous_time is not None and times[i]-image_previous_time>.30:image_previous=None
            load(i,repaired=True);a,e,w=[rig.pose.bones[n+'_'+side] for n in ('Arm','Elbow','Wrist')]
            cue=np.array(directions[i][0],float);cue[0]*=cameras[i][2]/data['screen']['aspect'];cue/=max(1e-8,np.linalg.norm(cue))
            ee,ww,selected,diagnostic=fit_direction(a.head,e.head,cameras[i][0].inverted()@Vector(cp[i]),
                *lengths[i],cameras[i],(cue,*directions[i][1:]),torso_front_plane(rig),priors[i][5],image_previous)
            if selected is not None:
                image_offset[i]=-(cameras[i][0]@ww).z-depth[i];image_weight[i]=4.
                image_previous=(np.array(ee),np.array(ww))
                image_previous_time=times[i]
        image_offset=trajectory(times,image_offset,.12,image_weight,robust=.025)
        depth+=image_offset
        cp=np.c_[rays*depth[:,None],-depth]
        # Keep fingers coherent after every articulation / fist stage as well.
        fingers=[n for n in channels if n.endswith('_'+side) and n.startswith(('Thumb','IndexFinger','MiddleFinger','RingFinger','LittleFinger'))]
        for name in fingers:
            q=quaternions(times,[q for _,q in original[name]],.04)
            channels[name]=[(f,tuple(q[i]) if allowed[i] else value) for i,(f,value) in enumerate(original[name])]
        samples=[];hinge={};last_bend=None;last_image=None;last_image_time=None
        for i in range(len(times)):
            load(i,repaired=True);a,e,w=[rig.pose.bones[n+'_'+side] for n in ('Arm','Elbow','Wrist')]
            upper,lower=lengths[i];origin=a.head.copy();goal=cameras[i][0].inverted()@Vector(cp[i])
            floor=priors[i][5]
            from sam_ellen_arm_policy import camera_reachable_goal
            goal=camera_reachable_goal(origin,goal,upper,lower,floor,cameras[i],torso_front_plane(rig))
            goal=reachable_goal(origin,goal,upper,lower,min_elbow=floor)
            axis=goal-origin;distance=axis.length;axis.normalize()
            along=(distance*distance+upper*upper-lower*lower)/(2*max(1e-8,distance))
            frame=rig.pose.bones['UpperBody2'].matrix.to_quaternion()@bendv[i]
            bend=frame@Vector((0,1,0));bend-=axis*bend.dot(axis)
            if bend.length<.05 and last_bend is not None:bend=last_bend-axis*last_bend.dot(axis)
            if bend.length<1e-7:bend=(e.head-origin)-axis*(e.head-origin).dot(axis)
            bend.normalize();radius=math.sqrt(max(0.,upper*upper-along*along))
            hand=cameras[i][0].to_quaternion().inverted()@handq[i]
            plane=torso_front_plane(rig)
            prior=transport_bend(priors[i][4],priors[i][3],axis)
            # A palm orientation does not own the elbow circle. Retain the
            # pre-image branch; resolve incompatible wrist swing at the wrist.
            angle=math.atan2(axis.dot(prior.cross(bend)),prior.dot(bend))
            bend=Quaternion(axis,float(np.clip(angle,-math.pi/4,math.pi/4)))@prior
            elbow,goal,bend=front_circle(origin,goal,upper,lower,bend,plane,floor,last_bend)
            image_report=None
            if directions[i] is not None and measured[i]:
                if last_image_time is not None and times[i]-last_image_time>.30:last_image=None
                direction,confidence,extent=directions[i]
                direction=np.asarray(direction,float).copy();direction[0]*=cameras[i][2]/data['screen']['aspect']
                direction/=max(1e-8,np.linalg.norm(direction))
                elbow,goal,selected,image_report=fit_direction(origin,elbow,goal,upper,lower,cameras[i],
                    (direction,confidence,extent),plane,floor,last_image,depth_preference=30.)
                if selected is not None:
                    bend=selected;last_image=(np.array(elbow),np.array(goal));last_image_time=times[i]
            elif last_image_time is not None and measured[i] and times[i]-last_image_time<.30:
                # A brief independent-pose loss must not snap back to the SAM
                # elbow branch. Release the last image plane continuously.
                amount=1-float((times[i]-last_image_time)/.30)
                held=transport_bend(last_bend,None,(goal-origin).normalized())
                blend=held*amount+bend*(1-amount)
                elbow,goal,bend=front_circle(origin,goal,upper,lower,blend,plane,floor,last_bend)
            last_bend=bend.copy()
            from sam_ellen_arm_policy import wrist_limit
            hand=bounded_wrist_rotation(hand,goal-elbow,wrist_limit(180-math.degrees((elbow-origin).angle(goal-elbow))))
            anatomical_rotations(a,e,w,origin,elbow,goal,hand,set_global_rotation,view,hinge)
            raw=hinge.get('hand_twist',0.)
            ref_u=(priors[i][1]-priors[i][0]).normalized();ref_v=(priors[i][2]-priors[i][1]).normalized()
            policy[i][side]={'minimum_elbow':floor,'source_elbow':180-math.degrees(ref_u.angle(ref_v)),
                'reference_normal':list(ref_u.cross(ref_v).normalized()),'reference_upper':list(ref_u),
                'hand_orientation_owns_elbow':False}
            if image_report is not None:policy[i][side]['image_direction_fit']=image_report
            samples.append({'hand':w.matrix.to_quaternion().copy(),'roll':raw})
            if allowed[i]:
                for prefix in ('Arm_','Elbow_','Wrist_','ArmTwist_'):
                    name=prefix+side
                    if name in channels:channels[name][i]=channels[name][i][0],tuple(rig.pose.bones[name].rotation_quaternion)
            if i%50==0:yield len(times)+i+1,len(times)*3
        # Repartition roll without changing the final palm pose. Smooth this
        # scalar physical controller, not equivalent wrist q/-q components.
        rolls=np.array([s['roll'] for s in samples]);rolls=np.clip(rolls*.75,-math.radians(150),math.radians(150))
        rolls=trajectory(times,rolls,.12,robust=math.radians(25))
        for i in range(len(times)):
            if not allowed[i]:continue
            for n,rows in channels.items():rig.pose.bones[n].rotation_quaternion=rows[i][1]
            controller=rig.pose.bones.get('HandTwist_'+side)
            if controller:
                controller.rotation_quaternion=Quaternion((0,1,0),float(rolls[i]));view.update()
                set_global_rotation(rig.pose.bones['Wrist_'+side],samples[i]['hand']);view.update()
                for prefix in ('HandTwist_','Wrist_'):
                    name=prefix+side
                    if name in channels:channels[name][i]=channels[name][i][0],tuple(rig.pose.bones[name].rotation_quaternion)
            if i%80==0:yield 2*len(times)+i+1,len(times)*3
        for name in [p+side for p in ('Arm_','Elbow_','Wrist_','ArmTwist_','HandTwist_')]+fingers:
            if name not in channels:continue
            for i,weight in enumerate(activation):
                if weight<1:
                    frame,q=channels[name][i];channels[name][i]=frame,tuple(Quaternion(original[name][i][1]).slerp(Quaternion(q),float(weight)))
        report.append({'model':'final-motion-trajectory-v2','side':side,'frames':int(allowed.sum()),'source_wrist_ray_samples':int((measured&~punch).sum()),
            'ray_seconds':.045,'depth_seconds':.06,'hinge_seconds':.085,'palm_seconds':.065,'roll_seconds':.12,'fingers_seconds':.04,'translation_trust_region_source_height':.016})
        yield len(times)*3,len(times)*3

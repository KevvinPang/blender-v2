"""Unobserved arms return to a fixed, lowered pose instead of SAM guesses."""
import numpy as np
from mathutils import Quaternion,Vector


def screen_hand(screen,index,side,*,include_inferred=False):
    tracked=screen.get('_tracked_hands',{}).get(side)
    if tracked is not None and len(tracked)==len(screen['frames']):
        hand=tracked[index]
        return hand if hand is not None and (include_inferred or not hand.get('inferred')) else None
    primary=screen['frames'][index].get('hands',{}).get(side)
    if hand_localization_usable(primary):return primary
    fallback=screen.get('wilor_hands',[])
    return fallback[index].get(side) if len(fallback)==len(screen['frames']) else None


def merge_cached_wilor(data,take_path):
    """Use the same evidence preparation as the entry video worker."""
    from sam_ellen_entry import merge_wilor
    return merge_wilor(data,take_path)


def hand_localization_usable(hand):
    if not hand:return False
    p=np.asarray(hand.get('points',[]),dtype=float);mask=np.asarray(hand.get('in_frame',[True]*21))
    if p.shape!=(21,2) or not np.isfinite(p).all() or mask.shape!=(21,) or mask.sum()<4:return False
    if np.linalg.norm(np.ptp(p,axis=0))<.015:return False
    return True


def hand_landmark_mask(hand):
    """Reject an unreliable joint without suppressing the visible whole hand."""
    if not hand_localization_usable(hand):return np.zeros(21,dtype=bool)
    p=np.asarray(hand['points'],dtype=float)
    mask=np.asarray(hand.get('in_frame',[True]*21),dtype=bool).copy()
    # A wrist deep inside the fingers cannot supply the palm/forearm axis.
    # MCPs and visible fingers still constrain a fitted native hand and wrist.
    from sam_ellen_punch_mesh import hull
    polygon=hull(p[1:]);edges=np.roll(polygon,-1,axis=0)-polygon
    relative=p[0]-polygon;cross=edges[:,0]*relative[:,1]-edges[:,1]*relative[:,0]
    distances=cross/np.maximum(1e-8,np.linalg.norm(edges,axis=1))
    if len(polygon)>=3 and np.min(distances)>.002:mask[0]=False
    mcp=p[[5,9,13,17]];proximal=(p[[6,10,14,18]]-mcp).mean(0)
    # Folded fingers point back towards the wrist. Their MCP-to-PIP vector
    # cannot identify the base side of a closed fist.
    curled=np.mean([np.linalg.norm(p[j+3]-p[0])<1.35*np.linalg.norm(p[j]-p[0])
                    for j in (5,9,13,17)])>=.75
    if not curled and np.linalg.norm(proximal)>.015:
        # Wrist should be on the base side of the MCP row, not on its finger
        # side. The latter is common for a large foreshortened gripping hand.
        if (p[0]-mcp.mean(0))@(proximal/np.linalg.norm(proximal))>.01:mask[0]=False
    return mask


def observed_frames(data,side):
    result=[]
    screen=data.get('screen',{})
    for i,row in enumerate(screen.get('frames',[])):
        hand=screen_hand(screen,i,side)
        result.append(bool(hand_localization_usable(hand) and not hand.get('inferred')))
    result=np.asarray(result,dtype=bool)
    edges=np.diff(np.r_[False,result,False].astype(int))
    for first,end in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        if end-first<3:result[first:end]=False
    return result


def protected_frames(data,side,contact=None,punch=None):
    """Explicit contact/punch tracking can supply evidence during hand loss."""
    times=np.array([r['time']-data['frames'][0]['time'] for r in data['frames']]);mask=np.zeros(len(times),dtype=bool)
    from sam_punch_evidence import detect
    segments,observations=detect(data.get('screen',{}),side) if data.get('screen') else ([],[])
    for segment in segments:
        if segment.get('silhouette'):
            last=segment.get('last_mask_index',len(times)-1)
            mask|=(times>=segment['start'])&(times<segment['end'])&(np.arange(len(times))<=last)&np.array([bool(o and o.get('silhouette')) for o in observations])
    if contact and contact.get('side')==side and contact.get('strength',0)>0:
        mask|=(times>contact['start'])&(times<contact['end'])
    if punch and punch.get('side')==side and punch.get('strength',0)>0:
        if punch.get('automatic'):
            from sam_punch_evidence import detect
            segments,observations=detect(data['screen'],side)
            for s in segments:
                mask|=(times>s['start'])&(times<s['end'])&np.array([o is not None for o in observations])
        else:mask|=(times>punch['start'])&(times<punch['end'])
    return mask


def visibility_envelope(times,observed,hold=.10,fade=.30):
    """Bridge short losses; release extended losses to idle with no SAM motion."""
    times=np.asarray(times);visible=np.flatnonzero(observed);weights=np.zeros(len(times))
    if not len(visible):return weights
    active=observed.copy()
    for left,right in zip(visible,visible[1:]):
        if times[right]-times[left]<=.24:active[left:right+1]=True
    dt=float(np.median(np.diff(times))) if len(times)>1 else 1/30
    for i,time in enumerate(times):
        if active[i]:weights[i]=1.;continue
        left=visible[visible<i];right=visible[visible>i]
        if len(left) and len(right) and times[right[0]]-times[left[-1]]<=.24:
            weights[i]=1.;continue
        if len(left):
            gap=time-times[left[-1]]
            w=np.clip(1-(gap-hold)/fade,0,1);weights[i]=w*w*(3-2*w)
        # A future detection is not evidence for an unseen current arm. Only
        # short bounded dropouts bridge; extended absence returns to idle.
    return weights


def arm_names(rig,side):
    return [b.name for b in rig.pose.bones if b.name.endswith('_'+side) and b.name.startswith(
        ('Shoulder','Arm_','Elbow_','Wrist_','ArmTwist_','HandTwist_',
         'Thumb','IndexFinger','MiddleFinger','RingFinger','LittleFinger'))]


def neutral_pose(rig,side,view,spread=.24):
    """Return local rotations for a relaxed lowered arm in the torso frame."""
    from sam_ellen_retarget import set_global_rotation
    names=arm_names(rig,side);saved={n:rig.pose.bones[n].rotation_quaternion.copy() for n in names}
    for n in names:rig.pose.bones[n].rotation_quaternion=Quaternion()
    view.update()
    chest=rig.pose.bones.get('UpperBody2') or rig.pose.bones.get('UpperBody1')
    delta=(chest.matrix.to_quaternion()@chest.bone.matrix_local.to_quaternion().inverted()) if chest else Quaternion()
    sign=1 if side=='L' else -1
    # About fourteen degrees at the shoulder leaves a small gap to the torso;
    # a slightly forward, bent elbow avoids a rigid arm glued to the body.
    forearm_spread=spread-.04
    for part,v in [('Arm',(sign*spread,-.06,-1)),('Elbow',(sign*forearm_spread,-.36,-1)),('Wrist',(sign*forearm_spread,-.36,-1))]:
        b=rig.pose.bones[part+'_'+side];q=delta@b.bone.matrix_local.to_quaternion()
        desired=delta@Vector(v).normalized()
        set_global_rotation(b,(q@Vector((0,1,0))).rotation_difference(desired)@q);view.update()
    from sam_ellen_arm_chain import anatomical_rotations
    a,e,w=[rig.pose.bones[n+'_'+side] for n in ('Arm','Elbow','Wrist')]
    anatomical_rotations(a,e,w,a.head.copy(),e.head.copy(),w.head.copy(),w.matrix.to_quaternion().copy(),
                         set_global_rotation,view)
    result={n:rig.pose.bones[n].rotation_quaternion.copy() for n in names}
    for n,q in saved.items():rig.pose.bones[n].rotation_quaternion=q
    view.update()
    return result


def apply_visibility(scratch,rig,channels,data,side,protected=None,report=None,initialize_only=False,cameras=None,allowed=None,mirror=False,image_weights=None):
    """Follow reliable image poses; converge to a fixed idle when evidence ends.

    Measured wrist pixels own visible hand translation, while a fitted elbow
    phase joins new direction evidence continuously. Whole-chain rate limits
    remain responsible for off-screen entry and unsupported loss/release poses.
    """
    import math
    from sam_ellen_arm_chain import frame_rotation,anatomical_rotations,continuous_wrist_rotation,front_safe_positions,torso_front_plane
    from sam_ellen_retarget import set_global_rotation
    times=np.array([r['time'] for r in data['frames']]);observed=observed_frames(data,side)
    if len(observed)!=len(times):return
    allowed=np.ones(len(times),dtype=bool) if allowed is None else np.asarray(allowed,dtype=bool)
    protected=np.zeros(len(times),dtype=bool) if protected is None else np.asarray(protected,dtype=bool)
    observed|=protected
    observations=None
    if cameras is not None and not initialize_only:
        from sam_ellen_screen_fit import landmark_observations
        observations=landmark_observations(data['screen'],side,space='FRAME')
        # Entry timing is measured directly in video pixels. A centered filter
        # can still round a quick raise followed by a stop into a delayed sweep.
        starts=np.flatnonzero(np.diff(np.r_[False,observed].astype(int))==1)
        for start in starts:
            for i in range(int(start),len(times)):
                elapsed=times[i]-times[start]
                if elapsed>=.20 or not observed[i]:break
                hand=screen_hand(data['screen'],i,side);o=observations[i]
                if hand is None or o is None or o.get('held'):continue
                blend=float(np.clip((.20-elapsed)/.10,0,1));blend=blend*blend*(3-2*blend)
                o['points']=o['points']*(1-blend)+np.asarray(hand['points'])*blend
    weights=visibility_envelope(times,observed);valid=np.flatnonzero(observed);view=scratch.view_layers[0]
    policy=data.get('_sam_arm_policy',[])
    # A lost hand releases the LAST supported chain. The hidden SAM elbow
    # must not impose a new flexion floor while that release is running.
    supported_floors=np.full(len(times),35.)
    if len(policy)==len(times):
        for i in range(len(times)):
            if observed[i]:
                supported_floors[i]=policy[i].get(side,{}).get('minimum_elbow',35.)
            else:
                before=valid[valid<i];after=valid[valid>i]
                if len(before):
                    left=int(before[-1]);supported_floors[i]=policy[left].get(side,{}).get('minimum_elbow',35.)
                    if len(after) and times[after[0]]-times[left]<=.24:
                        right=int(after[0]);alpha=(times[i]-times[left])/(times[right]-times[left])
                        supported_floors[i]=(1-alpha)*supported_floors[i]+alpha*policy[right].get(side,{}).get('minimum_elbow',35.)
    entry={}
    if cameras is not None:
        # A measured hand entering at an image edge permits an off-screen
        # continuation, never a visible preparatory gesture. It supplies a
        # boundary and destination, rather than reviving an unseen SAM arm.
        raw=observed_frames(data,side)
        starts=np.flatnonzero(np.diff(np.r_[False,raw].astype(int))==1)
        for start in starts:
            hand=screen_hand(data['screen'],int(start),side)
            if not hand:continue
            p=np.asarray(hand['palm'])/np.array([data['screen']['aspect'],1.])
            if mirror:p[0]=1-p[0]
            margins=[p[0],1-p[0],p[1],1-p[1]];edge=int(np.argmin(margins))
            if min(margins)>.16:continue
            for i in range(start-1,-1,-1):
                if times[start]-times[i]>=.23 or observed[i]:break
                weight=float(np.clip(1-(times[start]-times[i])/.23,0,1))
                weight=weight*weight*(3-2*weight)
                points=np.asarray(hand['points']);offset=points-np.asarray(hand['wrist'])
                if mirror:offset[:,0]*=-1
                margins=[max(0.,offset[:,0].max()),max(0.,-offset[:,0].min()),
                         max(0.,offset[:,1].max()),max(0.,-offset[:,1].min())]
                entry[i]=(int(start),edge,float(margins[edge]+.015))
                weights[i]=max(weights[i],weight)
    neutral=neutral_pose(rig,side,view)
    for n,q in neutral.items():
        if n not in channels:channels[n]=[(f,tuple(q)) for f,_ in next(iter(channels.values()))]
    if initialize_only:
        # Seed missing input with idle. Only the final solved stage owns fades
        # and rate limits; applying them before IK as well introduces double lag.
        for i,weight in enumerate(weights):
            if allowed[i] and weight<=0:
                for n,q in neutral.items():channels[n][i]=channels[n][i][0],tuple(q)
        return
    originals={n:list(rows) for n,rows in channels.items()}
    image_goals={}
    if observations is not None:
        from sam_ellen_arm_transition import observed_goals
        image_goals=observed_goals(scratch,rig,originals,data,side,cameras,observations,
                                   observed&allowed,protected,supported_floors,mirror,image_weights)
    arm,elbow,wrist=[rig.pose.bones[n+'_'+side] for n in ('Arm','Elbow','Wrist')]
    def geometry(rotations):
        for n,q in rotations.items():
            if not n.startswith('Shoulder'):rig.pose.bones[n].rotation_quaternion=q
        view.update();a,e,w=[b.head.copy() for b in (arm,elbow,wrist)]
        u=(e-a).normalized();v=(w-e).normalized();normal=u.cross(v).normalized()
        return (frame_rotation(u,normal),u.angle(v),wrist.matrix.to_quaternion().copy(),
                (e-a).length,(w-e).length,a,e,w)
    def toward(first,last,limit):
        angle=2*math.acos(min(1.,abs(first.dot(last))))
        return first.slerp(last,min(1.,limit/max(1e-8,angle))),angle<=limit+1e-6
    previous=None;hinge={};idle_settled=True;previous_roll=None
    last_axis=None;last_bend=None
    def fraction_to(frame,flex,previous,step,flex_step):
        upper_angle=2*math.acos(min(1.,abs(previous[0].dot(frame))))
        old_lower=Quaternion(previous[0]@Vector((1,0,0)),previous[1])@previous[0]
        new_lower=Quaternion(frame@Vector((1,0,0)),flex)@frame
        lower_angle=2*math.acos(min(1.,abs(old_lower.dot(new_lower))))
        # Fast observed motion can rotate the hinge faster, without allowing
        # its combined lower-arm rotation to jump beyond the scene limit.
        return min(1.,step/max(1e-8,upper_angle),flex_step/max(1e-8,abs(flex-previous[1])),
                   math.radians(1140*dt)/max(1e-8,lower_angle))
    for i,weight in enumerate(weights):
        if not allowed[i]:
            previous=None;hinge={};idle_settled=True;previous_roll=None;last_axis=None;last_bend=None
            continue
        if weight<=0 and idle_settled:
            for n,q in neutral.items():channels[n][i]=channels[n][i][0],tuple(q)
            previous=None
            last_axis=None;last_bend=None
            if len(policy)==len(times) and side in policy[i]:
                policy[i][side].update(mode='neutral',observed=False,arm_motion_weight=0.,active_minimum_elbow=35.)
            if report is not None:report.append({'frame':i+1,'observed':False,'arm_motion_weight':0.,'mode':'neutral'})
            continue
        for n,rows in originals.items():rig.pose.bones[n].rotation_quaternion=rows[i][1]
        left=valid[valid<i];right=valid[valid>i];targets={}
        for n,idle in neutral.items():
            if i in entry:target=Quaternion(originals[n][entry[i][0]][1])
            elif observed[i]:target=Quaternion(originals[n][i][1])
            elif len(left):
                target=Quaternion(originals[n][left[-1]][1])
                if len(right) and times[right[0]]-times[left[-1]]<=.24:
                    t=(times[i]-times[left[-1]])/(times[right[0]]-times[left[-1]])
                    target=target.slerp(Quaternion(originals[n][right[0]][1]),float(t))
            elif len(right):target=Quaternion(originals[n][right[0]][1])
            else:target=idle.copy()
            targets[n]=target
            rig.pose.bones[n].rotation_quaternion=idle.slerp(target,float(weight))
        view.update();target_pose=geometry(targets)
        idle_pose=geometry(neutral) if weight<1 or previous is None else target_pose
        # Interpolate the wrist in space before solving its hinge. Interpolating
        # the two joint angles separately can sweep a straight arm into the lens.
        a=arm.head.copy();length1,length2=idle_pose[3:5]
        w=idle_pose[7].lerp(target_pose[7],float(weight));axis=w-a
        image_owned=i in image_goals
        if image_owned:
            # Joint-angle limiting after image fitting makes an entering hand
            # lag, then race across the image. The observed wrist ray owns its
            # position; only the ambiguous elbow-circle phase is rate limited.
            w=image_goals[i]['goal'].copy()
            axis=w-a
        if i in entry:
            from sam_ellen_screen import project,unproject
            from sam_ellen_motion_continuity import translation_goal
            from sam_ellen_screen_fit import hand_points
            cam=cameras[i];future=entry[i][0];first=observations[future] if observations else None
            # Approach the measured entry boundary, not the projection of a
            # world-space idle blend (which can lie far below the video).
            boundary=target_pose[7].copy()
            if first is not None:
                for n,q in targets.items():rig.pose.bones[n].rotation_quaternion=q
                view.update();q=wrist.matrix.to_quaternion();origin=wrist.head.copy()
                local=np.array([q.inverted()@(p-origin) for p in hand_points(rig,side)])
                boundary=translation_goal(local,q,origin,first,cam,data['screen']['aspect'],mirror)
            xy=project(boundary,cam);depth=max(.10,-(cam[0]@boundary).z)
            edge,margin=entry[i][1:]
            if edge==0:xy.x=min(xy.x,-margin)
            elif edge==1:xy.x=max(xy.x,cam[2]+margin)
            elif edge==2:xy.y=min(xy.y,-margin)
            else:xy.y=max(xy.y,1+margin)
            w=unproject(xy,depth,cam);axis=w-a
        minimum=math.sqrt(length1**2+length2**2-2*length1*length2*math.cos(math.radians(35)))
        maximum=math.sqrt(length1**2+length2**2-2*length1*length2*math.cos(math.radians(165)))
        distance=float(np.clip(axis.length,minimum,maximum));axis.normalize();w=a+axis*distance
        along=(distance**2+length1**2-length2**2)/(2*distance)
        bend=idle_pose[6].lerp(target_pose[6],float(weight))-a;bend-=axis*bend.dot(axis)
        if bend.length<1e-6:bend=axis.orthogonal()
        e=a+axis*along+bend.normalized()*math.sqrt(max(0.,length1**2-along**2))
        if i in entry and entry[i][0] in image_goals:
            from sam_ellen_arm_policy import transport_bend,front_circle
            seed=image_goals[entry[i][0]];unit=axis.normalized()
            # The off-screen boundary continuation must meet the same fitted
            # first-frame branch, rather than the old unsmoothed SAM branch.
            wanted=transport_bend(seed['bend'],seed['axis'],unit)
            idle_bend=transport_bend(idle_pose[6]-a,None,unit)
            turn=math.atan2(unit.dot(idle_bend.cross(wanted)),idle_bend.dot(wanted))
            bend=Quaternion(unit,float(weight)*turn)@idle_bend
            e,w,_=front_circle(a,w,length1,length2,bend,torso_front_plane(rig),
                               35.+(supported_floors[i]-35.)*weight,idle_bend)
        u=(e-a).normalized();v=(w-e).normalized();normal=u.cross(v).normalized()
        if image_owned:
            from sam_ellen_arm_policy import transport_bend,front_circle
            unit=axis.normalized();wanted=transport_bend(image_goals[i]['bend'],None,unit)
            prior=transport_bend(last_bend,last_axis,unit) if last_bend is not None else wanted
            turn=math.atan2(unit.dot(prior.cross(wanted)),prior.dot(wanted))
            elapsed=times[i]-times[i-1] if i else 1/data.get('fps',30)
            # A new direction estimate must join the existing branch. Palm
            # image speed cannot justify an instantaneous unseen elbow turn.
            limit=math.radians(360*elapsed)
            bend=Quaternion(unit,float(np.clip(turn,-limit,limit)))@prior
            e,w,_=front_circle(a,w,length1,length2,bend,torso_front_plane(rig),supported_floors[i],prior)
            u=(e-a).normalized();v=(w-e).normalized();normal=u.cross(v).normalized()
        upper=frame_rotation(u,normal);flex=u.angle(v)
        hand=idle_pose[2].slerp(target_pose[2],float(weight))
        dt=times[i]-times[i-1] if i else 1/data.get('fps',30)
        rate=540.
        pose_rows=data['screen'].get('pose_arms',[])
        pose=pose_rows[i].get(side) if len(pose_rows)==len(times) else None
        if not image_owned and i and observed[i] and observed[i-1] and pose and pose.get('confidence',0)>=.5:
            now=screen_hand(data['screen'],i,side);before=screen_hand(data['screen'],i-1,side)
            if now and before:
                image_speed=np.linalg.norm(np.array(now['palm'])-before['palm'])/max(dt,1e-8)
                rate+=600*float(np.clip((image_speed-.5)/1.,0,1))
        step=math.radians(rate*dt);flex_step=math.radians(540*dt);hand_step=math.radians(900*dt)
        settled=weight<=0
        if previous is None:previous=target_pose[:3] if i==0 and observed[i] else idle_pose[:3]
        if previous is not None:
            angle=2*math.acos(min(1.,abs(previous[0].dot(upper))))
            difference=flex-previous[1]
            fraction=1. if image_owned else fraction_to(upper,flex,previous,step,flex_step)
            upper=previous[0].slerp(upper,fraction);flex=previous[1]+difference*fraction
            settled&=fraction>=1
            hand,ok=toward(previous[2],hand,hand_step);settled&=ok
            u=upper@Vector((0,1,0));normal=upper@Vector((1,0,0));v=Quaternion(normal,flex)@u
            from sam_ellen_arm_policy import wrist_limit
            hand=continuous_wrist_rotation(hand,v,previous[2],hand_step,wrist_limit(180-math.degrees(flex)))
        # Restore the already blended finger/shoulder controls before solving.
        for n,idle in neutral.items():rig.pose.bones[n].rotation_quaternion=idle.slerp(targets[n],float(weight))
        view.update()
        if settled:
            for n,q in neutral.items():rig.pose.bones[n].rotation_quaternion=q
            view.update();hinge={}
        else:
            u=upper@Vector((0,1,0));normal=upper@Vector((1,0,0));v=Quaternion(normal,flex)@u
            a=arm.head.copy();e=a+u*idle_pose[3];w=e+v*idle_pose[4]
            # Intersect the front half-space with the frame/flexion speed
            # limits. A final position projection alone can increase a step.
            plane=torso_front_plane(rig)
            for _ in range(24):
                from sam_ellen_arm_policy import front_circle
                policy=data.get('_sam_arm_policy',[])
                minimum=supported_floors[i]
                minimum=35.+(minimum-35.)*weight
                axis=(w-a).normalized();bend=e-a;bend-=axis*bend.dot(axis)
                e,w,_=front_circle(a,w,idle_pose[3],idle_pose[4],bend,plane,minimum)
                su=(e-a).normalized();sv=(w-e).normalized()
                safe_frame=frame_rotation(su,su.cross(sv));safe_flex=su.angle(sv)
                angle=2*math.acos(min(1.,abs(previous[0].dot(safe_frame))))
                difference=safe_flex-previous[1]
                fraction=1. if image_owned else fraction_to(safe_frame,safe_flex,previous,step,flex_step)
                if fraction>=1-1e-6:break
                safe_frame=previous[0].slerp(safe_frame,fraction)
                safe_flex=previous[1]+difference*fraction
                su=safe_frame@Vector((0,1,0));sn=safe_frame@Vector((1,0,0))
                sv=Quaternion(sn,safe_flex)@su
                e=a+su*idle_pose[3];w=e+sv*idle_pose[4]
            from sam_ellen_arm_policy import wrist_limit
            hand=continuous_wrist_rotation(hand,w-e,previous[2],hand_step,wrist_limit(180-math.degrees((e-a).angle(w-e))))
            anatomical_rotations(arm,elbow,wrist,a,e,w,hand,set_global_rotation,view,hinge)
        # The MMD twist controller drives fractional helper bones. A smooth
        # palm alone cannot prevent a controller branch reset from snapping
        # the forearm skin. Limit this physical control, preserving the palm.
        controller=rig.pose.bones.get('HandTwist_'+side)
        roll_settled=True
        if controller:
            palm=wrist.matrix.to_quaternion().copy();q=controller.rotation_quaternion.normalized()
            if q.w<0:q.negate()
            _,roll=q.to_swing_twist('Y')
            # MMD fractional twist helpers read a wrapped driver angle. A
            # controller crossing pi can flip those helpers despite a smooth
            # palm. Keep its physical driver away from that branch boundary;
            # the residual wrist rotation below retains the captured palm.
            roll=float(np.clip(roll,-np.deg2rad(170.),np.deg2rad(170.)))
            if previous_roll is not None:
                desired=roll;roll=previous_roll+float(np.clip(roll-previous_roll,-math.radians(600*dt),math.radians(600*dt)))
                roll_settled=abs(roll-desired)<1e-5
            previous_roll=roll
            controller.rotation_quaternion=Quaternion((0,1,0),roll);view.update()
            set_global_rotation(wrist,palm);view.update()
        actual_u=(elbow.head-arm.head).normalized();actual_v=(wrist.head-elbow.head).normalized()
        last_axis=(wrist.head-arm.head).normalized()
        last_bend=elbow.head-arm.head;last_bend-=last_axis*last_bend.dot(last_axis)
        last_bend.normalize()
        previous=(frame_rotation(actual_u,actual_u.cross(actual_v)),float(actual_u.angle(actual_v)),wrist.matrix.to_quaternion().copy());idle_settled=bool(settled and roll_settled)
        policy=data.get('_sam_arm_policy',[])
        if len(policy)==len(times) and side in policy[i]:
            policy[i][side].update(mode='neutral' if settled else 'tracked' if observed[i] else 'edge-entry' if i in entry else 'returning',
                observed=bool(observed[i]),arm_motion_weight=float(weight),
                active_minimum_elbow=35.+(supported_floors[i]-35.)*weight)
            policy[i][side]['wrist_image_owned']=image_owned
        for n in neutral:channels[n][i]=channels[n][i][0],tuple(rig.pose.bones[n].rotation_quaternion)
        if report is not None:report.append({'frame':i+1,'observed':bool(observed[i]),
            'arm_motion_weight':float(weight),'mode':'neutral' if settled else 'tracked' if observed[i] else 'edge-entry' if i in entry else 'returning'})
    if cameras is not None and not initialize_only:
        from sam_ellen_entry_solver import solve_entries
        solve_entries(scratch,rig,channels,data,side,cameras,allowed,protected,supported_floors,
                      mirror,image_weights,report)
        from sam_ellen_entry_direction import align_forearms
        direction_allowed=allowed if image_weights is None else allowed&(np.asarray(image_weights)>=.999)
        align_forearms(scratch,rig,channels,data,side,cameras,direction_allowed,protected,supported_floors,mirror)
        # Image fitting is a proposal. A single final trajectory solve owns
        # elbow evidence, loss recovery, clearance and speed simultaneously.
        # Independent body/elbow guards here used to invalidate one another.

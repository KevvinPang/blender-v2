"""Image-supported two-link arm geometry and anatomical bone frames."""
import math
import numpy as np
from mathutils import Matrix, Quaternion, Vector


def frame_rotation(direction, normal):
    y=Vector(direction).normalized();x=Vector(normal)
    x-=y*x.dot(y);x.normalize()
    return Matrix((x,y,x.cross(y))).transposed().to_quaternion().normalized()


def direction_evidence(data,side,options,independent_only=False):
    from sam_ellen_temporal import trajectory
    rows=data['screen']['frames'];times=np.array([r['time'] for r in rows])
    values=np.zeros((len(rows),2));weights=np.zeros(len(rows));extent=np.zeros(len(rows))
    from sam_ellen_arm_visibility import observed_frames,hand_landmark_mask,screen_hand
    usable=observed_frames(data,side)
    pose_rows=data['screen'].get('pose_arms',[])
    for i,row in enumerate(rows):
        if not usable[i]:continue
        hand=screen_hand(data['screen'],i,side)
        # Visible fingers constrain hand articulation, never an unseen elbow.
        # A rejected wrist also invalidates the wrist-seeded color-axis cue.
        if not hand_landmark_mask(hand)[0]:continue
        pose=pose_rows[i].get(side) if len(pose_rows)==len(rows) else None
        if independent_only and not (pose and pose.get('confidence',0)>=.5 and pose.get('elbow_observed')):continue
        observed=pose if pose and pose.get('confidence',0)>=.5 and pose.get('elbow_observed') else row.get('arms',{}).get(side)
        if observed and observed.get('confidence',0)>.25:
            values[i]=observed['direction'];weights[i]=observed['confidence'];extent[i]=observed['visible_length']
        else:
            hand=row.get('hands',{}).get(side)
            if hand:
                p=np.array(hand['points']);v=p[0]-p[[5,9,13,17]].mean(0)
                if np.linalg.norm(v)>.02:values[i]=v/np.linalg.norm(v);weights[i]=.12
    valid=np.flatnonzero(weights)
    if len(valid)<2:return [None]*len(rows)
    for j in range(2):values[:,j]=np.interp(times,times[valid],values[valid,j])
    values=trajectory(times,values,.065,weights,robust=.25)
    # Bridge a brief landmark dropout with an image direction, not a fabricated elbow.
    nearest=np.min(abs(times[:,None]-times[valid][None,:]),axis=1)
    result=[]
    for i in range(len(rows)):
        if usable[i] and not hand_landmark_mask(screen_hand(data['screen'],i,side))[0]:
            result.append(None);continue
        if nearest[i]>.20 or np.linalg.norm(values[i])<.2:result.append(None);continue
        if independent_only and weights[i]==0:
            before=valid[valid<i];after=valid[valid>i]
            # A later visible forearm cannot establish an earlier entry pose.
            # Bridge bounded gaps, or briefly hold a preceding measurement.
            if not len(before) or times[i]-times[before[-1]]>.20:
                result.append(None);continue
            if (not len(after) or times[after[0]]-times[i]>.20) and times[i]-times[before[-1]]>.10:
                result.append(None);continue
        v=values[i]/np.linalg.norm(values[i])
        if options.get('mirror'):v[0]*=-1
        confidence=weights[i]
        if independent_only and confidence==0:
            confidence=float(np.interp(times[i],times[valid],weights[valid]))
        result.append((v,float(max(confidence,.1)),float(extent[i])))
    return result


def refine_goal(arm,elbow,wrist,goal,bend,camera,evidence,previous=None,depth_strength=.25,front_depth=None,
                max_step_degrees=18,palm=None,front_plane=None,hand_axis=None):
    """Select wrist depth and elbow together, retaining its image ray and lengths.

    Anatomical angle limits exclude folded/fully straight singularities. Visible
    forearm direction remains useful when the actual elbow is outside the image.
    """
    from sam_ellen_screen import project,unproject
    from sam_ellen_screen_fit import projected
    a=np.array(arm.head);e=np.array(elbow.head);w=np.array(wrist.head)
    upper=float(np.linalg.norm(e-a));lower=float(np.linalg.norm(w-e));reach=upper+lower
    xy=project(Vector(goal),camera);transform=np.asarray(camera[0]);inverse=camera[0].inverted()
    source_depth=-(camera[0]@Vector(goal)).z;capture_depth=-(camera[0]@wrist.head).z
    shoulder_depth=-(camera[0]@arm.head).z
    depths=np.unique(np.r_[np.linspace(max(.085,shoulder_depth-reach),shoulder_depth+reach,49),
                           source_depth,capture_depth])
    candidates=np.array([unproject(xy,float(z),camera) for z in depths])
    vectors=candidates-a;dist=np.linalg.norm(vectors,axis=1)
    min_dist=math.sqrt(upper**2+lower**2-2*upper*lower*math.cos(math.radians(35)))
    max_dist=math.sqrt(upper**2+lower**2-2*upper*lower*math.cos(math.radians(165)))
    valid=(dist>=min_dist)&(dist<=max_dist)
    ray_limited=not valid.any()
    if ray_limited:
        vector=np.array(goal)-a;length=max(1e-8,np.linalg.norm(vector))
        candidates=(a+vector/length*np.clip(length,min_dist+1e-5,max_dist-1e-5))[None,:]
        depths=np.array([-(camera[0]@Vector(candidates[0])).z]);dist=np.linalg.norm(candidates-a,axis=1)
    else:candidates=candidates[valid];dist=dist[valid];depths=depths[valid]
    axis=(candidates-a)/dist[:,None]
    along=(dist**2+upper**2-lower**2)/(2*dist)
    radius=np.sqrt(np.maximum(0,upper**2-along**2))
    seed=np.tile(np.array([0.,1.,0.]),(len(axis),1))
    seed[abs(axis[:,1])>.9]=[1.,0.,0.]
    x=np.cross(axis,seed);x/=np.linalg.norm(x,axis=1)[:,None];y=np.cross(axis,x)
    angles=np.arange(96)*math.tau/96
    bends=x[:,None,:]*np.cos(angles)[None,:,None]+y[:,None,:]*np.sin(angles)[None,:,None]
    elbows=a+axis[:,None,:]*along[:,None,None]+bends*radius[:,None,None]
    goals=np.broadcast_to(candidates[:,None,:],elbows.shape)
    lower_dirs=(goals-elbows)/lower;upper_dirs=(elbows-a)/upper
    normals=np.cross(upper_dirs,lower_dirs);normals/=np.maximum(1e-8,np.linalg.norm(normals,axis=2))[:,:,None]
    camera_points=np.c_[elbows.reshape(-1,3),np.ones(elbows.shape[0]*elbows.shape[1])]@transform.T
    projected_elbows=projected(camera_points[:,:3],camera).reshape(*elbows.shape[:2],2)
    projected_vectors=projected_elbows-np.array(xy)
    lengths=np.linalg.norm(projected_vectors,axis=2)
    directions=projected_vectors/np.maximum(1e-7,lengths[:,:,None])
    cost=depth_strength*((depths-source_depth)/reach)**2
    if front_depth is not None:
        cost+=20*(np.maximum(0,depths-front_depth)/reach)**2
    cost=cost[:,None]+.10*np.sum(((elbows-e)/reach)**2,axis=2)
    anatomical_valid=np.ones(cost.shape,dtype=bool)
    wrist_compatible=None
    if front_plane is not None:
        origin,forward=front_plane
        # Visible limbs should not choose the mirror solution behind the torso.
        # This is a joint-direction prior; it does not infer surface contact.
        behind=np.maximum(0,-((elbows-np.asarray(origin))@np.asarray(forward))-.025)
        # A visible palm needs a modest forward pose prior: the shoulder plane
        # passes through the torso, so merely excluding its back half can still
        # bury a hand in the chest. Scale the prior with arm reach.
        wrist_behind=np.maximum(0,reach*.22-((goals-np.asarray(origin))@np.asarray(forward)))
        cost+=1000*((behind/reach)**2+(wrist_behind/reach)**2)
        front_valid=(behind<1e-7)&(wrist_behind<1e-7)
        if front_valid.any():anatomical_valid&=front_valid
    if hand_axis is not None:
        forward=np.asarray(hand_axis,dtype=float);forward/=max(1e-8,np.linalg.norm(forward))
        wrist_dot=lower_dirs@forward
        wrist_excess=np.maximum(0,math.cos(math.radians(70))-wrist_dot)
        cost+=1000*wrist_excess**2
        wrist_compatible=wrist_excess<1e-7
    if palm is not None:
        offsets,target,weight=palm
        points=candidates[:,None,:]+np.asarray(offsets)[None,:,:]
        cp=np.c_[points.reshape(-1,3),np.ones(points.shape[0]*points.shape[1])]@transform.T
        image=projected(cp[:,:3],camera).reshape(points.shape[0],points.shape[1],2)
        scale=max(.02,float(np.linalg.norm(target-target.mean(0),axis=1).mean()))
        image_error=np.mean(np.sum((image-target[None,:,:])**2,axis=2),axis=1)/scale**2
        cost+=6*weight*image_error[:,None]
    if evidence is not None:
        observed,confidence,extent=evidence
        cost+=4*confidence*(1-np.clip(directions@observed,-1,1))
        # A visible section cannot terminate above its own observed lower end.
        cost+=confidence*4*np.maximum(0,extent-lengths)**2
        # Temporal smoothness and a bad capture prior cannot select an elbow
        # on the opposite side of a visibly directed palm/forearm.
        compatible=anatomical_valid&((directions@observed)>math.cos(math.radians(55)))
        if confidence>=.25 and compatible.any():anatomical_valid=compatible
    else:
        original=np.array(bend);original/=max(1e-8,np.linalg.norm(original))
        cost+=.20*(1-np.clip(bends@original,-1,1))
    if wrist_compatible is not None:
        compatible=anatomical_valid&wrist_compatible
        if compatible.any():anatomical_valid=compatible
    consistent=None
    if previous is not None:
        pa,pe,pw,pn=previous
        cost+=.10*np.sum(((elbows-pe)/reach)**2,axis=2)+.08*np.sum(((goals-pw)/reach)**2,axis=2)
        cap=math.cos(math.radians(24))
        previous_upper=(pe-pa)/upper;previous_lower=(pw-pe)/lower
        for dot in (upper_dirs@previous_upper,lower_dirs@previous_lower,normals@pn):
            cost+=15*np.maximum(0,cap-dot)**2
        previous_upper/=np.linalg.norm(previous_upper);previous_lower/=np.linalg.norm(previous_lower)
        previous_z=np.cross(pn,previous_upper);previous_lower_z=np.cross(pn,previous_lower)
        upper_trace=normals@pn+upper_dirs@previous_upper+np.cross(normals,upper_dirs)@previous_z
        lower_trace=normals@pn+lower_dirs@previous_lower+np.cross(normals,lower_dirs)@previous_lower_z
        trace_min=1+2*math.cos(math.radians(max_step_degrees))
        consistent=(upper_trace>=trace_min)&(lower_trace>=trace_min)
        consistent&=anatomical_valid
        if consistent.any():
            # A hard branch filter can preserve a wrong initial arm and delay a
            # real raise for many frames. Keep it only if image direction agrees.
            image_good=evidence is None or float(np.min(1-np.clip(directions[consistent]@evidence[0],-1,1)))<.20
            if image_good:cost=np.where(consistent,cost,np.inf)
            else:consistent=None
    cost=np.where(anatomical_valid,cost,np.inf)
    index=np.unravel_index(np.argmin(cost),cost.shape)
    goal=Vector(goals[index]);selected=Vector(bends[index]);chosen_elbow=elbows[index]
    limited=previous is not None and consistent is not None and not consistent.any()
    if limited:
        # A missing/cropped detector can request an impossible instantaneous
        # switch. Interpolate the complete hinge pose, retaining both lengths.
        old_upper=frame_rotation(previous_upper,pn)
        new_upper=frame_rotation(upper_dirs[index],normals[index])
        difference=2*math.acos(min(1,abs(old_upper.dot(new_upper))))
        step=math.radians(max_step_degrees)*.5
        q=old_upper.slerp(new_upper,min(1,step/max(1e-8,difference)))
        old_flex=math.acos(np.clip(previous_upper@previous_lower,-1,1))
        new_flex=math.acos(np.clip(upper_dirs[index]@lower_dirs[index],-1,1))
        flex=old_flex+np.clip(new_flex-old_flex,-step,step)
        u=q@Vector((0,1,0));n=q@Vector((1,0,0));v=Quaternion(n,float(flex))@u
        chosen_elbow=a+np.array(u)*upper;goal=Vector(chosen_elbow+np.array(v)*lower)
        axis=(goal-Vector(a)).normalized();selected=Vector(chosen_elbow-a)
        selected-=axis*selected.dot(axis);selected.normalize()
        normal=np.array(n)
    else:normal=normals[index]
    actual_vector=np.array(project(Vector(chosen_elbow),camera))-np.array(project(goal,camera))
    actual_vector/=max(1e-8,np.linalg.norm(actual_vector))
    direction_error=None if evidence is None else math.degrees(math.acos(np.clip(actual_vector@evidence[0],-1,1)))
    return goal,selected,{'arm_fit':'image-forearm-chain-v1','forearm_direction_error_degrees':direction_error,
                          'wrist_depth':float(-(camera[0]@goal).z),'elbow_position':chosen_elbow.tolist(),
                          'normal':normal.tolist(),'limited_by_continuity':bool(limited),'image_ray_limited':bool(ray_limited)}


def torso_front_plane(rig):
    chest=rig.pose.bones.get('UpperBody2') or rig.pose.bones.get('UpperBody1')
    delta=(chest.matrix.to_quaternion()@chest.bone.matrix_local.to_quaternion().inverted()) if chest else Quaternion()
    shoulders=(rig.pose.bones['Arm_L'].head+rig.pose.bones['Arm_R'].head)*.5
    return np.array(shoulders),np.array(delta@Vector((0,-1,0)))


def front_safe_positions(a,e,w,plane):
    """Keep both fixed-length links in front of their shoulder's torso plane.

    An unseen elbow has no evidence for the equivalent behind-body IK branch.
    Project directions onto the allowed half-space, preserving both lengths.
    This final geometric check also applies after temporal interpolation.
    """
    origin,normal=map(Vector,plane);normal.normalize()
    a,e,w=map(Vector,(a,e,w));upper=(e-a).length;lower=(w-e).length
    minimum=(a-origin).dot(normal)
    def link(start,goal,length):
        floor=minimum-(start-origin).dot(normal)
        direction=goal-start
        if direction.dot(normal)>=floor-1e-8:return goal.copy()
        lateral=direction-normal*direction.dot(normal)
        if lateral.length<1e-8:lateral=normal.orthogonal()
        floor=float(np.clip(floor,-length*.999,length*.999))
        return start+normal*floor+lateral.normalized()*math.sqrt(max(0.,length*length-floor*floor))
    safe_e=link(a,e,upper)
    safe_w=link(safe_e,safe_e+(w-e),lower)
    return safe_e,safe_w


def bounded_wrist_rotation(rotation,forearm,limit_degrees=75):
    """Final geometric guard, including transitions and low quality detections."""
    forward=rotation@Vector((0,1,0));axis=Vector(forearm).normalized()
    angle=forward.angle(axis);limit=math.radians(limit_degrees)
    if angle>limit:
        turn=forward.rotation_difference(axis)
        rotation=Quaternion().slerp(turn,(angle-limit)/angle)@rotation
    return rotation


def continuous_wrist_rotation(rotation,forearm,previous,step_radians,limit_degrees=75):
    """Intersect wrist anatomy and speed limits as the forearm itself moves."""
    q=bounded_wrist_rotation(rotation,forearm,limit_degrees)
    for _ in range(32):
        angle=2*math.acos(min(1.,abs(previous.dot(q))))
        if angle<=step_radians+1e-6:break
        q=previous.slerp(q,step_radians/angle)
        q=bounded_wrist_rotation(q,forearm,limit_degrees)
    return q


def anatomical_rotations(arm,elbow,wrist,a,e,w,hand_rotation,set_rotation,view,previous=None):
    """Give both links the same hinge plane; move wrist roll into its twist bone."""
    controller=wrist.parent
    if controller and controller.name.startswith('HandTwist_'):
        controller.rotation_quaternion=Quaternion();view.update()
    normal=(e-a).cross(w-e).normalized()
    set_rotation(arm,frame_rotation(e-a,normal));view.update()
    set_rotation(elbow,frame_rotation(w-e,normal));view.update()
    hand_rotation=bounded_wrist_rotation(hand_rotation,w-e)
    set_rotation(wrist,hand_rotation);view.update()
    # Wrist swing and forearm roll are distinct DOFs. The dedicated MMD twist
    # controller distributes roll through its existing weighted helper bones.
    q=wrist.rotation_quaternion.normalized()
    # Solver stages may choose either quaternion sign. Canonicalize before
    # taking a fractional twist: 0.75 * (angle + 2*pi) is a different physical
    # forearm roll, although the original wrist quaternion is equivalent.
    if q.w<0:q.negate()
    swing,twist=q.to_swing_twist('Y')
    if previous is not None:
        if 'hand_twist' in previous:twist+=math.tau*round((previous['hand_twist']-twist)/math.tau)
        previous['hand_twist']=twist
    # The source hand fit is expressed in the model's actual palm frame. Do not
    # clamp this raw local quaternion as though MMD wrist axes were joint angles.
    wrist.rotation_quaternion=swing@Quaternion((0,1,0),twist);view.update()
    limited_rotation=wrist.matrix.to_quaternion()
    if controller and controller.name.startswith('HandTwist_'):
        controller.rotation_quaternion=Quaternion((0,1,0),float(np.clip(twist*.75,-math.radians(150),math.radians(150))))
        view.update()
        set_rotation(wrist,limited_rotation)
    view.update()

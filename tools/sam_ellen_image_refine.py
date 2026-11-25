"""Final image fitting with fixed limb lengths and anatomical wrist geometry."""
import math
import numpy as np
from mathutils import Vector,Matrix,Quaternion


def turn(a,b):
    a=a/max(1e-10,np.linalg.norm(a));b=b/max(1e-10,np.linalg.norm(b))
    axis=np.cross(a,b);angle=math.atan2(np.linalg.norm(axis),np.clip(a@b,-1,1))
    if np.linalg.norm(axis)<1e-8:
        if angle<1e-6:return np.eye(3)
        axis=np.cross(a,np.eye(3)[np.argmin(abs(a))])
    from sam_ellen_frame_body import rotation
    return rotation(axis/np.linalg.norm(axis)*angle)


def fit_chain(a,e,w,hand_matrix,offsets,target,weights,camera,evidence=None,previous=None,outline=None,hand_step=None,hand_limit=None,wrist_limit=None,front_plane=None,close_range=False,wrist_target=None,orientation_prior=.045,frame_guard=False,pose_prior=None):
    from sam_ellen_frame_body import rotation
    from sam_ellen_screen import project,unproject
    from sam_ellen_screen_fit import projected
    a,e,w=np.array(a),np.array(e),np.array(w)
    upper=np.linalg.norm(e-a);lower=np.linalg.norm(w-e)
    u=(e-a)/upper;v=(w-e)/lower;normal=np.cross(u,v);normal/=max(1e-8,np.linalg.norm(normal))
    base=np.stack([normal,u,np.cross(normal,u)],axis=1)
    flex=math.acos(np.clip(u@v,-1,1));hand_matrix=np.asarray(hand_matrix)
    minimum_flex=math.radians(15)
    if pose_prior is None:
        maximum_flex=math.radians(145)
    else:
        from sam_ellen_arm_policy import elbow_floor
        maximum_flex=math.radians(180-elbow_floor(a,pose_prior[0],pose_prior[1],evidence))
    transform=np.asarray(camera[0]);weights=np.asarray(weights,float);weights/=max(1e-8,weights.sum())
    center=np.sum(target*weights[:,None],axis=0)
    size=max(.025,float(np.sqrt(np.sum(np.sum((target-center)**2,axis=1)*weights))))
    if outline is not None:
        from sam_ellen_punch_mesh import hull,clip_polygon,supports
        outline=clip_polygon(hull(outline),camera[2])
        target_support=supports(outline)
        def polygon_area(p):
            return max(1e-8,float(abs(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1)))*.5))
        target_area=polygon_area(outline)
        center=(outline.min(0)+outline.max(0))*.5
        size=max(.025,float(np.linalg.norm(np.ptp(outline,axis=0))*.5))
    reach=upper+lower
    def geometry(x):
        frame=rotation(x[:3])@base;n=frame[:,0];u=frame[:,1]
        f=np.clip(flex+x[3],minimum_flex,maximum_flex)
        v=rotation(n*f)@u;elbow=a+upper*u;wrist=elbow+lower*v
        hand_delta=x[4:]
        if hand_limit is not None:
            hand_delta=hand_delta*min(1.,hand_limit/max(1e-8,np.linalg.norm(hand_delta)))
        hand=rotation(hand_delta)@hand_matrix
        angle=math.acos(np.clip(hand[:,1]@v,-1,1))
        maximum=wrist_limit if wrist_limit is not None else math.radians(74.5)
        if hand_limit is None and angle>maximum:
            axis=np.cross(hand[:,1],v)
            if np.linalg.norm(axis)<1e-8:axis=hand[:,0]
            hand=rotation(axis/np.linalg.norm(axis)*(angle-maximum))@hand
        if wrist_limit is None and hand_limit is None and hand_step is not None and previous is not None:
            from sam_ellen_arm_chain import continuous_wrist_rotation
            hand=np.array(continuous_wrist_rotation(Matrix(hand.tolist()).to_quaternion(),Vector(v),
                 Matrix(np.asarray(previous[2]).tolist()).to_quaternion(),hand_step).to_matrix())
        points=wrist+offsets@hand.T
        cp=np.c_[points,np.ones(len(points))]@transform.T
        image=projected(cp[:,:3],camera)
        return elbow,wrist,hand,image,cp
    def residual(x):
        elbow,wrist,hand,image,cp=geometry(x)
        actual_center=np.sum(image*weights[:,None],axis=0)
        if outline is None:
            r=list(((image-target)*np.sqrt(weights[:,None])/size).ravel())
            r.extend(((actual_center-center)/size*2).tolist())
            extent=np.sqrt(np.sum(np.sum((image-actual_center)**2,axis=1)*weights))
            r.append(.65*math.log(max(1e-6,extent)/size))
        else:
            polygon=clip_polygon(hull(image),camera[2])
            if len(polygon)<3:polygon=image
            r=list((supports(polygon)-target_support)/size*2)
            r.extend((((polygon.min(0)+polygon.max(0))*.5-center)/size*4).tolist())
            r.append(2*math.log(polygon_area(polygon)/target_area))
        r.extend((x[:4]*.012).tolist());r.extend((x[4:]*orientation_prior).tolist())
        if frame_guard and outline is None:
            # Only constrain joints actually observed within the crop. A source
            # hand crossing the border must remain allowed to cross that border.
            margin=np.minimum(.004,np.minimum.reduce([target[:,0],camera[2]-target[:,0],target[:,1],1-target[:,1]]))
            usable=(margin>=0)&(weights>0)
            excess=np.maximum(0.,np.stack([margin-image[:,0],image[:,0]-(camera[2]-margin),
                                          margin-image[:,1],image[:,1]-(1-margin)],axis=1))
            r.extend((excess[usable]/size*6).ravel().tolist())
        if wrist_limit is not None:
            # The fist and forearm are solved together; this is inside the
            # silhouette objective, not a wrist-only rotation after fitting.
            r.append(.35*math.acos(np.clip(hand[:,1]@((wrist-elbow)/lower),-1,1)))
        if wrist_target is not None:
            r.extend(((np.array(project(Vector(wrist),camera))-wrist_target)/size*3).tolist())
        if front_plane is not None:
            origin,forward=map(np.asarray,front_plane)
            floor=float((a-origin)@forward)
            r.extend((40*np.maximum(0.,floor-(np.array([elbow,wrist])-origin)@forward)/reach).tolist())
        if hand_limit is not None:
            forearm=(wrist-elbow)/lower
            wrist_angle=math.acos(np.clip(hand[:,1]@forearm,-1,1))
            # Keep the captured fist facing intact. Make the arm choose a
            # compatible elbow plane instead of rolling the fist to pass a
            # wrist clamp after the silhouette fit.
            r.append(80*max(0.,wrist_angle-(wrist_limit if wrist_limit is not None else math.radians(74.3))))
        near=.025 if close_range else .10
        r.extend((np.maximum(0,cp[:,2]+near)/reach*20).tolist())
        if evidence is not None:
            delta=np.array(project(Vector(elbow),camera))-np.array(project(Vector(wrist),camera))
            delta/=max(1e-8,np.linalg.norm(delta))
            r.extend(((delta-evidence[0])*.35*evidence[1]).tolist())
        if previous is not None:
            pe,pw,ph=previous
            r.extend(((wrist-pw)/reach*.12).tolist())
            r.extend(((hand-ph)*.20).ravel().tolist())
            pu=(pe-a)/max(1e-8,np.linalg.norm(pe-a));pv=(pw-pe)/max(1e-8,np.linalg.norm(pw-pe))
            pn=np.cross(pu,pv);pn/=max(1e-8,np.linalg.norm(pn))
            nu=(elbow-a)/upper;nv=(wrist-elbow)/lower;nn=np.cross(nu,nv);nn/=max(1e-8,np.linalg.norm(nn))
            r.extend(((nn-pn)*.25).tolist())
            for prior,current in ((pu,nu),(pv,nv),(pn,nn)):
                r.append(12*max(0.,math.cos(math.radians(25))-prior@current))
        if pose_prior is not None:
            # A hand image cannot justify an arbitrary elbow depth/hinge plane.
            # Retain a coherent pre-image-fit chain when repairing a folded fit.
            pe,pw,strength=pose_prior
            ru=(np.asarray(pe)-a)/upper;rv=(np.asarray(pw)-np.asarray(pe))/lower
            rn=np.cross(ru,rv);rn/=max(1e-8,np.linalg.norm(rn))
            nu=(elbow-a)/upper;nv=(wrist-elbow)/lower;nn=np.cross(nu,nv);nn/=max(1e-8,np.linalg.norm(nn))
            r.extend(((nu-ru)*strength).tolist())
            r.extend(((nv-rv)*strength).tolist())
            r.extend(((nn-rn)*strength).tolist())
            wrist_angle=math.acos(np.clip(hand[:,1]@nv,-1,1))
            r.append(2*max(0.,wrist_angle-math.radians(25)))
        return np.array(r)
    def parameter(elbow,wrist):
        u=(elbow-a)/upper;v=(wrist-elbow)/lower;n=np.cross(u,v);n/=max(1e-8,np.linalg.norm(n))
        frame=np.stack([n,u,np.cross(n,u)],axis=1)
        q=Matrix((frame@base.T).tolist()).to_quaternion()
        if q.w<0:q.negate()
        axis,angle=q.to_axis_angle()
        return np.r_[np.array(axis)*angle,math.acos(np.clip(u@v,-1,1))-flex,[0.,0.,0.]]
    initial=np.zeros(7);seeds=[initial]
    if previous is not None and frame_guard:
        prior=parameter(np.asarray(previous[0]),np.asarray(previous[1]))
        delta=Matrix((np.asarray(previous[2])@hand_matrix.T).tolist()).to_quaternion()
        if delta.w<0:delta.negate()
        prior[4:]=np.array(delta.axis)*delta.angle
        seeds.append(prior)
    # A wrong near-camera solution must not be the only starting basin.
    current=geometry(initial);actual_center=np.sum(current[3]*weights[:,None],axis=0)
    span=np.sqrt(np.sum(np.sum((current[3]-actual_center)**2,axis=1)*weights))
    depth=-(camera[0]@Vector(w)).z
    expected=depth*span/size
    for z in np.unique(np.clip([expected*.8,expected,expected*1.2,depth],.14,1.6)):
        goal=np.array(unproject(Vector(center),float(z),camera));axis=goal-a
        distance=np.linalg.norm(axis);axis/=max(1e-8,distance)
        d=np.clip(distance,math.sqrt(upper**2+lower**2-2*upper*lower*math.cos(math.radians(35)))+1e-5,
                  math.sqrt(upper**2+lower**2-2*upper*lower*math.cos(math.radians(165)))-1e-5)
        goal=a+axis*d;along=(d*d+upper*upper-lower*lower)/(2*d)
        radius=math.sqrt(max(0,upper*upper-along*along))
        x=np.cross(axis,np.eye(3)[np.argmin(abs(axis))]);x/=np.linalg.norm(x);y=np.cross(axis,x)
        for angle in np.arange(12)*math.tau/12:
            elbow=a+axis*along+radius*(x*math.cos(angle)+y*math.sin(angle))
            seeds.append(parameter(elbow,goal))
    costs=[float(residual(x)@residual(x)) for x in seeds]
    best=initial;best_cost=costs[0]
    for k in np.argsort(costs)[:(3 if outline is not None else 2)]:
        value=seeds[k].copy();r=residual(value)
        for _ in range(36 if outline is not None else 22):
            jac=np.stack([(residual(value+np.eye(7)[j]*1e-4)-r)/1e-4 for j in range(7)],axis=1)
            step=np.linalg.solve(jac.T@jac+np.eye(7)*.015,-jac.T@r)
            step*=min(1.,.25/max(1e-8,np.linalg.norm(step)))
            improved=False
            for gain in (1.,.5,.2):
                trial=value+gain*step
                trial[3]=np.clip(trial[3],minimum_flex-flex,maximum_flex-flex)
                rr=residual(trial)
                if rr@rr<r@r:value,r,improved=trial,rr,True;break
            if not improved or np.linalg.norm(step)<1e-5:break
        cost=float(r@r)
        if cost<best_cost:best,best_cost=value,cost
    elbow,wrist,hand,image,cp=geometry(best)
    raw=np.linalg.norm(image-target,axis=1)
    error=float(np.sum(raw*weights));ratio=float(np.sqrt(np.sum(np.sum((image-np.sum(image*weights[:,None],axis=0))**2,axis=1)*weights))/size)
    if outline is not None:
        polygon=clip_polygon(hull(image),camera[2])
        if len(polygon)>2:
            error=float(np.mean(abs(supports(polygon)-target_support)))
            ratio=math.sqrt(polygon_area(polygon)/target_area)
    toward=np.asarray(camera[0].inverted().translation)-wrist;toward/=max(1e-8,np.linalg.norm(toward))
    return (Vector(elbow),Vector(wrist),Matrix(hand.tolist()).to_quaternion(),
            {'error_px':error*1920,'size_ratio':ratio,
             'near_depth':float(-cp[:,2].max()),'cost':best_cost,
             'knuckle_camera_degrees':math.degrees(math.acos(np.clip(hand[:,1]@toward,-1,1))),
             'forearm_camera_degrees':math.degrees(math.acos(np.clip(((wrist-elbow)/lower)@toward,-1,1)))})


def refine_images(scratch,rig,channels,data,options,cameras,report,geometry=None,protected=None):
    # Partial-strength/interval corrections retain the existing envelope solver.
    # This final pass owns full-frame capture only.
    if options.get('strength',1.)<.999 or options.get('start',0)>0 or options.get('end',0)>0:return
    from sam_ellen_screen_fit import hand_points,landmark_observations
    from sam_ellen_arm_chain import anatomical_rotations,direction_evidence
    from sam_ellen_retarget import set_global_rotation
    from sam_ellen_screen import project
    from sam_ellen_temporal import trajectory
    view=scratch.view_layers[0];times=np.array([r['time'] for r in data['frames']])
    originals={n:list(rows) for n,rows in channels.items()}
    sides=('L','R') if options['side']=='BOTH' else (options['side'],)
    for side in sides:
        if side==options.get('exclude_side') or not options.get('finger_articulation',True):continue
        obs=landmark_observations(data['screen'],side,space='FRAME')
        evidence=direction_evidence(data,side,options);shifts=np.zeros(len(times));targets={}
        for i,o in enumerate(obs):
            if o is None or o.get('held') and not o.get('interpolated'):continue
            relative=times[i]-times[0]
            if relative<options.get('start',0) or options.get('end',0)>0 and relative>options['end']:continue
            for n,rows in originals.items():rig.pose.bones[n].rotation_quaternion=rows[i][1]
            view.update();cam=cameras[i]
            target=o['points'].copy();target[:,0]*=cam[2]/data['screen']['aspect']
            if options.get('mirror'):target[:,0]=cam[2]-target[:,0]
            targets[i]=target
            face=data['screen']['frames'][i].get('face')
            if geometry and face and 'jaw' in face:
                center=np.array(face['center']);width=face['width'];chin=np.array(face['chin'])
                raw=o['points'];mask=o['mask'];mean=raw[mask].mean(0)
                sign=1 if mean[0]>center[0] else -1
                # Only hands visibly beside the temple. Cross-face gestures and
                # forward punches retain their original image targets.
                proximity=np.clip((3.-np.linalg.norm(mean-center)/width)/.6,0,1)
                lateral=np.clip((np.min(sign*(raw[mask,0]-center[0]))/width-.65)/.25,0,1)
                vertical=np.clip((chin[1]-mean[1])/(width*.4),0,1)
                strength=float(proximity*lateral*vertical)
                if strength>0:
                    head=rig.pose.bones[geometry['bone']].matrix
                    cloud=np.array([project(head@Vector(p),cam) for p in geometry['outline']])
                    jaw=np.array(face['jaw']);source_edge=max(sign*(jaw[:,0]-center[0]))
                    required=[]
                    for j in np.flatnonzero(mask):
                        near=cloud[abs(cloud[:,1]-target[j,1])<.035]
                        if len(near)<3:continue
                        boundary=float(np.max(sign*near[:,0]))
                        gap=max(0.,sign*(raw[j,0]-center[0])-source_edge)
                        required.append(boundary+.004+gap*.5-sign*target[j,0])
                    if required:shifts[i]=sign*max(0.,max(required))*strength
        shifts=trajectory(times,shifts,.10,robust=.03)
        previous=None;last=-100.;hinge={}
        for i,o in enumerate(obs):
            if i not in targets:
                previous=None;hinge={};continue
            for n,rows in originals.items():rig.pose.bones[n].rotation_quaternion=rows[i][1]
            view.update();a,e,w=[rig.pose.bones[n+'_'+side] for n in ('Arm','Elbow','Wrist')]
            points=hand_points(rig,side);q=w.matrix.to_quaternion();inverse=q.inverted()
            offsets=np.array([inverse@(p-w.head) for p in points])
            target=targets[i].copy();target[:,0]+=shifts[i]
            weights=o.get('joint_weights',o['mask']).copy()
            if times[i]-last>.15:previous=None
            ee,ww,qq,row=fit_chain(a.head,e.head,w.head,q.to_matrix(),offsets,target,weights,cameras[i],evidence[i],previous)
            anatomical_rotations(a,e,w,a.head.copy(),ee,ww,qq,set_global_rotation,view,hinge)
            previous=(np.array(e.head),np.array(w.head),np.array(w.matrix.to_quaternion().to_matrix()));last=times[i]
            for b in (a,e,w):channels[b.name][i]=channels[b.name][i][0],tuple(b.rotation_quaternion)
            for b in (rig.pose.bones.get('HandTwist_'+side),rig.pose.bones.get('ArmTwist_'+side)):
                if b and b.name in channels:channels[b.name][i]=channels[b.name][i][0],tuple(b.rotation_quaternion)
            row.update(frame=i+1,side=side,head_gap_shift_px=float(shifts[i]*1920),model='final-image-chain-v1')
            report.append(row)
            if (i+1)%30==0:yield i+1,len(times)
        yield len(times),len(times)


def settle_images(scratch,rig,channels,data,options,cameras,report,geometry=None,protected=None):
    """Plan final entry/exit trajectories without sweeping an arm into the lens.

    Prepare a newly observed pose while the hand is still outside the image.
    The final time filter is symmetric, with short image-ray filtering and a
    longer hinge-plane filter. The final arm solve rechecks anatomy.
    """
    if options.get('strength',1.)<.999 or options.get('start',0)>0 or options.get('end',0)>0:return
    from sam_ellen_arm_visibility import observed_frames
    from sam_ellen_arm_chain import frame_rotation,anatomical_rotations,continuous_wrist_rotation
    from sam_ellen_retarget import set_global_rotation
    from sam_ellen_hand_temporal import smooth_goals
    from sam_ellen_temporal import quaternions
    times=np.array([r['time'] for r in data['frames']]);view=scratch.view_layers[0]
    originals={n:list(rows) for n,rows in channels.items()}
    for side in (('L','R') if options['side']=='BOTH' else (options['side'],)):
        if side==options.get('exclude_side') or not options.get('finger_articulation',True):continue
        active=observed_frames(data,side)
        if protected is not None:active|=np.asarray(protected.get(side,np.zeros(len(times),dtype=bool)))
        if not active.any():continue
        positions=[];rotations=[];bends=[];uppers=[];flexes=[]
        for i in range(len(times)):
            for n,rows in channels.items():rig.pose.bones[n].rotation_quaternion=rows[i][1]
            view.update();a,e,w=[rig.pose.bones[n+'_'+side] for n in ('Arm','Elbow','Wrist')]
            u=(e.head-a.head).normalized();v=(w.head-e.head).normalized();normal=u.cross(v).normalized()
            axis=(w.head-a.head).normalized();bend=e.head-a.head;bend-=axis*bend.dot(axis)
            positions.append(tuple(w.head));rotations.append(tuple(w.matrix.to_quaternion()));bends.append(tuple(bend.normalized()))
            uppers.append(tuple(frame_rotation(u,normal)));flexes.append(u.angle(v))
        p,q,b=smooth_goals(times,positions,rotations,bends,cameras,active,active.astype(float))
        # A normal-vector sign jump is a large arm-roll jump even when the
        # projected wrist hardly moves. Smooth the complete hinge frame.
        upperq=[Quaternion(v) for v in uppers]
        from sam_ellen_temporal import trajectory
        from sam_ellen_hand_temporal import runs
        for lo,hi in runs(active):
            upperq[lo:hi]=quaternions(times[lo:hi],uppers[lo:hi],.085)
            flexes[lo:hi]=trajectory(times[lo:hi],flexes[lo:hi],.055).tolist()
        hinge={};previous=None
        for i in range(len(times)):
            if not active[i]:previous=None;hinge={};continue
            for n,rows in channels.items():rig.pose.bones[n].rotation_quaternion=rows[i][1]
            view.update();a,e,w=[rig.pose.bones[n+'_'+side] for n in ('Arm','Elbow','Wrist')]
            upper=(e.head-a.head).length;lower=(w.head-e.head).length
            # Preserve the filtered wrist ray and depth. The filtered hinge
            # normal chooses the elbow branch around that wrist target.
            goal=Vector(p[i]);axis=goal-a.head;d=axis.length;axis.normalize()
            minimum=math.sqrt(upper**2+lower**2-2*upper*lower*math.cos(math.radians(35)))
            maximum=math.sqrt(upper**2+lower**2-2*upper*lower*math.cos(math.radians(165)))
            d=float(np.clip(d,minimum,maximum));goal=a.head+axis*d
            along=(d*d+upper*upper-lower*lower)/(2*d)
            n=upperq[i]@Vector((1,0,0));bend=axis.cross(n)
            if bend.length<1e-7:bend=Vector(b[i])
            bend.normalize();elbow=a.head+axis*along+bend*math.sqrt(max(0,upper*upper-along*along))
            u=(elbow-a.head).normalized();v=(goal-elbow).normalized();normal=u.cross(v).normalized()
            frame=frame_rotation(u,normal);flex=u.angle(v);hand=Quaternion(q[i])
            if previous is not None:
                prior_frame,prior_flex,prior_hand=previous
                angle=2*math.acos(min(1,abs(frame.dot(prior_frame))))
                dt=max(1e-5,times[i]-times[i-1]);cap=math.radians(720*dt)
                if angle>cap:frame=prior_frame.slerp(frame,cap/angle)
                flex=prior_flex+float(np.clip(flex-prior_flex,-math.radians(450*dt),math.radians(450*dt)))
                u=frame@Vector((0,1,0));normal=frame@Vector((1,0,0));v=Quaternion(normal,flex)@u
                elbow=a.head+u*upper;goal=elbow+v*lower
                hand=continuous_wrist_rotation(hand,v,prior_hand,math.radians(900*dt))
            anatomical_rotations(a,e,w,a.head.copy(),elbow,goal,hand,set_global_rotation,view,hinge)
            previous=(frame.copy(),flex,w.matrix.to_quaternion().copy())
            for n in ('Arm_','Elbow_','Wrist_','HandTwist_','ArmTwist_'):
                bone=rig.pose.bones.get(n+side)
                if bone and bone.name in channels:channels[bone.name][i]=channels[bone.name][i][0],tuple(bone.rotation_quaternion)
        report.append({'side':side,'model':'observed-image-hinge-v2','supported_frames':int(active.sum())})

"""Camera-space palm evidence with image-constrained wrist rotation."""
import math
import numpy as np
from mathutils import Matrix,Quaternion,Vector


def palm_basis(points):
    p=np.asarray(points,dtype=float)
    if p.shape!=(21,3) or not np.isfinite(p).all():return None
    along=p[[5,9,13,17]].mean(0)-p[0]
    across=p[5]-p[17]
    if np.linalg.norm(along)<1e-5 or np.linalg.norm(across)<1e-5:return None
    along/=np.linalg.norm(along)
    across-=along*np.dot(across,along)
    if np.linalg.norm(across)<1e-5:return None
    across/=np.linalg.norm(across)
    return np.stack([across,along,np.cross(across,along)],axis=1)


def fit_rotation(local,wrist,original,world,target,mask,camera,forearm,limit,mirror=False,ambiguous=False,return_candidates=False,normal_hint=None):
    """Fit image shape and observed palm normal; do not move the arm chain."""
    from sam_ellen_screen_fit import projected,rotation_vector
    from sam_ellen_arm_chain import bounded_wrist_rotation
    basis=palm_basis(local);observed=palm_basis(np.asarray(world)*[-1. if mirror else 1.,-1.,-1.])
    if basis is None or observed is None:return (original.copy(),{},[]) if return_candidates else (original.copy(),{})
    target=np.asarray(target);weights=np.asarray(mask,float).copy()
    weights*=np.array([1,.5,.5,.5,.5,1,.6,.6,1,1,.6,.6,1,1,.6,.6,1,1,.6,.6,1])
    weights/=max(1e-8,weights.sum())
    center=np.sum(target*weights[:,None],axis=0)
    shape=target-center;size=max(.02,float(np.sqrt(np.sum(np.sum(shape**2,axis=1)*weights))))
    transform=camera[0].to_3x3();origin=np.asarray(camera[0]@wrist)
    current=np.asarray(transform@original.to_matrix());measured=observed@basis.T
    def evaluate(delta,base):
        q=camera[0].to_quaternion().inverted()@Matrix((rotation_vector(delta)@base).tolist()).to_quaternion()
        q=bounded_wrist_rotation(q,forearm,limit)
        rotation=np.asarray(transform@q.to_matrix())
        image=projected(np.asarray(local)@rotation.T+origin,camera)
        c=np.sum(image*weights[:,None],axis=0);p=image-c
        span=max(1e-6,float(np.sqrt(np.sum(np.sum(p**2,axis=1)*weights))))
        # Palm depth may be uncertain; direction and labelled finger layout
        # must still match the source crop, independently of translation.
        residual=list(((p/span-shape/size)*np.sqrt(weights[:,None])).ravel())
        # Matched landmarks do not resolve monocular depth reflection. Even
        # sustained MediaPipe normals can be inverted while 2-D stays correct.
        normal=rotation@basis[:,2]
        residual.extend(((normal-observed[:,2])*(.12 if ambiguous else .75)).tolist())
        # A curled profile hand has almost no 2D evidence for axial roll.
        # Retain the captured labelled palm axes as a weak full-frame prior.
        residual.extend(((rotation@basis[:,:2]-observed[:,:2])*.25).ravel().tolist())
        if normal_hint is not None:
            direction,weight=normal_hint
            residual.extend(((normal-direction)*2.5*weight).tolist())
        residual.append(.15*math.log(span/size))
        return np.asarray(residual),q,image,normal
    best=None;candidates=[]
    seeds=[measured,current]
    if ambiguous:
        # Monocular landmark depth can choose the wrong palm/back solution.
        # Explicitly test the reflected-depth hypothesis against labelled 2D
        # fingers, rather than locking the arm to that uncertain normal.
        reflected=palm_basis(np.asarray(world)*[-1. if mirror else 1.,-1.,1.])
        if reflected is not None:seeds.append(reflected@basis.T)
        seeds.extend([base@np.diag([-1.,1.,-1.]) for base in (measured,current)])
    for base in seeds:
        value=np.zeros(3);r,seed_q,*_=evaluate(value,base)
        # Optimizing every seed can collapse all hypotheses into the same
        # bad image minimum. Keep the measured/native pose available to the
        # temporal path as well, scored by the identical image objective.
        if not any(abs(seed_q.dot(old[1]))>.9995 for old in candidates):
            candidates.append((float(r@r),seed_q.copy()))
        for _ in range(28):
            jac=np.stack([(evaluate(value+np.eye(3)[j]*1e-4,base)[0]-r)/1e-4 for j in range(3)],axis=1)
            step=np.linalg.solve(jac.T@jac+np.eye(3)*.004,-jac.T@r)
            step*=min(1.,.3/max(1e-8,np.linalg.norm(step)))
            improved=False
            for gain in (1.,.5,.2):
                trial=value+gain*step;rr,*_=evaluate(trial,base)
                if rr@rr<r@r:value,r,improved=trial,rr,True;break
            if not improved or np.linalg.norm(step)<1e-5:break
        rr,q,image,normal=evaluate(value,base);cost=float(rr@rr)
        if not any(abs(q.dot(previous[1]))>.9995 for previous in candidates):candidates.append((cost,q.copy()))
        if best is None or cost<best[0]:best=cost,q,image,normal
    cost,q,image,normal=best
    details={'cost':cost,'normal_error_degrees':math.degrees(math.acos(float(np.clip(normal@observed[:,2],-1,1)))),
              'image_error':float(np.sum(np.linalg.norm(image-target,axis=1)*weights))}
    return (q,details,candidates) if return_candidates else (q,details)


def continuous_candidates(times,candidates,fast=None):
    """Select the palm/back hypotheses over each observed run, not per frame."""
    times=np.asarray(times);fast=np.zeros(len(times)) if fast is None else np.asarray(fast)
    valid=np.array([i for i,items in enumerate(candidates) if items],int);result={}
    if not len(valid):return result
    boundaries=np.r_[0,np.flatnonzero(np.diff(times[valid])>.24)+1,len(valid)]
    for start,end in zip(boundaries[:-1],boundaries[1:]):
        ids=valid[start:end];cost=np.array([c for c,q in candidates[ids[0]]]);parents=[]
        for a,b in zip(ids[:-1],ids[1:]):
            old=np.array([tuple(q) for c,q in candidates[a]]);new=np.array([tuple(q) for c,q in candidates[b]])
            turn=2*np.arccos(np.clip(abs(old@new.T),0,1));dt=times[b]-times[a]
            penalty=.5*(turn/max(.12,dt*5))**2
            score=cost[:,None]+penalty;parent=np.argmin(score,axis=0)
            cost=np.array([c for c,q in candidates[b]])+score[parent,np.arange(len(parent))]
            parents.append(parent)
        k=int(np.argmin(cost));path=[k]
        for parent in reversed(parents):k=int(parent[k]);path.append(k)
        for i,k in zip(ids,reversed(path)):result[int(i)]=candidates[i][k][1].copy()
        for a,b in zip(ids[:-1],ids[1:]):
            for i in range(a+1,b):
                result[i]=result[int(a)].slerp(result[int(b)],float((times[i]-times[a])/(times[b]-times[a])))
    return result


def orientation_observations(data,side):
    """Require matched visible hand evidence, usable axes and temporal support."""
    from sam_ellen_arm_visibility import screen_hand,hand_landmark_mask,observed_frames
    screen=data.get('screen',{});rows=screen.get('hand_pose_depth',{}).get('frames',[])
    n=len(data['frames']);result=[None]*n
    if len(rows)!=n:return result
    visible=observed_frames(data,side)
    for i,row in enumerate(rows):
        d=row.get(side);h=screen_hand(screen,i,side)
        if not visible[i] or not d or not h or d.get('image_match_error',1.)>.025:continue
        mask=hand_landmark_mask(h)
        if not mask[0] or mask[[5,9,13,17]].sum()<3 or mask.sum()<12:continue
        basis=palm_basis(d.get('points',[]))
        if basis is None:continue
        result[i]=d
    times=np.array([r['time'] for r in data['frames']])
    # A lone estimate or a one-frame normal flip is not a rotation anchor.
    raw=list(result)
    for i,d in enumerate(raw):
        if d is None:continue
        neighbors=[j for j in range(max(0,i-2),min(n,i+3)) if j!=i and raw[j] is not None and abs(times[j]-times[i])<=.08]
        if not neighbors:result[i]=None;continue
        normal=palm_basis(d['points'])[:,2]
        if len(neighbors)>=2 and all(normal@palm_basis(raw[j]['points'])[:,2]<.5 for j in neighbors):result[i]=None
    return result


def align_palms(scratch,rig,channels,data,options,cameras,report):
    """Apply continuous palm pose evidence while retaining the existing arm branch."""
    if not options or options.get('space')!='FRAME' or not options.get('landmarks',True) or options.get('strength',1.)<=0:return
    from sam_ellen_arm_policy import activation,excluded,front_circle,elbow_floor,camera_reachable_goal,wrist_limit
    from sam_ellen_arm_visibility import screen_hand,hand_landmark_mask
    from sam_ellen_screen_fit import hand_points,landmark_observations
    from sam_ellen_motion_continuity import translation_goal
    from sam_ellen_screen import project,unproject
    from sam_ellen_arm_chain import anatomical_rotations,torso_front_plane,direction_evidence,bounded_wrist_rotation
    from sam_ellen_image_chain import fit_direction
    from sam_ellen_retarget import set_global_rotation
    from sam_ellen_temporal import quaternions,trajectory
    from sam_punch_evidence import detect
    times=np.array([r['time'] for r in data['frames']]);allowed=activation(times,options);view=scratch.view_layers[0]
    original={n:list(rows) for n,rows in channels.items()};policy=data.get('_sam_arm_policy',[])
    sides=('L','R') if options.get('side')=='BOTH' else (options['side'],)
    for side in sides:
        if excluded(options,side):continue
        observed=orientation_observations(data,side);segments,_=detect(data['screen'],side)
        protected=np.array([any(s['start']-.15<=t-times[0]<=s['end']+.2 for s in segments) for t in times])
        observations=landmark_observations(data['screen'],side,space='FRAME')
        directions=direction_evidence(data,side,options,independent_only=True)
        corrections=[Quaternion() for _ in times];shift=np.zeros((len(times),2));valid=np.zeros(len(times),bool)
        captures=[];stats=[]
        def load(i):
            for n,rows in original.items():rig.pose.bones[n].rotation_quaternion=rows[i][1]
            view.update()
        for i,d in enumerate(observed):
            load(i);a,e,w=[rig.pose.bones[n+'_'+side] for n in ('Arm','Elbow','Wrist')];camera=cameras[i]
            q=w.matrix.to_quaternion();local=np.array([q.inverted()@(p-w.head) for p in hand_points(rig,side)])
            captures.append((a.head.copy(),e.head.copy(),w.head.copy(),q.copy(),local,torso_front_plane(rig)))
            o=observations[i]
            if d is None or o is None or o.get('held') or allowed[i]<=0 or protected[i]:continue
            target=o['points'].copy();target[:,0]*=camera[2]/data['screen']['aspect']
            if options.get('mirror'):target[:,0]=camera[2]-target[:,0]
            floor=elbow_floor(a.head,e.head,w.head)
            candidate,details=fit_rotation(local,w.head.copy(),q,d['points'],target,o['mask'],camera,w.head-e.head,
                wrist_limit(math.degrees((a.head-e.head).angle(w.head-e.head))),options.get('mirror',False))
            if not details:continue
            goal=translation_goal(local,candidate,w.head.copy(),o,camera,data['screen']['aspect'],options.get('mirror',False))
            xy=project(goal,camera);before=project(w.head,camera);delta=xy-before
            delta*=min(1.,.16/max(1e-8,delta.length))
            shift[i]=delta
            camq=camera[0].to_quaternion()
            corrections[i]=(camq@candidate)@(camq@q).inverted()
            valid[i]=True;stats.append(dict(frame=i+1,side=side,**details))
            if i%30==0:yield i+1,len(times)*2
        if not valid.any():continue
        # Fill bounded dropouts only; extended loss releases the correction.
        ids=np.flatnonzero(valid);envelope=np.zeros(len(times))
        for i in range(len(times)):
            if valid[i]:envelope[i]=1.;continue
            k=int(np.searchsorted(ids,i));left=int(ids[k-1]) if k else None;right=int(ids[k]) if k<len(ids) else None
            if left is not None and right is not None and times[right]-times[left]<=.20:
                t=float((times[i]-times[left])/(times[right]-times[left]))
                corrections[i]=corrections[left].slerp(corrections[right],t);shift[i]=shift[left]*(1-t)+shift[right]*t;envelope[i]=1.
            elif left is not None and times[i]-times[left]<.22:
                t=float(np.clip((times[i]-times[left]-.06)/.16,0,1));envelope[i]=1-t*t*(3-2*t)
                corrections[i]=corrections[left].copy();shift[i]=shift[left]
        qfiltered=quaternions(times,corrections,.055,weights=np.where(valid,6.,1.))
        shift=trajectory(times,shift,.055,weights=np.where(valid,6.,1.),robust=.03)
        # Reacquisition eases in using past evidence; it cannot anticipate a
        # hand that has not appeared yet. Confirmed fist intervals stay intact.
        start=None;previous=None;hinge={}
        for i in range(len(times)):
            load(i);a,e,w=[rig.pose.bones[n+'_'+side] for n in ('Arm','Elbow','Wrist')];camera=cameras[i]
            ca,ce,cw,cq,local,plane=captures[i]
            if envelope[i]<=0 or allowed[i]<=0 or protected[i]:start=None;previous=None;continue
            if start is None:start=times[i]
            weight=envelope[i]*allowed[i]*float(np.clip((times[i]-start+1/data['fps'])/.14,0,1))
            weight=weight*weight*(3-2*weight)
            delta=Quaternion().slerp(qfiltered[i],weight);camq=camera[0].to_quaternion()
            hand=camq.inverted()@delta@camq@cq
            xy=project(cw,camera)+Vector(shift[i])*weight;depth=-(camera[0]@cw).z
            goal=unproject(xy,depth,camera);upper=(ce-ca).length;lower=(cw-ce).length
            floor=elbow_floor(ca,ce,cw)
            if len(policy)==len(times) and side in policy[i]:floor=policy[i][side].get('active_minimum_elbow',policy[i][side]['minimum_elbow'])
            goal=camera_reachable_goal(ca,goal,upper,lower,floor,camera,plane)
            axis=(goal-ca).normalized();bend=ce-ca;bend-=axis*bend.dot(axis)
            ee,ww,_=front_circle(ca,goal,upper,lower,bend,plane,floor)
            if directions[i] is not None:
                cue=np.asarray(directions[i][0]).copy();cue[0]*=camera[2]/data['screen']['aspect'];cue/=np.linalg.norm(cue)
                ee,ww,_,_=fit_direction(ca,ee,ww,upper,lower,camera,(cue,*directions[i][1:]),plane,floor,previous,depth_preference=30.)
            hand=bounded_wrist_rotation(hand,ww-ee,wrist_limit(math.degrees((ca-ee).angle(ww-ee))))
            anatomical_rotations(a,e,w,ca,ee,ww,hand,set_global_rotation,view,hinge)
            previous=(np.array(ee),np.array(ww))
            for prefix in ('Arm_','Elbow_','Wrist_','ArmTwist_','HandTwist_'):
                name=prefix+side
                if name in channels:channels[name][i]=channels[name][i][0],tuple(rig.pose.bones[name].rotation_quaternion)
            if i%30==0:yield len(times)+i+1,len(times)*2
        report.append({'model':'observed-palm-normal-and-image-v1','side':side,'observed_frames':int(valid.sum()),'proposals':stats})
        yield len(times)*2,len(times)*2

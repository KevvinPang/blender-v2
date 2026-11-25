"""Near-lens hand scale and bounded torso reach in the final camera frame.

No camera transform, root translation, finger key or bone length is written.
The final whole-arm trajectory owns the resulting hand targets and safety.
"""
import math
import numpy as np
from mathutils import Matrix, Quaternion, Vector


def near_lens_evidence(screen, side):
    """Require persistent source palm/face scale, independent of model size.

    A small projected character hand is not proof of a near-camera action.
    Use a conservative gate: uncertain gestures stay with the normal solver.
    """
    from sam_ellen_arm_visibility import screen_hand, hand_landmark_mask
    frames = screen.get('frames', [])
    supported = np.zeros(len(frames), bool)
    for i, row in enumerate(frames):
        hand, face = screen_hand(screen, i, side), row.get('face')
        if not hand or not face or face.get('width', 0) <= .005: continue
        mask = hand_landmark_mask(hand)
        if not mask[[0, 5, 9, 13, 17]].all(): continue
        points = np.asarray(hand['points'], float)
        span = float(np.linalg.norm(points[5]-points[17]))
        supported[i] = span >= .055 and span/face['width'] >= 1.5
    # A single noisy box or foreshortened frame cannot activate body reach.
    edges = np.diff(np.r_[False, supported, False].astype(int))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b-a < 3: supported[a:b] = False
    return supported


def rigid_target(points, target, mask, camera):
    from sam_ellen_screen_fit import projected, rotation_vector
    source=np.array([camera[0]@p for p in points]);offsets=source-source[0]
    image=projected(source,camera)
    palm=np.array([0,5,9,13,17]);palm=palm[mask[palm]]
    if len(palm)<3:return None
    pairs=[(a,b) for j,a in enumerate(palm) for b in palm[j+1:]]
    native=np.median([np.linalg.norm(image[a]-image[b]) for a,b in pairs])
    observed=np.median([np.linalg.norm(target[a]-target[b]) for a,b in pairs])
    ratio=observed/max(.005,native)
    if ratio<1.35 or observed<.055:return None
    weights=np.full(21,.25);weights[palm]=1.;weights[[4,8,12,16,20]]=1.5
    weights*=mask
    depth=max(.12,min(1.5,-source[0,2]/ratio))
    from sam_ellen_screen import unproject
    origin=np.array(camera[0]@unproject(Vector(target[palm].mean(0)),depth,camera))
    origin-=offsets[palm].mean(0)
    def residual(x):
        pts=offsets@rotation_vector(x[3:]).T+x[:3]
        error=(projected(pts,camera)-target)*np.sqrt(weights[:,None])
        return np.r_[error.ravel(),x[3:]*.006,np.maximum(0,pts[:,2]+.06)*20]
    results=[]
    for seed in (np.zeros(3),np.array([0.,0.,.65]),np.array([0.,0.,-.65])):
        x=np.r_[origin,seed];lam=.001
        for _ in range(45):
            error=residual(x);eps=1e-5
            jac=np.column_stack([(residual(x+np.eye(6)[j]*eps)-error)/eps for j in range(6)])
            step=np.linalg.solve(jac.T@jac+lam*np.eye(6),-jac.T@error)
            step[:3]*=min(1.,.1/max(1e-8,np.linalg.norm(step[:3])))
            step[3:]*=min(1.,.3/max(1e-8,np.linalg.norm(step[3:])))
            trial=x+step
            if np.dot(residual(trial),residual(trial))<np.dot(error,error):
                x=trial;lam=max(1e-7,lam*.6)
            else:lam=min(1e5,lam*5)
            if np.linalg.norm(step)<1e-7:break
        results.append((float(np.dot(residual(x),residual(x))),x))
    score,x=min(results,key=lambda item:item[0])
    if score/max(1.,weights.sum())>.006 or x[2]>-.10:return None
    return Vector(x[:3]),Matrix(rotation_vector(x[3:])).to_quaternion(),ratio,score


def coordinate_close_hands(scratch,rig,channels,data,options,cameras,report):
    """Give the shoulders a bounded, continuous reach before solving arms."""
    if options.get('space')!='FRAME' or not options.get('landmarks') or not options.get('coordinated',True):return
    from sam_ellen_arm_policy import excluded,activation
    from sam_ellen_arm_visibility import screen_hand,hand_landmark_mask,observed_frames
    from sam_ellen_screen_fit import hand_points
    from sam_ellen_retarget import set_global_rotation,gaze_bones
    from sam_ellen_temporal import trajectory,quaternions,bridge_missing
    from sam_ellen_screen_fit import projected
    times=np.array([r['time'] for r in data['frames']]);n=len(times)
    if n<3:return
    spine=[rig.pose.bones.get(name) for name in ('UpperBody1','UpperBody2')]
    if not all(spine):return
    gaze=gaze_bones(rig);headnames=[gaze[k].name for k in ('首','頭')]
    original={name:list(values) for name,values in channels.items()}
    view=scratch.view_layers[0];active=activation(times,options);targets={};corrections=np.zeros((n,2));fit_rows=[]
    def load(i):
        for name,values in original.items():rig.pose.bones[name].rotation_quaternion=values[i][1]
        view.update()
    sides=('L','R') if options.get('side')=='BOTH' else (options['side'],)
    for side in sides:
        if excluded(options,side):continue
        visible=observed_frames(data,side);fits=[None]*n
        near = near_lens_evidence(data['screen'], side)
        for i in np.flatnonzero(visible&near&(active>0)):
            load(i);hand=screen_hand(data['screen'],i,side);mask=hand_landmark_mask(hand)
            target=np.asarray(hand['points'],float).copy();target[:,0]*=cameras[i][2]/data['screen']['aspect']
            if options.get('mirror'):target[:,0]=cameras[i][2]-target[:,0]
            fit=rigid_target(hand_points(rig,side),target,mask,cameras[i])
            if fit is not None:
                p,delta,ratio,score=fit
                palm=cameras[i][0].to_quaternion()@rig.pose.bones['Wrist_'+side].matrix.to_quaternion()
                fits[i]=(p,delta@palm,ratio,score)
            if i%20==0:yield i+1,n*2
        good=np.array([f is not None for f in fits]);ids=np.flatnonzero(good)
        if len(ids)<3:continue
        # Reject isolated close-range fits; short gaps remain interpolated in
        # the same camera frame, not held as a world-space position.
        for i in ids:
            if np.count_nonzero(good[max(0,i-2):min(n,i+3)])<2:good[i]=False
        ids=np.flatnonzero(good)
        if len(ids)<3:continue
        points=np.array([fits[i][0] for i in ids]);rotations=[fits[i][1] for i in ids]
        points=trajectory(times[ids],points,.045)
        rotations=quaternions(times[ids],rotations,.045)
        # A scale threshold must not switch the hand depth or palm frame in
        # one frame. Keep the weight continuous inside each supported run.
        raw_weight=np.zeros(n)
        for i in ids:raw_weight[i]=np.clip((fits[i][2]-1.35)/.65,0.,1.)*active[i]
        for a,b in zip(ids,ids[1:]):
            if times[b]-times[a]<=.16:
                raw_weight[a:b+1]=np.interp(times[a:b+1],times[[a,b]],raw_weight[[a,b]])
        weights=np.clip(trajectory(times,raw_weight,.065),0.,1.)
        rows=[None]*n
        for k,i in enumerate(ids):rows[i]={'wrist':points[k].tolist(),'rotation':list(rotations[k]),'weight':float(weights[i])}
        for a,b in zip(ids,ids[1:]):
            if times[b]-times[a]>.16:continue
            for i in range(a+1,b):
                t=float((times[i]-times[a])/(times[b]-times[a]))
                rows[i]={'wrist':((1-t)*points[np.searchsorted(ids,a)]+t*points[np.searchsorted(ids,b)]).tolist(),
                         'rotation':list(rotations[np.searchsorted(ids,a)].slerp(rotations[np.searchsorted(ids,b)],t)),
                         'weight':float(weights[i])}
        targets[side]=rows
        fit_rows.extend({'frame':int(i+1),'side':side,'ratio':fits[i][2],'score':fits[i][3]} for i in ids)
    if not targets:return
    if options.get('unified_arms'):
        # Fixed capture has already solved the subject displacement. A later
        # hand-size proposal must not silently bend the torso and invalidate
        # that framing. Reach and palm depth belong to the coupled arm solver.
        data['_sam_close_targets']=targets
        report.append({'model':'near-hand-proposals-v2','body_modified':False,'fits':fit_rows})
        return
    # Optimize a shared chest pitch/yaw from all close hands. Hands must not
    # independently move the body in opposite directions.
    observed=np.zeros(n,bool)
    bound=math.radians(min(35.,max(15.,options.get('body_limit',20.)*1.6)))
    previous=np.zeros(2)
    for i in range(n):
        goals=[(side,cameras[i][0].inverted()@Vector(rows[i]['wrist']),rows[i]['weight'])
               for side,rows in targets.items() if rows[i] and rows[i]['weight']>.001]
        if not goals:continue
        load(i);axis=cameras[i][0].to_quaternion().inverted()
        pivots=[b.head.copy() for b in spine];base={b.name:b.matrix.to_quaternion().copy() for b in spine}
        shoulders={side:rig.pose.bones['Arm_'+side].head.copy() for side,_,_ in goals}
        lengths={side:sum((rig.pose.bones[b+'_'+side].head-rig.pose.bones[a+'_'+side].head).length
                         for a,b in (('Arm','Elbow'),('Elbow','Wrist'))) for side,_,_ in goals}
        # Retain the existing face framing while gaining arm reach. Its head
        # orientation is compensated below, so only the head pivot moves.
        head=rig.pose.bones[headnames[-1]]
        eyes=[gaze[k].head.copy() for k in ('左目','右目')]
        eye_offsets=[p-head.head for p in eyes]
        face_before=projected(np.array([cameras[i][0]@p for p in eyes]),cameras[i])
        center_before=face_before.mean(0)
        width_before=max(.005,float(np.linalg.norm(face_before[0]-face_before[1])))
        def moved(point,value):
            p=point.copy();cumulative=Quaternion()
            # Two equal increments, pivoting the upper segment with the lower.
            for pivot in reversed(pivots):
                step=Quaternion(axis@Vector((1,0,0)),float(value[0]/2))@Quaternion(axis@Vector((0,1,0)),float(value[1]/2))
                p=pivot+step@(p-pivot)
            return p
        def cost(value):
            r=[value[0]*.07,value[1]*.10,*(value-previous)*.06]
            for side,goal,weight in goals:
                delta=(moved(shoulders[side],value)-goal).length
                r.append(8*weight*max(0.,delta-lengths[side]*.96)/lengths[side])
            pivot=moved(head.head,value)
            face=projected(np.array([cameras[i][0]@(pivot+offset) for offset in eye_offsets]),cameras[i])
            center_shift=np.linalg.norm(face.mean(0)-center_before)
            scale=float(np.linalg.norm(face[0]-face[1]))/width_before
            r.extend((30*max(0.,center_shift-.025),8*max(0.,abs(math.log(max(.01,scale)))-.08)))
            return np.asarray(r)
        value=previous.copy()
        for _ in range(25):
            r=cost(value);eps=1e-4;jac=np.column_stack([(cost(value+np.eye(2)[j]*eps)-r)/eps for j in range(2)])
            step=np.linalg.solve(jac.T@jac+np.eye(2)*.001,-jac.T@r)
            step*=min(1.,.15/max(1e-8,np.linalg.norm(step)))
            trial=value+step;trial[0]=np.clip(trial[0],0,bound);trial[1]=np.clip(trial[1],-bound*.3,bound*.3)
            if cost(trial)@cost(trial)>=r@r:break
            value=trial
        corrections[i]=value*active[i];previous=value;observed[i]=True
        yield n+i+1,n*2
    corrections=bridge_missing(times,corrections,observed,max_gap=.18,hold=.04,fade=.28)
    corrections=trajectory(times,corrections,.075,np.where(observed,12.,1.))
    # Limit influence to the supported gesture and its smooth transition.
    distance=np.min(abs(times[:,None]-times[np.flatnonzero(observed)][None,:]),axis=1)
    envelope=np.clip(1-distance/.30,0,1);envelope=envelope*envelope*(3-2*envelope)
    corrections*=envelope[:,None];corrections[:,0]=np.clip(corrections[:,0],0,bound)
    changed=[b.name for b in spine]+headnames
    for i,value in enumerate(corrections):
        if np.linalg.norm(value)<1e-6:continue
        load(i);global_head={name:rig.pose.bones[name].matrix.to_quaternion().copy() for name in headnames}
        axes=cameras[i][0].to_quaternion().inverted()
        for b in spine:
            q=Quaternion(axes@Vector((1,0,0)),float(value[0]/2))@Quaternion(axes@Vector((0,1,0)),float(value[1]/2))
            set_global_rotation(b,q@b.matrix.to_quaternion());view.update()
        for name in headnames:set_global_rotation(rig.pose.bones[name],global_head[name]);view.update()
        for name in changed:channels[name][i]=(channels[name][i][0],tuple(rig.pose.bones[name].rotation_quaternion))
    data['_sam_close_targets']=targets
    report.append({'model':'coupled-close-hand-reach-v1','fits':fit_rows,
                   'maximum_body_lean_degrees':math.degrees(float(corrections[:,0].max())),
                   'active_frames':(np.flatnonzero(observed)+1).tolist(),'camera_modified':False})

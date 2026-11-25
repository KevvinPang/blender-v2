"""Image-fit proposals constrained by the pre-image arm pose."""
import math
import numpy as np
from mathutils import Matrix,Vector,Quaternion


def close_projection(scratch,rig,source,channels,data,options,cameras,report,protected=None,reference_channels=None):
    if not options or options.get('space')!='FRAME' or options.get('strength',1.)<=0:return
    from sam_ellen_arm_policy import excluded,PREFIXES
    from sam_ellen_screen_fit import landmark_observations,hand_points
    from sam_ellen_arm_chain import anatomical_rotations,torso_front_plane,direction_evidence,frame_rotation,continuous_wrist_rotation,front_safe_positions
    from sam_ellen_retarget import set_global_rotation
    from sam_ellen_image_refine import fit_chain
    from sam_ellen_punch_mesh import hand_geometry,posed_hand,image_shape
    from sam_punch_evidence import detect
    from sam_ellen_punch_pose import measured_rotation,hand_boundaries
    from sam_ellen_fist_final import bridge_silhouettes
    from sam_ellen_temporal import quaternions
    from sam_ellen_screen import project
    view=scratch.view_layers[0]
    times=np.array([r['time'] for r in data['frames']]);relative=times-times[0]
    original={n:list(rows) for n,rows in channels.items()}
    depth=data['screen'].get('hand_pose_depth',{}).get('frames',[])
    for side in (('L','R') if options.get('side')=='BOTH' else (options['side'],)):
        if excluded(options,side):continue
        obs=landmark_observations(data['screen'],side,space='FRAME')
        if not any(o is not None for o in obs):continue
        priors=[]
        if reference_channels:
            for i in range(len(times)):
                for n,rows in original.items():rig.pose.bones[n].rotation_quaternion=rows[i][1]
                for prefix in PREFIXES:
                    name=prefix+side
                    if name in reference_channels:rig.pose.bones[name].rotation_quaternion=reference_channels[name][i][1]
                view.update()
                priors.append(tuple(rig.pose.bones[n+'_'+side].head.copy() for n in ('Arm','Elbow','Wrist')))
        evidence=direction_evidence(data,side,options)
        segments,silhouettes=detect(data['screen'],side)
        silhouettes=bridge_silhouettes(silhouettes,relative)
        outlines,wrists,modes=hand_boundaries(data,segments,silhouettes,side)
        blocks=hand_geometry(source,side) if segments else None
        anchors={}
        # Read palm orientation from measured 3D joints, independently of its
        # image position. Near-camera depth loss holds the last observed pose.
        for i in range(len(times)):
            measured=depth[i].get(side) if len(depth)==len(times) else None
            if not measured:continue
            for n,rows in original.items():rig.pose.bones[n].rotation_quaternion=rows[i][1]
            view.update();w=rig.pose.bones['Wrist_'+side];q=w.matrix.to_quaternion()
            local=np.array([q.inverted()@(p-w.head) for p in hand_points(rig,side)])
            anchors[i]=cameras[i][0].to_quaternion()@measured_rotation(local,measured['points'],cameras[i],options.get('mirror',False))
        if anchors:
            ids=np.array(sorted(anchors));values=[]
            for i in range(len(times)):
                at=np.searchsorted(ids,i);lo=ids[max(0,at-1)];hi=ids[min(len(ids)-1,at)]
                alpha=0. if lo==hi else float(np.clip((times[i]-times[lo])/(times[hi]-times[lo]),0,1))
                values.append(anchors[int(lo)].slerp(anchors[int(hi)],alpha))
            values=quaternions(times,values,.06)
        else:values=None
        previous=None;previous_shoulder=None;hinge={};fitted=[]
        for i,o in enumerate(obs):
            if relative[i]<options.get('start',0) or options.get('end',0)>0 and relative[i]>options['end']:continue
            segment=next((s for s in segments if s['start']<=relative[i]<=s['end']),None)
            if options.get('only_punch_segments') and segment is None:continue
            silhouette=silhouettes[i] if segment else None
            use_outline=bool(silhouette and silhouette.get('silhouette') and i in outlines)
            for n,rows in original.items():rig.pose.bones[n].rotation_quaternion=rows[i][1]
            view.update();a,e,w=[rig.pose.bones[n+'_'+side] for n in ('Arm','Elbow','Wrist')]
            if previous_shoulder is not None and previous is not None:
                shift=np.array(a.head)-previous_shoulder
                previous=(previous[0]+shift,previous[1]+shift,previous[2])
            previous_shoulder=np.array(a.head)
            if previous is None:previous=(np.array(e.head),np.array(w.head),np.array(w.matrix.to_quaternion().to_matrix()))
            if (o is None or o.get('held') and not o.get('interpolated')) and not use_outline:
                # This stage proposes measured image fits only. The shared
                # final visibility owner handles all missing/returning frames.
                if reference_channels:
                    previous=None;previous_shoulder=None
                    continue
                if not fitted or times[i]-times[fitted[-1]]>.6:
                    previous=(np.array(e.head),np.array(w.head),np.array(w.matrix.to_quaternion().to_matrix()))
                    continue
                # Release the whole hinge, not separate local joint deltas.
                # The next measured frame inherits this actual preceding pose.
                pe,pw,ph=previous;origin=a.head.copy();upper=(e.head-origin).length;lower=(w.head-e.head).length
                pu=Vector(pe)-origin;pv=Vector(pw-pe);u=e.head-origin;v=w.head-e.head
                prior=frame_rotation(pu,pu.cross(pv));desired=frame_rotation(u,u.cross(v))
                dt=times[i]-times[i-1] if i else 1/data.get('fps',30);step=math.radians(540*dt)
                angle=2*math.acos(min(1.,abs(prior.dot(desired))))
                fraction=min(1.,step/max(1e-8,angle));frame=prior.slerp(desired,fraction)
                flex=pu.angle(pv)+float(np.clip(u.angle(v)-pu.angle(pv),-step,step))
                uu=frame@Vector((0,1,0));normal=frame@Vector((1,0,0));vv=Quaternion(normal,flex)@uu
                ee=origin+uu*upper;ww=ee+vv*lower
                ee,ww=front_safe_positions(origin,ee,ww,torso_front_plane(rig))
                qq=continuous_wrist_rotation(w.matrix.to_quaternion(),ww-ee,Matrix(ph.tolist()).to_quaternion(),math.radians(900*dt),72)
                anatomical_rotations(a,e,w,origin,ee,ww,qq,set_global_rotation,view,hinge)
                previous=(np.array(e.head),np.array(w.head),np.array(w.matrix.to_quaternion().to_matrix()))
                for prefix in ('Arm_','Elbow_','Wrist_','ArmTwist_','HandTwist_'):
                    name=prefix+side
                    if name in channels:channels[name][i]=channels[name][i][0],tuple(rig.pose.bones[name].rotation_quaternion)
                continue
            q=w.matrix.to_quaternion();offsets=np.array([q.inverted()@(p-w.head) for p in hand_points(rig,side)])
            anchor=q.copy()
            measured=depth[i].get(side) if len(depth)==len(times) else None
            if values is not None and (measured or segment):
                index=i
                if segment and not measured:
                    candidates=[j for j in anchors if segment['reference_index']<=j<=i]
                    index=max(candidates) if candidates else segment['reference_index']
                anchor=cameras[i][0].to_quaternion().inverted()@values[index]
            outline=None;wrist_target=None
            if use_outline:
                offsets=posed_hand(blocks,rig,side)
                if len(offsets)>180:
                    indices=np.unique(np.r_[np.linspace(0,len(offsets)-1,160).astype(int),np.argmin(offsets,axis=0),np.argmax(offsets,axis=0)])
                    offsets=offsets[indices]
                outline=image_shape(dict(silhouette['silhouette'],outline=outlines[i].tolist()),cameras[i],data['screen']['aspect'],options.get('mirror',False))
                edge=np.roll(outline,-1,axis=0)-outline;length=np.linalg.norm(edge,axis=1);distance=np.r_[0,np.cumsum(length)]
                sample=np.linspace(0,distance[-1],len(offsets),endpoint=False)
                index=np.minimum(np.searchsorted(distance,sample,side='right')-1,len(outline)-1)
                target=outline[index]+edge[index]*((sample-distance[index])/np.maximum(1e-8,length[index]))[:,None]
                weights=np.ones(len(offsets));wrist_target=wrists[i].copy()
                wrist_target[0]*=cameras[i][2]/data['screen']['aspect']
                if options.get('mirror'):wrist_target[0]=cameras[i][2]-wrist_target[0]
            else:
                target=o['points'].copy();target[:,0]*=cameras[i][2]/data['screen']['aspect']
                if options.get('mirror'):target[:,0]=cameras[i][2]-target[:,0]
                weights=np.asarray(o.get('joint_weights',o['mask']),float)*o['mask']
            pose_prior=None
            if priors:
                pa,pe,pw=priors[i];shift=a.head-pa
                pose_prior=(np.asarray(pe+shift),np.asarray(pw+shift),.7 if use_outline else 1.1)
            ee,ww,qq,row=fit_chain(a.head,e.head,w.head,anchor.to_matrix(),offsets,target,weights,cameras[i],
                evidence[i],previous,outline=outline,wrist_limit=math.radians(65 if use_outline else 72),
                front_plane=torso_front_plane(rig),close_range=True,wrist_target=wrist_target,
                # A fist can roll in the source image while depth detection is
                # lost. Its contour and wrist boundary remain observations;
                # allow them to update the held palm, without a facing flip.
                hand_limit=math.radians(30) if use_outline else None,
                orientation_prior=.7 if use_outline else (.4 if measured else .15),frame_guard=True,
                pose_prior=pose_prior)
            anatomical_rotations(a,e,w,a.head.copy(),ee,ww,qq,set_global_rotation,view,hinge)
            previous=(np.array(e.head),np.array(w.head),np.array(w.matrix.to_quaternion().to_matrix()))
            for prefix in ('Arm_','Elbow_','Wrist_','ArmTwist_','HandTwist_'):
                name=prefix+side
                if name in channels:channels[name][i]=channels[name][i][0],tuple(rig.pose.bones[name].rotation_quaternion)
            fitted.append(i)
            row['error_px']*=data.get('video_orientation',{}).get('decoded_height_width',[1920])[0]/1920
            row.update(frame=i+1,side=side,model='final-projection-closure-v2',silhouette=use_outline,
                orientation_source='video-3d-palm' if measured else 'observed-punch-palm-held' if use_outline and values is not None else 'captured-pose',
                wrist_limit_degrees=65 if use_outline else 72)
            report.append(row)
            if i%20==0:yield i+1,len(times)*2
        allowed=(relative>=options.get('start',0))&((relative<=options['end']) if options.get('end',0)>0 else True)
        if not reference_channels:smooth_switches(scratch,rig,channels,data,side,report,allowed)
        yield len(times)*2,len(times)*2
    if not reference_channels:
        from sam_ellen_fold_guard import repair_folds
        for done,total in repair_folds(scratch,rig,channels,original,data,options,cameras,report,protected):
            yield done,total


def smooth_switches(scratch,rig,channels,data,side,report,allowed=None):
    """Repair short detection switches without lagging the whole gesture."""
    from sam_ellen_arm_chain import frame_rotation,anatomical_rotations,bounded_wrist_rotation,front_safe_positions,torso_front_plane
    from sam_ellen_retarget import set_global_rotation
    from sam_ellen_temporal import quaternions,trajectory
    from sam_ellen_screen_fit import landmark_observations
    observations=landmark_observations(data['screen'],side,space='FRAME')
    view=scratch.view_layers[0];times=np.array([r['time'] for r in data['frames']])
    original={n:list(rows) for n,rows in channels.items()};frames=[];flexes=[];hands=[]
    a,e,w=[rig.pose.bones[n+'_'+side] for n in ('Arm','Elbow','Wrist')]
    for i in range(len(times)):
        for n,rows in original.items():rig.pose.bones[n].rotation_quaternion=rows[i][1]
        view.update();u=e.head-a.head;v=w.head-e.head
        frames.append(frame_rotation(u,u.cross(v)));flexes.append(u.angle(v));hands.append(w.matrix.to_quaternion())
    steps=np.array([0.]+[max(math.degrees(2*math.acos(min(1.,abs(frames[i].dot(frames[i-1]))))),
        math.degrees(2*math.acos(min(1.,abs(hands[i].dot(hands[i-1]))))),math.degrees(abs(flexes[i]-flexes[i-1]))) for i in range(1,len(times))])
    ids=np.flatnonzero(steps>35)
    if not len(ids):return
    envelope=np.max(np.maximum(0,1-abs(times[:,None]-times[ids][None,:])/.16),axis=1)
    if allowed is not None:envelope*=np.asarray(allowed,dtype=bool)
    filtered=quaternions(times,frames,.055);filtered_hands=quaternions(times,hands,.055);filtered_flex=trajectory(times,flexes,.055)
    hinge={}
    for i,weight in enumerate(envelope):
        if weight<=0:continue
        for n,rows in original.items():rig.pose.bones[n].rotation_quaternion=rows[i][1]
        view.update();origin=a.head.copy();upper=(e.head-origin).length;lower=(w.head-e.head).length
        q=frames[i].slerp(filtered[i],float(weight));flex=flexes[i]*(1-weight)+filtered_flex[i]*weight
        u=q@Vector((0,1,0));normal=q@Vector((1,0,0));v=Quaternion(normal,float(flex))@u
        ee=origin+u*upper;ww=ee+v*lower
        observed=observations[i]
        if observed is not None and (not observed.get('held') or observed.get('interpolated')):
            # A hinge-plane switch is not evidence for a different hand image
            # position. Retain half the fitted wrist correction while sharing
            # the remaining transition between the two fixed-length links.
            retain=.65 if np.mean(observed['points'][:,1])>.80 else .45
            ww=ww.lerp(w.head,retain);axis=ww-origin;distance=axis.length;axis.normalize()
            along=(distance*distance+upper*upper-lower*lower)/(2*max(1e-8,distance))
            bend=axis.cross(normal)
            if bend.length>1e-7:
                ee=origin+axis*along+bend.normalized()*math.sqrt(max(0.,upper*upper-along*along))
        ee,ww=front_safe_positions(origin,ee,ww,torso_front_plane(rig))
        hand=bounded_wrist_rotation(hands[i].slerp(filtered_hands[i],float(weight)),ww-ee,72)
        anatomical_rotations(a,e,w,origin,ee,ww,hand,set_global_rotation,view,hinge)
        for prefix in ('Arm_','Elbow_','Wrist_','ArmTwist_','HandTwist_'):
            n=prefix+side
            if n in channels:channels[n][i]=channels[n][i][0],tuple(rig.pose.bones[n].rotation_quaternion)
    report.append({'side':side,'model':'short-observation-switch-v1','switch_frames':(ids+1).tolist(),'modified_samples':int((envelope>0).sum())})

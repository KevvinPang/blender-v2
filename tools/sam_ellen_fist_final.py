"""Final anatomical chain fit to the visible fist after trajectory smoothing."""
import math
import numpy as np
from mathutils import Quaternion


def bridge_silhouettes(observations,times,max_gap=.105):
    """Bridge at most two missing mask samples inside a measured punch."""
    from sam_punch_evidence import bridge_silhouettes as bridge
    return bridge(observations,times,max_gap=max_gap)


def refine_final_fist(scratch,rig,source,channels,data,options,cameras,report):
    if not options or not options.get('automatic') or options.get('strength',0)<.999:return
    from sam_punch_evidence import detect
    from sam_ellen_punch_mesh import hand_geometry,posed_hand,image_shape
    from sam_ellen_image_refine import fit_chain
    from sam_ellen_arm_chain import anatomical_rotations,direction_evidence,torso_front_plane
    from sam_ellen_retarget import set_global_rotation
    side=options['side'];segments,observations=detect(data['screen'],side)
    if not segments:return
    view=scratch.view_layers[0];blocks=hand_geometry(source,side)
    times=np.array([r['time']-data['frames'][0]['time'] for r in data['frames']])
    observations=bridge_silhouettes(observations,times)

    evidence=direction_evidence(data,side,options);previous=None;last=-100;hinge={}
    originals={n:list(rows) for n,rows in channels.items()}
    fitted=[]
    from sam_ellen_screen_fit import hand_points
    from sam_ellen_punch_pose import observed_pose,measured_rotation,hand_boundaries
    from sam_ellen_punch_mesh import fist_pose_weights
    anchors={}
    outlines,wrist_targets,boundary_modes=hand_boundaries(data,segments,observations,side)
    fingers=[n for n in channels if n.endswith('_'+side) and n.startswith(('Thumb','IndexFinger','MiddleFinger','RingFinger','LittleFinger'))]
    depth=data['screen'].get('hand_pose_depth',{}).get('frames',[])
    from sam_punch_evidence import fist_observation
    from sam_ellen_temporal import quaternions
    for segment in segments:
        reference=segment['reference_index']
        for n,rows in channels.items():rig.pose.bones[n].rotation_quaternion=rows[reference][1]
        view.update();w=rig.pose.bones['Wrist_'+side]
        q=w.matrix.to_quaternion();points=hand_points(rig,side)
        offsets=np.array([q.inverted()@(p-w.head) for p in points])
        samples={};last=reference
        for index in range(reference,len(times)):
            if times[index]>=segment['end']:break
            measured=depth[index].get(side) if len(depth)==len(times) else None
            hand=data['screen']['frames'][index].get('hands',{}).get(side)
            if not measured or fist_observation(hand) is None:continue
            if times[index]-times[last]>.18:break
            for n,rows in originals.items():rig.pose.bones[n].rotation_quaternion=rows[index][1]
            view.update();w=rig.pose.bones['Wrist_'+side]
            world=np.array(measured['points'])
            anchor=measured_rotation(offsets,world,cameras[index],options.get('mirror'))
            set_global_rotation(w,anchor);view.update()
            world*=np.array([-1. if options.get('mirror') else 1.,-1.,-1.])
            for label,start in (('Thumb',1),('IndexFinger',5),('MiddleFinger',9),('RingFinger',13),('LittleFinger',17)):
                for j in range(3):
                    name=label+str(j if label=='Thumb' else j+1)+'_'+side
                    bone=rig.pose.bones.get(name)
                    if bone is None:continue
                    direction=cameras[index][0].to_quaternion().inverted()@__import__('mathutils').Vector(world[start+j+1]-world[start+j])
                    if direction.length<1e-6:continue
                    current=bone.matrix.to_quaternion()
                    model_points=hand_points(rig,side)
                    link=model_points[start+j+1]-model_points[start+j]
                    if link.length<1e-6:continue
                    delta=link.normalized().rotation_difference(direction.normalized())
                    set_global_rotation(bone,delta@current);view.update()
            samples[index]=(cameras[index][0].to_quaternion()@anchor,{n:rig.pose.bones[n].rotation_quaternion.copy() for n in fingers})
            last=index
        if not samples:
            hand=data['screen']['frames'][reference]['hands'][side]
            target=np.array(hand['points']);target[:,0]*=cameras[reference][2]/data['screen']['aspect']
            if options.get('mirror'):target[:,0]=cameras[reference][2]-target[:,0]
            anchor,row=observed_pose(offsets,target,hand.get('in_frame',[True]*21),cameras[reference],q,w.head)
            samples[reference]=(cameras[reference][0].to_quaternion()@anchor,{n:Quaternion(originals[n][reference][1]) for n in fingers})
            row.update(reference_frame=reference+1);report.append(row)
        else:
            report.append({'model':'measured-hand-depth-trajectory-v2','first_frame':min(samples)+1,'last_frame':max(samples)+1,'observed_samples':len(samples)})
        ids=sorted(samples);expanded=[];finger_rows={n:[] for n in fingers}
        for i,time in enumerate(times):
            left=max((j for j in ids if j<=i),default=ids[0]);right=min((j for j in ids if j>=i),default=ids[-1])
            alpha=0. if right==left else float(np.clip((time-times[left])/(times[right]-times[left]),0,1))
            expanded.append(samples[left][0].slerp(samples[right][0],alpha))
            for n in fingers:finger_rows[n].append(samples[left][1][n].slerp(samples[right][1][n],alpha))
        anchors[reference]=quaternions(times,expanded,.065)
        for n in fingers:finger_rows[n]=quaternions(times,finger_rows[n],.045)
        for j,weight in enumerate(fist_pose_weights(times,segment,data['screen']['frames'],side)):
            if weight<=0:continue
            for n in fingers:
                f,value=channels[n][j];channels[n][j]=f,tuple(Quaternion(value).slerp(finger_rows[n][j],weight))
    for i,time in enumerate(times):
        obs=observations[i]
        if not obs or not obs.get('silhouette') or not any(s['start']+.10<time<s['end']-.16 for s in segments):
            continue
        for n,rows in channels.items():rig.pose.bones[n].rotation_quaternion=rows[i][1]
        view.update();a,e,w=[rig.pose.bones[n+'_'+side] for n in ('Arm','Elbow','Wrist')]
        local=posed_hand(blocks,rig,side)
        segment=next(s for s in segments if s['start']+.10<time<s['end']-.16)
        anchor=cameras[i][0].to_quaternion().inverted()@anchors[segment['reference_index']][i]
        # Preserve the silhouette extrema with a moderate, deterministic cloud.
        if len(local)>220:
            extrema=np.unique(np.r_[np.linspace(0,len(local)-1,200).astype(int),np.argmin(local,axis=0),np.argmax(local,axis=0)])
            local=local[extrema]
        outline=image_shape(dict(obs['silhouette'],outline=outlines[i].tolist()) if i in outlines else obs['silhouette'],cameras[i],data['screen']['aspect'],options.get('mirror'))
        wrist_target=wrist_targets.get(i)
        if wrist_target is not None:
            wrist_target=wrist_target.copy();wrist_target[0]*=cameras[i][2]/data['screen']['aspect']
            if options.get('mirror'):wrist_target[0]=cameras[i][2]-wrist_target[0]
        # Point correspondences are unused by the outline objective. Uniform
        # contour sampling only supplies its depth/centering initialization.
        edge=np.roll(outline,-1,axis=0)-outline
        length=np.linalg.norm(edge,axis=1);distance=np.r_[0,np.cumsum(length)]
        sample=np.linspace(0,distance[-1],len(local),endpoint=False)
        index=np.minimum(np.searchsorted(distance,sample,side='right')-1,len(outline)-1)
        target=outline[index]+edge[index]*((sample-distance[index])/np.maximum(1e-8,length[index]))[:,None]
        if time-last>.12:previous=None;hinge={}
        ee,ww,qq,row=fit_chain(a.head,e.head,w.head,anchor.to_matrix(),local,target,
                               np.ones(len(local)),cameras[i],evidence[i],previous,outline=outline,
                               hand_step=math.radians(900*(times[i]-times[i-1] if i else 1/data['fps'])),
                               wrist_limit=math.radians(45),front_plane=torso_front_plane(rig),
                               hand_limit=math.radians(3),close_range=True,wrist_target=wrist_target)
        anatomical_rotations(a,e,w,a.head.copy(),ee,ww,qq,set_global_rotation,view,hinge)
        previous=(np.array(e.head),np.array(w.head),np.array(w.matrix.to_quaternion().to_matrix()));last=time
        for prefix in ('Arm_','Elbow_','Wrist_','HandTwist_','ArmTwist_'):
            bone=rig.pose.bones.get(prefix+side)
            if bone and bone.name in channels:channels[bone.name][i]=channels[bone.name][i][0],tuple(bone.rotation_quaternion)
        row.update(frame=i+1,model='video-observed-hand-pose-chain-v4',wrist_limit_degrees=45,
            observed_camera_rotation=list(cameras[i][0].to_quaternion()@anchor),
            wrist_boundary=boundary_modes.get(i),
            fitted_pose_delta_degrees=math.degrees(2*math.acos(min(1,abs(anchor.dot(qq))))));report.append(row)
        fitted.append(i)
        yield i+1,len(times)
    # Release the final fitted pose into the captured return. Restoring the
    # old chain on the next sample can otherwise create a large wrist roll.
    if fitted:
        edges=np.flatnonzero(np.diff(np.r_[-100,fitted])>1)
        ends=np.r_[edges[1:]-1,len(fitted)-1]
        for first,last_index in zip(edges,ends):
            lo,hi=fitted[first],fitted[last_index]
            for boundary,direction,duration in ((lo,-1,.16),(hi,1,.26)):
                for i in range(boundary+direction,-1 if direction<0 else len(times),direction):
                    gap=abs(times[i]-times[boundary])
                    if gap>=duration or i in fitted:break
                    weight=1-gap/duration;weight=weight*weight*(3-2*weight)
                    for n in ('Arm_','Elbow_','Wrist_','ArmTwist_','HandTwist_'):
                        name=n+side
                        if name not in channels:continue
                        delta=Quaternion(channels[name][boundary][1])@Quaternion(originals[name][boundary][1]).inverted()
                        if delta.w<0:delta.negate()
                        channels[name][i]=channels[name][i][0],tuple(Quaternion().slerp(delta,float(weight))@Quaternion(originals[name][i][1]))
        limit_final_wrist(scratch,rig,channels,data,side,segments)


def limit_final_wrist(scratch,rig,channels,data,side,segments):
    """Check global wrist motion after both wrist and twist release together."""
    from sam_ellen_arm_chain import continuous_wrist_rotation,anatomical_rotations,frame_rotation
    from sam_ellen_retarget import set_global_rotation
    times=np.array([r['time']-data['frames'][0]['time'] for r in data['frames']]);view=scratch.view_layers[0]
    previous=None;previous_frame=None;last=-100;hinge={}
    for i,time in enumerate(times):
        if not any(s['start']-.4<=time<=s['end']+.6 for s in segments):continue
        for n,rows in channels.items():rig.pose.bones[n].rotation_quaternion=rows[i][1]
        view.update();a,e,w=[rig.pose.bones[n+'_'+side] for n in ('Arm','Elbow','Wrist')]
        current=w.matrix.to_quaternion()
        if time-last>.12:previous=current.copy();previous_frame=None;hinge={}
        dt=times[i]-times[i-1] if i else 1/data['fps']
        weight=0.
        for s in segments:
            rise=float(np.clip((time-(s['start']-.25))/.25,0,1))
            fall=float(np.clip(((s['end']+.35)-time)/.35,0,1))
            blend=min(rise,fall);weight=max(weight,blend*blend*(3-2*blend))
        hand=continuous_wrist_rotation(current,w.head-e.head,previous,math.radians(900*dt),75-30*weight)
        elbow=e.head.copy();wrist=w.head.copy();u=(elbow-a.head).normalized();v=(wrist-elbow).normalized()
        frame=frame_rotation(u,u.cross(v));flex=u.angle(v);changed=False
        if previous_frame is not None:
            pf,pflex=previous_frame;angle=2*math.acos(min(1,abs(frame.dot(pf))))
            fraction=min(1.,math.radians(540*dt)/max(1e-8,angle),math.radians(540*dt)/max(1e-8,abs(flex-pflex)))
            if fraction<1:
                frame=pf.slerp(frame,fraction);flex=pflex+(flex-pflex)*fraction
                uu=frame@__import__('mathutils').Vector((0,1,0));normal=frame@__import__('mathutils').Vector((1,0,0))
                vv=Quaternion(normal,flex)@uu
                elbow=a.head+uu*(e.head-a.head).length;wrist=elbow+vv*(w.head-e.head).length;changed=True
                hand=continuous_wrist_rotation(hand,vv,previous,math.radians(900*dt),75-30*weight)
        if changed or abs(current.dot(hand))<1-1e-8:
            anatomical_rotations(a,e,w,a.head.copy(),elbow,wrist,hand,set_global_rotation,view,hinge)
            for prefix in ('Arm_','Elbow_','Wrist_','ArmTwist_','HandTwist_'):
                bone=rig.pose.bones.get(prefix+side)
                if bone and bone.name in channels:channels[bone.name][i]=channels[bone.name][i][0],tuple(bone.rotation_quaternion)
        actual_u=(e.head-a.head).normalized();actual_v=(w.head-e.head).normalized()
        previous_frame=(frame_rotation(actual_u,actual_u.cross(actual_v)),actual_u.angle(actual_v))
        previous=w.matrix.to_quaternion().copy();last=time

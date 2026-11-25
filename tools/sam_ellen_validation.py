"""Validate committed, evaluated poses using the actual animated scene camera."""
import numpy as np
import bpy
from mathutils import Vector


def polygon_area(points):
    p=np.asarray(points)
    return float(abs(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1)))*.5) if len(p)>2 else 0.


def curve_motion_review(rows):
    """Measure stop/start changes even when hand evidence is absent."""
    samples=[r for r in rows if not r.get('intermediate')]
    for i in range(1,len(samples)-1):
        before,row,after=samples[i-1:i+2]
        left=row['time']-before['time'];right=after['time']-row['time']
        if min(left,right)<=0:continue
        for side,arm in row.get('arms',{}).items():
            triple=[r.get('arms',{}).get(side,{}) for r in (before,row,after)]
            if not all('torso_relative_elbow_wrist' in a for a in triple):continue
            p,c,n=[np.array(a['torso_relative_elbow_wrist']) for a in triple]
            delta=np.linalg.norm((n-c)/right-(c-p)/left,axis=1)
            arm['curve_velocity_change_mps']=float(max(delta))
            arm['unobserved_stop_start_review']=bool(not any(a.get('hand_observed') for a in triple)
                and max(delta)>.8*max(.5,(left+right)*15))


def image_motion_review(actual,previous,target,old_target,height,dt,elbow=None,old_elbow=None):
    """Compare temporal screen motion, even when the input pose also jumps.

    Pixel limits scale with elapsed time and source resolution. An independently
    observed moving elbow may explain elbow motion; a moving palm alone does
    not establish an arbitrarily different elbow branch.
    """
    actual,previous,target,old_target=map(np.asarray,(actual,previous,target,old_target))
    hand_delta=target-old_target;wrist_delta=actual[2]-previous[2];elbow_delta=actual[1]-previous[1]
    hand_step=float(np.linalg.norm(hand_delta));wrist_step=float(np.linalg.norm(wrist_delta));elbow_step=float(np.linalg.norm(elbow_delta))
    wrist_residual=float(np.linalg.norm(wrist_delta-hand_delta))
    elbow_residual=float(np.linalg.norm(elbow_delta-(np.asarray(elbow)-old_elbow))) if elbow is not None and old_elbow is not None else elbow_step
    scale=max(.5,dt*30.)
    wrist_flag=wrist_residual>.06*scale and wrist_step>hand_step+.04*scale
    elbow_flag=elbow_residual>.075*scale+2*hand_step
    return {'wrist_step_px':wrist_step*height,'elbow_step_px':elbow_step*height,
            'source_hand_step_px':hand_step*height,'wrist_motion_residual_px':wrist_residual*height,
            'elbow_motion_residual_px':elbow_residual*height,'elbow_motion_independently_measured':elbow is not None and old_elbow is not None,
            'wrist_motion_review':bool(wrist_flag),'elbow_motion_review':bool(elbow_flag),
            'image_motion_review':bool(wrist_flag or elbow_flag)}


def validate_final(scene,rig,data,options,report):
    from sam_ellen_screen import camera_frames,project
    from sam_ellen_screen_fit import hand_points,landmark_observations
    from sam_ellen_sequence import timeline_frame
    from sam_ellen_shot import mapping,render_points
    from sam_ellen_gesture_layout import restore,points as layout_points
    restore(scene,data)
    shot=mapping(scene.camera)
    render_cameras=camera_frames(scene,rig,scene.camera,data)
    cameras=camera_frames(scene,rig,scene.camera,data,observation_space=True) if shot else render_cameras
    sides=('L','R') if options['side']=='BOTH' else (options['side'],)
    # Validate against decoded-frame detections, not the smoothed trajectories
    # used by the solver. Otherwise the same filter can hide its timing error.
    observations={s:landmark_observations(data['screen'],s,space='FRAME',position_seconds=0.) for s in sides}
    from sam_ellen_arm_visibility import observed_frames
    usable={s:observed_frames(data,s) for s in sides}
    from sam_ellen_body_clearance import observed_elbows
    measured_elbows={s:observed_elbows(data,s) for s in sides}
    from sam_ellen_palm_orientation import orientation_observations,palm_basis
    palm_observations={s:orientation_observations(data,s) for s in sides}
    from sam_ellen_arm_chain import direction_evidence
    directions={s:direction_evidence(data,s,options) for s in sides}
    independent_directions={s:direction_evidence(data,s,options,independent_only=True) for s in sides}
    # All image distances use decoded source pixels, including appended clips.
    height=data.get('video_orientation',{}).get('decoded_height_width',[options.get('height',1920)])[0]
    from sam_punch_evidence import detect
    from sam_ellen_punch_pose import hand_boundaries
    from sam_ellen_fist_final import bridge_silhouettes
    from sam_ellen_punch_mesh import hand_geometry,posed_hand,image_shape,hull,clip_polygon,supports
    from sam_ellen_contact_geometry import coverage_ratio
    outline_data={}
    for side in sides:
        segments,masks=detect(data['screen'],side)
        if not segments:continue
        masks=bridge_silhouettes(masks,np.array([r['time'] for r in data['frames']]))
        outlines,_,_=hand_boundaries(data,segments,masks,side)
        outline_data[side]=(outlines,hand_geometry(rig,side))
    frame,subframe=scene.frame_current,scene.frame_subframe
    physics=getattr(scene,'wiggle_enable',None)
    previous={};previous_velocity={};previous_center={};previous_image={};previous_projection={}
    physical_previous={}
    def physical_speeds(bones,time):
        result={}
        for bone in bones:
            q=bone.matrix.to_quaternion().normalized()
            if bone.name in physical_previous:
                prior,old_time=physical_previous[bone.name]
                result[bone.name]=float(np.degrees(2*np.arccos(np.clip(abs(q.dot(prior)),0,1)))/max(1e-8,time-old_time))
            physical_previous[bone.name]=(q.copy(),time)
        return result
    policy=data.get('_sam_arm_policy',[])
    from sam_ellen_image_geometry import head_geometry,posed_geometry
    geometry=head_geometry(rig)
    from sam_ellen_body_clearance import torso_geometry
    palm_geometry=torso_geometry(rig,include_bust=True)
    from sam_ellen_upper_posture import posture_evidence
    upper_policy={side:posture_evidence(data,side) for side in sides}
    from sam_ellen_body_clearance import torso_geometry,torso_envelope,arm_clearance
    body_geometry=torso_geometry(rig)
    try:
        if physics is not None:scene.wiggle_enable=False
        for i,sample in enumerate(data['frames']):
            f=timeline_frame(data,sample['time'])
            scene.frame_set(int(f),subframe=f-int(f))
            e=rig.evaluated_get(bpy.context.evaluated_depsgraph_get())
            body_envelope=torso_envelope(e,body_geometry)
            camera=cameras[i]
            render_camera=render_cameras[i]
            row={'frame':float(f),'time':sample['time'],'source_height':height,
                 'render_camera':scene.camera.name,'hands':{},'arms':{},'silhouettes':{},
                 'image_measurement_space':'source observation coordinates', 'shot':shot}
            face=data['screen']['frames'][i].get('face')
            if face:
                eye=(e.pose.bones['Eye_L'].head+e.pose.bones['Eye_R'].head)*.5
                target=np.array(face['center']);target[0]*=camera[2]/data['screen']['aspect']
                if options.get('mirror'):target[0]=camera[2]-target[0]
                row['face_center_px']=float(np.linalg.norm(np.array(project(eye,camera))-target)*height)
                rendered=np.array(project(eye,render_camera))
                row['render_face_source_px']=float(np.linalg.norm(rendered-target)*height)
                row['render_face_target_px']=float(np.linalg.norm(rendered-render_points(target,shot))*height)
                row['render_eye_center']=rendered.tolist()
                if geometry and face.get('jaw'):
                    points=posed_geometry(e,geometry)
                    width=(project(points[1],camera)-project(points[2],camera)).length
                    row['face_surface_width_ratio']=float(width/max(1e-8,np.ptp(np.array(face['jaw'])[:,0])))
                    row['render_face_surface_width_ratio']=float((project(points[1],render_camera)-project(points[2],render_camera)).length/max(1e-8,np.ptp(np.array(face['jaw'])[:,0])))
            for side in sides:
                arm,elbow,wrist=[e.pose.bones[n+'_'+side] for n in ('Arm','Elbow','Wrist')]
                u,v=arm.head-elbow.head,wrist.head-elbow.head
                angle=float(np.degrees(np.arccos(np.clip(u.normalized().dot(v.normalized()),-1,1))))
                steps={};accelerations={};speeds={}
                tracked=[arm,elbow,wrist]+[e.pose.bones[n+'_'+side] for n in ('ArmTwist','ArmTwist1','ArmTwist2','ArmTwist3','HandTwist','HandTwist1','HandTwist2','HandTwist3') if n+'_'+side in e.pose.bones]
                for bone in tracked:
                    q=bone.matrix.to_quaternion().normalized()
                    if bone.name in previous:
                        prior,time=previous[bone.name]
                        steps[bone.name]=float(np.degrees(2*np.arccos(np.clip(abs(q.dot(prior)),0,1))))
                        delta=q@prior.inverted()
                        if delta.w<0:delta.negate()
                        axis,turn=delta.to_axis_angle();dt=max(1e-8,sample['time']-time)
                        velocity=np.array(axis)*float(turn)/dt
                        speeds[bone.name]=float(np.degrees(np.linalg.norm(velocity)))
                        if bone.name in previous_velocity:
                            # Report physical units without treating a real fast
                            # source gesture as a failure by an arbitrary cutoff.
                            accelerations[bone.name]=float(np.degrees(np.linalg.norm(velocity-previous_velocity[bone.name])/dt))
                        previous_velocity[bone.name]=velocity
                    previous[bone.name]=q.copy(),sample['time']
                palm_forward=wrist.matrix.to_quaternion()@Vector((0,1,0))
                wrist_angle=float(np.degrees(v.normalized().angle(palm_forward)))
                lengths=[(elbow.head-arm.head).length,(wrist.head-elbow.head).length]
                rest=[(elbow.bone.head_local-arm.bone.head_local).length,(wrist.bone.head_local-elbow.bone.head_local).length]
                arm_row={'elbow_angle_degrees':angle,'wrist_bend_degrees':wrist_angle,'rotation_step_degrees':steps,
                    'angular_speed_degrees_per_second':speeds,'angular_acceleration_degrees_per_second2':accelerations,
                    'link_length_ratio':[a/b for a,b in zip(lengths,rest)]}
                for name,speed in physical_speeds(tracked,sample['time']).items():
                    speeds[name]=max(speeds.get(name,0.),speed)
                chest=e.pose.bones.get('UpperBody2') or e.pose.bones.get('UpperBody1')
                down=chest.matrix.to_quaternion()@chest.bone.matrix_local.to_quaternion().inverted()@Vector((0,0,-1))
                body_rotation=chest.matrix.to_quaternion()@chest.bone.matrix_local.to_quaternion().inverted()
                arm_row['torso_relative_elbow_wrist']=[list(body_rotation.inverted()@(b.head-arm.head)) for b in (elbow,wrist)]
                arm_row['hand_observed']=bool(usable[side][i])
                arm_row['upper_posture_evidence']=upper_policy[side][i]
                axis=(wrist.head-arm.head).normalized();distance=(wrist.head-arm.head).length
                along=(distance**2+lengths[0]**2-lengths[1]**2)/(2*max(1e-8,distance))
                radius=np.sqrt(max(0.,lengths[0]**2-along**2));low=down-axis*down.dot(axis)
                lowest=axis*along+(low.normalized() if low.length>1e-8 else axis.orthogonal())*radius
                raised=float(np.degrees((elbow.head-arm.head).angle(down)))
                minimum_raise=float(np.degrees(lowest.angle(down)))
                arm_row.update(elbow_independently_measured=bool(measured_elbows[side][i]),
                    upper_arm_raise_degrees=raised,lowest_reachable_raise_degrees=minimum_raise,
                    unsupported_elbow_raise=bool(not measured_elbows[side][i] and raised>upper_policy[side][i]['limit_degrees']+5 and raised-minimum_raise>10))
                arm_row['torso_proxy_clearance']=arm_clearance(arm.head,elbow.head,wrist.head,body_envelope)
                palm=palm_observations[side][i]
                if palm is not None:
                    expected=palm_basis(np.asarray(palm['points'])*[-1. if options.get('mirror') else 1.,-1.,-1.])
                    actual=palm_basis(np.array([camera[0]@p for p in hand_points(e,side)]))
                    if actual is not None and expected is not None:
                        arm_row['palm_normal_error_degrees']=float(np.degrees(np.arccos(np.clip(actual[:,2]@expected[:,2],-1,1))))
                        reflected=palm_basis(np.asarray(palm['points'])*[-1. if options.get('mirror') else 1.,-1.,1.])
                        reflection_error=float(np.degrees(np.arccos(np.clip(actual[:,2]@reflected[:,2],-1,1))))
                        arm_row['palm_depth_hypothesis_min_error_degrees']=min(arm_row['palm_normal_error_degrees'],reflection_error)
                        physical=palm_basis(np.array(hand_points(e,side)))[:,2]*(-1 if side=='L' else 1)
                        arm_row['palm_up_dot']=float(physical@np.asarray(-down))
                        arm_row['palm_evidence_source']='matched monocular hand depth; reflection ambiguous, not 3D ground truth'
                        arm_row['palm_orientation_review']=bool(arm_row['palm_normal_error_degrees']>35)
                if len(policy)==len(data['frames']) and side in policy[i]:
                    p=policy[i][side]
                    from sam_ellen_arm_policy import supported_elbow_minimum
                    arm_row.update(minimum_supported_elbow=supported_elbow_minimum(data,side,i),
                        evidence_state=p.get('mode','unknown'),source_elbow=p['source_elbow'])
                pose_rows=data['screen'].get('pose_arms',[])
                direct_pose=pose_rows[i].get(side) if len(pose_rows)==len(data['frames']) else None
                independent=independent_directions[side][i] if direct_pose and direct_pose.get('confidence',0)>=.5 else None
                source=independent if independent is not None else directions[side][i]
                if source is not None and source[1]>.25 and usable[side][i]:
                    expected=np.array(source[0],dtype=float)
                    expected[0]*=camera[2]/data['screen']['aspect']
                    expected/=max(1e-8,np.linalg.norm(expected))
                    actual=np.array(project(elbow.head,camera))-np.array(project(wrist.head,camera))
                    actual/=max(1e-8,np.linalg.norm(actual))
                    arm_row['forearm_direction_error_degrees']=float(np.degrees(np.arccos(np.clip(actual@expected,-1,1))))
                    arm_row['direction_evidence_weight']=float(source[1])
                    arm_row['direction_evidence_source']='independent-video-pose' if independent is not None else 'color-axis-or-hand'
                    arm_row['projected_shoulder_elbow_wrist']=[list(project(b.head,camera)) for b in (arm,elbow,wrist)]
                obs=observations[side][i]
                actual_projection=np.array([project(b.head,camera) for b in (arm,elbow,wrist)])
                if obs is not None and not obs.get('held') and side not in previous_image and side in previous_projection:
                    delta=actual_projection-previous_projection[side]
                    arm_row['first_observation_motion']={'wrist_step_px':float(np.linalg.norm(delta[2])*height),
                        'elbow_step_px':float(np.linalg.norm(delta[1])*height),
                        'previous_frame_without_comparable_landmarks':True}
                previous_projection[side]=actual_projection
                if obs is not None and not obs.get('held') and obs['mask'].sum()>=4:
                    target=obs['points'].copy();target[:,0]*=camera[2]/data['screen']['aspect']
                    if options.get('mirror'):target[:,0]=camera[2]-target[:,0]
                    # Use the wrist when localized, otherwise the same stable
                    # palm joints in both frames; a changing mask is no anchor.
                    indices=[0] if obs['mask'][0] else [j for j in (5,9,13,17) if obs['mask'][j]]
                    if not indices:indices=list(np.flatnonzero(obs['mask']))
                    center=target[indices].mean(0)
                    actual=np.array([project(b.head,camera) for b in (arm,elbow,wrist)])
                    source_elbow=None
                    if direct_pose and direct_pose.get('elbow_observed') and direct_pose.get('confidence',0)>=.5:
                        source_elbow=np.array(direct_pose['points'][1],float);source_elbow[0]*=camera[2]/data['screen']['aspect']
                        if options.get('mirror'):source_elbow[0]=camera[2]-source_elbow[0]
                    prior=previous_image.get(side)
                    if prior is not None and prior[0]==i-1 and prior[4]==indices:
                        arm_row['image_motion']=image_motion_review(actual,prior[1],center,prior[2],height,
                            sample['time']-data['frames'][i-1]['time'],source_elbow,prior[3])
                    previous_image[side]=(i,actual,center,source_elbow,indices)
                else:previous_image.pop(side,None)
                if i and obs is not None and not obs.get('held') and side in previous_center:
                    raw=np.asarray(sample['positions']);oldraw=np.asarray(data['frames'][i-1]['positions'])
                    indices=[75,76,77] if side=='L' else [39,40,41]
                    pu,pv=np.diff(raw[indices],axis=0);ou,ov=np.diff(oldraw[indices],axis=0)
                    turn=lambda a,b:np.degrees(np.arccos(np.clip(a@b/max(1e-8,np.linalg.norm(a)*np.linalg.norm(b)),-1,1)))
                    source_step=max(turn(pu,ou),turn(pv,ov))
                    center=obs['points'][obs['mask']].mean(0)
                    image_step=np.linalg.norm(center-previous_center[side])
                    arm_row['unsupported_acceleration_spike']=bool(source_step<4 and image_step<.006 and
                        any(accelerations.get(n,0)>18000 and steps.get(n,0)>12 for n in ('Arm_'+side,'Elbow_'+side)))
                if obs is not None and not obs.get('held'):previous_center[side]=obs['points'][obs['mask']].mean(0)
                else:previous_center.pop(side,None)
                from sam_ellen_finger_constraints import measure as measure_fingers
                arm_row['fingers']=measure_fingers(e,side)
                row['arms'][side]=arm_row
                if side in outline_data and i in outline_data[side][0]:
                    outlines,blocks=outline_data[side]
                    target_outline=image_shape({'outline':outlines[i]},camera,data['screen']['aspect'],options.get('mirror',False))
                    q=wrist.matrix.to_quaternion()
                    skin=[wrist.head+q@Vector(p) for p in posed_hand(blocks,e,side)]
                    polygon=clip_polygon(hull([project(p,camera) for p in skin]),camera[2])
                    target_outline=clip_polygon(target_outline,camera[2])
                    if len(polygon)>2 and len(target_outline)>2:
                        row['silhouettes'][side]={'support_error_px':float(np.mean(abs(supports(polygon)-supports(target_outline)))*height),
                            'visible_area_ratio':polygon_area(polygon)/max(1e-8,polygon_area(target_outline))}
                obs=observations[side][i]
                if obs is None or obs.get('held'):continue
                actual=np.array([project(p,camera) for p in hand_points(e,side)])
                target=obs['points'].copy();target[:,0]*=camera[2]/data['screen']['aspect']
                if options.get('mirror'):target[:,0]=camera[2]-target[:,0]
                error=np.linalg.norm(actual-target,axis=1)*height
                tips=np.array([4,8,12,16,20]);tips=tips[obs['mask'][tips]]
                s=getattr(scene,'sam_ellen',None)
                phone=bool(s and s.phone_enabled and s.phone_side==side and scene.get('sam_phone_grip_weight',0)>.001)
                row['hands'][side]={'landmarks_px':float(error[obs['mask']].mean()),
                    'tips_px':float(error[tips].mean()) if len(tips) else None,
                    'intentional_phone_override':phone,'observation_weight':float(obs['weight']),
                    'first_observation':bool(i==0 or not usable[side][i-1])}
                # Independently evaluate the physical output camera as well.
                # Keep both raw-source difference and agreed-shot residual.
                rendered=np.array([project(p,render_camera) for p in hand_points(e,side)])
                desired=layout_points(obs['points'],data,i,camera[2],options.get('mirror',False),output=True) if data.get('_sam_gesture_layout') else render_points(target,shot)
                render_error=np.linalg.norm(rendered-desired,axis=1)*height
                row['hands'][side].update(
                    render_source_px=float(np.linalg.norm(rendered-target,axis=1)[obs['mask']].mean()*height),
                    render_target_px=float(render_error[obs['mask']].mean()))
                semantic=data.get('_sam_semantic',{}).get(side,[])
                descriptor=semantic[i] if len(semantic)==len(data['frames']) else None
                if descriptor and descriptor.get('region')=='mouth':
                    from sam_ellen_retarget import mouth_landmark
                    mouth_bone,mouth_local=mouth_landmark(e)
                    mouth_point=e.pose.bones[mouth_bone].matrix@Vector(mouth_local)
                    mouth_xy=np.asarray(project(mouth_point,render_camera))
                    output_eye_width=float((project(e.pose.bones['Eye_L'].head,render_camera)
                        -project(e.pose.bones['Eye_R'].head,render_camera)).length)
                    coverage=coverage_ratio(rendered,mouth_xy,output_eye_width)
                    palm_points=[hand_points(e,side)[j] for j in (0,5,9,13,17)]
                    palm_depth=float(np.median([-(render_camera[0]@p).z for p in palm_points]))
                    mouth_depth=float(-(render_camera[0]@mouth_point).z)
                    from sam_ellen_semantic import hand_anchor
                    contact_point=Vector(hand_anchor(np.asarray(hand_points(e,side)),descriptor))
                    front_gap=mouth_depth-float(-(render_camera[0]@contact_point).z)
                    row['hands'][side].update(semantic_intent='mouth-cover',
                        semantic_weight=float(descriptor['weight']),
                        semantic_phase=descriptor.get('phase','hold'),
                        contact_activation=float(descriptor.get('activation',1.)),
                        source_mouth_coverage=float(descriptor.get('coverage',0.)),
                        output_mouth_coverage=coverage,palm_front_gap_m=mouth_depth-palm_depth,
                        contact_front_gap_m=front_gap,
                        mouth_cover_pass=bool(coverage>=.70 and front_gap>=.003))
                source_visible=obs['mask']&(target[:,0]>=0)&(target[:,0]<=camera[2])&(target[:,1]>=0)&(target[:,1]<=1)
                depths=np.array([-(camera[0]@p).z for p in hand_points(e,side)])
                visible=(actual[:,0]>=0)&(actual[:,0]<=camera[2])&(actual[:,1]>=0)&(actual[:,1]<=1)&(depths>.01)
                row['hands'][side].update(source_visible_joints=int(source_visible.sum()),observation_visible_joints=int(visible.sum()),
                    source_visible_joints_outside=int((source_visible&~visible).sum()))
                output_visible=(rendered[:,0]>=0)&(rendered[:,0]<=camera[2])&(rendered[:,1]>=0)&(rendered[:,1]<=1)&(depths>.01)
                desired_visible=source_visible&(desired[:,0]>=0)&(desired[:,0]<=camera[2])&(desired[:,1]>=0)&(desired[:,1]<=1)
                row['hands'][side].update(output_visible_joints=int(output_visible.sum()),rendered_visible_joints=int(output_visible.sum()),
                    source_visible_output_joints_outside=int((source_visible&~output_visible).sum()),
                    desired_output_visible_joints=int(desired_visible.sum()),
                    source_joints_cropped_by_shot=int((source_visible&~desired_visible).sum()),
                    desired_output_joints_outside=int((desired_visible&~output_visible).sum()),
                    output_projection_flag=bool(render_error[obs['mask']].mean()>height*.12 or
                        source_visible.sum()>=15 and (source_visible&~output_visible).sum()>=5))
                mask=obs['mask'];p=actual[mask];t=target[mask]
                diameter=lambda x:float(np.max(np.linalg.norm(x[:,None,:]-x[None,:,:],axis=-1)))
                if len(p)>2:
                    row['hands'][side]['visible_extent_ratio']=diameter(p)/max(1e-8,diameter(t))
                    row['hands'][side]['projection_flag']=bool(row['hands'][side]['visible_extent_ratio']>1.8 or row['hands'][side]['landmarks_px']>height*.12 or
                        source_visible.sum()>=15 and (source_visible&~visible).sum()>=5)
                if data.get('_sam_gesture_layout') and len(p)>2:
                    extent=diameter(rendered[mask])/max(1e-8,diameter(desired[mask]))
                    row['hands'][side]['output_extent_ratio']=extent
                    row['hands'][side]['source_projection_flag']=row['hands'][side]['projection_flag']
                    row['hands'][side]['projection_flag']=bool(row['hands'][side]['output_projection_flag'] or extent>1.8)
                from sam_ellen_body_clearance import _normals
                envelope=torso_envelope(e,palm_geometry)
                if len(envelope):
                    palm=np.array(hand_points(e,side))[[0,5,9,13,17]]
                    clearance=(palm@_normals.T)[:,None,:]-envelope[None,:,:]
                    row['hands'][side]['palm_torso_proxy_clearance_m']=float(clearance.max(2).min())
                if not np.isfinite(actual).all():raise ValueError('Invalid final hand projection')
            report.append(row)
            # Check actual F-curve interpolation, not merely the sampled poses.
            for fraction in ((.25,.5,.75) if i+1<len(data['frames']) else ()):
                next_frame=timeline_frame(data,data['frames'][i+1]['time'])
                midpoint=f*(1-fraction)+next_frame*fraction
                scene.frame_set(int(midpoint),subframe=midpoint-int(midpoint))
                mid=rig.evaluated_get(bpy.context.evaluated_depsgraph_get())
                mid_envelope=torso_envelope(mid,body_geometry)
                check={'frame':float(midpoint),'time':sample['time']*(1-fraction)+data['frames'][i+1]['time']*fraction,
                    'intermediate':True,'hands':{},'arms':{},'silhouettes':{}}
                for side in sides:
                    a,b,c=[mid.pose.bones[n+'_'+side] for n in ('Arm','Elbow','Wrist')]
                    u=b.head-a.head;v=c.head-b.head
                    values={'elbow_angle_degrees':float(np.degrees((-u).angle(v))),
                        'wrist_bend_degrees':float(np.degrees(v.angle(c.matrix.to_quaternion()@Vector((0,1,0))))),
                        'rotation_step_degrees':{},'link_length_ratio':[
                            u.length/(b.bone.head_local-a.bone.head_local).length,
                            v.length/(c.bone.head_local-b.bone.head_local).length]}
                    tracked=[a,b,c]+[mid.pose.bones[n+'_'+side] for n in ('ArmTwist','ArmTwist1','ArmTwist2','ArmTwist3','HandTwist','HandTwist1','HandTwist2','HandTwist3') if n+'_'+side in mid.pose.bones]
                    values['angular_speed_degrees_per_second']=physical_speeds(tracked,check['time'])
                    values['torso_proxy_clearance']=arm_clearance(a.head,b.head,c.head,mid_envelope)
                    if len(policy)==len(data['frames']) and side in policy[i] and side in policy[i+1]:
                        from sam_ellen_arm_policy import supported_elbow_minimum
                        values['minimum_supported_elbow']=min(supported_elbow_minimum(data,side,k) for k in (i,i+1))-1.
                    from sam_ellen_finger_constraints import measure as measure_fingers
                    values['fingers']=measure_fingers(mid,side)
                    check['arms'][side]=values
                report.append(check)
            yield i+1,len(data['frames'])
        curve_motion_review(report)
    finally:
        scene.frame_set(frame,subframe=subframe)
        if physics is not None:scene.wiggle_enable=physics


def require_safe_arms(rows,sides=None,active_times=None):
    """Reject unsafe evaluated curves; callers restore the previous action.

    Image-fit and palm-evidence warnings remain reviewable. They must not be
    conflated with an impossible skeleton or a trajectory through the torso.
    """
    from sam_ellen_motion_evidence import HARD_ARM_RATE
    failures=[]
    for row in rows:
        if active_times is not None:
            times,active=active_times
            right=min(len(times)-1,int(np.searchsorted(times,row['time'])))
            left=max(0,right-1)
            if not (active[left] or active[right]):continue
        for side,arm in row.get('arms',{}).items():
            if sides is not None and side not in sides:continue
            causes=[]
            if arm.get('torso_proxy_clearance',0)<-.002:causes.append('手臂穿入躯干')
            if any(not np.isfinite(v) or abs(v-1)>.005 for v in arm.get('link_length_ratio',[])):causes.append('骨段长度异常')
            if arm.get('wrist_bend_degrees',0)>75.1:causes.append('手腕超出解剖范围')
            if arm.get('elbow_angle_degrees',180)<arm.get('minimum_supported_elbow',35.)-.5:causes.append('肘部过度翻折')
            if any(v>HARD_ARM_RATE+.1 or not np.isfinite(v) for v in arm.get('angular_speed_degrees_per_second',{}).values()):causes.append('手臂瞬时转动过快')
            if causes:failures.append('%g帧%s：%s'%(row['frame'],side,'、'.join(causes)))
    if failures:
        raise ValueError('动捕未通过手臂安全检查，已保留原动作。'+ '；'.join(failures[:4]))


def summary(rows):
    faces=[r['face_center_px'] for r in rows if 'face_center_px' in r]
    hands=[h['landmarks_px'] for r in rows for h in r['hands'].values() if not h.get('intentional_phone_override')]
    steps=[angle for r in rows for a in r.get('arms',{}).values() for angle in a['rotation_step_degrees'].values()]
    accelerations=[v for r in rows for a in r.get('arms',{}).values() for v in a.get('angular_acceleration_degrees_per_second2',{}).values()]
    directions=[a['forearm_direction_error_degrees'] for r in rows for a in r.get('arms',{}).values() if 'forearm_direction_error_degrees' in a]
    wrists=[a['wrist_bend_degrees'] for r in rows for a in r.get('arms',{}).values() if 'wrist_bend_degrees' in a]
    widths=[r['face_surface_width_ratio'] for r in rows if 'face_surface_width_ratio' in r]
    measured=[h for r in rows for h in r['hands'].values() if not h.get('intentional_phone_override')]
    mouth_checks=[(r['frame'],side,h) for r in rows for side,h in r.get('hands',{}).items()
                  if h.get('semantic_intent')=='mouth-cover' and h.get('semantic_phase','hold')=='hold'
                  and h.get('semantic_weight',0.)>=.45]
    mouth_fail=[(frame,side,h.get('output_mouth_coverage'),h.get('contact_front_gap_m',h.get('palm_front_gap_m')))
                for frame,side,h in mouth_checks if not h.get('mouth_cover_pass')]
    silhouettes=[s for r in rows for s in r.get('silhouettes',{}).values()]
    issues=[]
    speeds=[v for r in rows for a in r.get('arms',{}).values() for v in a.get('angular_speed_degrees_per_second',{}).values()]
    unsupported=[(r['frame'],s) for r in rows for s,a in r.get('arms',{}).items() if a.get('unsupported_acceleration_spike')]
    unsupported_raises=[(r['frame'],s) for r in rows for s,a in r.get('arms',{}).items() if a.get('unsupported_elbow_raise')]
    prior_folds=[(r['frame'],s) for r in rows for s,a in r.get('arms',{}).items()
                 if a['elbow_angle_degrees']<a.get('minimum_supported_elbow',35.)-.5]
    direction_fail=[(r['frame'],s) for r in rows for s,a in r.get('arms',{}).items()
                    if a.get('direction_evidence_weight',0)>=.5 and a.get('forearm_direction_error_degrees',0)>
                    (20 if a.get('direction_evidence_source')=='independent-video-pose' else 100)]
    palm_errors=[a['palm_normal_error_degrees'] for r in rows for a in r.get('arms',{}).values() if 'palm_normal_error_degrees' in a]
    palm_fail=[(r['frame'],s) for r in rows for s,a in r.get('arms',{}).items() if a.get('palm_orientation_review')]
    finger_fail=[(r['frame'],side) for r in rows for side,a in r.get('arms',{}).items() if a.get('fingers',{}).get('invalid')]
    if finger_fail:issues.append('finger articulation exceeds final joint limits')
    if prior_folds:issues.append('elbow flexion exceeds evidence-supported prior')
    if direction_fail:issues.append('forearm contradicts measured image direction')
    if palm_fail:issues.append('palm facing needs review against video hand evidence')
    if unsupported:issues.append('unsupported arm acceleration spike')
    if unsupported_raises:issues.append('unmeasured elbow chose an unnecessarily elevated branch')
    image_motion=[(r['frame'],s) for r in rows for s,a in r.get('arms',{}).items() if a.get('image_motion',{}).get('image_motion_review')]
    stop_start=[(r['frame'],s) for r in rows for s,a in r.get('arms',{}).items() if a.get('unobserved_stop_start_review')]
    velocity_changes=[a['curve_velocity_change_mps'] for r in rows for a in r.get('arms',{}).values() if 'curve_velocity_change_mps' in a]
    if stop_start:issues.append('unobserved arm trajectory stops or starts abruptly')
    if image_motion:issues.append('arm screen motion exceeds video evidence')
    if any(abs(v-1)>.005 for r in rows for a in r.get('arms',{}).values() for v in a.get('link_length_ratio',[])):issues.append('arm link length changed')
    folded=[(r['frame'],side) for r in rows for side,a in r.get('arms',{}).items()
            if a['elbow_angle_degrees']<45 and a.get('wrist_bend_degrees',0)>55]
    if any(h.get('source_visible_joints',0)>=15 and h.get('rendered_visible_joints',21)<=4 for h in measured):issues.append('observed hand lost outside camera')
    if any(v>1200.1 for v in speeds) or not speeds and any(v>40 for v in steps):issues.append('arm rotation exceeds 1200 degrees per second')
    if any(v>75.1 for v in wrists):issues.append('wrist anatomy limit exceeded')
    if folded:issues.append('deep elbow fold combined with extreme wrist bend')
    body_intersections=[(r['frame'],side) for r in rows for side,a in r.get('arms',{}).items()
                        if a.get('torso_proxy_clearance',0)<-.002]
    if body_intersections:issues.append('arm intersects skinned torso safety envelope')
    if any(h.get('projection_flag') for h in measured):issues.append('hand image projection needs review')
    if any(h.get('palm_torso_proxy_clearance_m',0)<-.005 for h in measured):issues.append('observed palm intersects torso proxy; review render')
    if any(h.get('output_projection_flag') for h in measured):issues.append('hand projection in selected output shot needs review')
    if mouth_fail:issues.append('mouth-cover gesture misses the mouth in the final camera')
    if any(s['visible_area_ratio']>1.8 or s['visible_area_ratio']<.55 for s in silhouettes):issues.append('fist silhouette size needs review')
    # Morphology is allowed to differ, but severe crop/scale mismatch is not a
    # pass. Normalize thresholds to decoded source pixels, not render height.
    framing_bad=[r['frame'] for r in rows if r.get('face_center_px',0)>r.get('source_height',1280)*.06]
    scale_bad=[r['frame'] for r in rows if 'face_surface_width_ratio' in r
               and not .65<=r['face_surface_width_ratio']<=1.55]
    if len(framing_bad)>max(2,len(faces)*.05):issues.append('face position differs from source framing')
    if len(scale_bad)>max(2,len(widths)*.05):issues.append('character face scale differs substantially from source')
    safety_issues=[issue for issue in issues if any(term in issue for term in
        ('link length','1200 degrees','wrist anatomy','deep elbow','intersects','elbow flexion'))]
    return {'measurement':'committed evaluated scene / source-coordinate evidence checks plus independent physical output-camera checks / decoded-height pixels',
        'shot':next((r['shot'] for r in rows if r.get('shot')),None),
        'render_face_source_mean_px':float(np.mean([r['render_face_source_px'] for r in rows if 'render_face_source_px' in r])) if any('render_face_source_px' in r for r in rows) else None,
        'render_face_target_mean_px':float(np.mean([r['render_face_target_px'] for r in rows if 'render_face_target_px' in r])) if any('render_face_target_px' in r for r in rows) else None,
        'render_hand_source_mean_px':float(np.mean([h['render_source_px'] for h in measured if 'render_source_px' in h])) if any('render_source_px' in h for h in measured) else None,
        'render_hand_target_mean_px':float(np.mean([h['render_target_px'] for h in measured if 'render_target_px' in h])) if any('render_target_px' in h for h in measured) else None,
        'output_projection_flagged_samples':sum(h.get('output_projection_flag',False) for h in measured),
        'safety_passed':not safety_issues,'safety_issues':safety_issues,
        'framing_review_frames':framing_bad,'scale_review_frames':scale_bad,
        'semantic_contact_status':('measured in final camera: 2D mouth coverage and palm depth order'
                                   if mouth_checks else 'no sustained mouth-cover segment detected'),
        'mouth_cover_samples':len(mouth_checks),
        'mouth_cover_review_frames':mouth_fail,
        'mouth_coverage_median':float(np.median([h['output_mouth_coverage'] for _,_,h in mouth_checks])) if mouth_checks else None,
        'coverage':{'face_samples':len(faces),'hand_samples':len(hands),'palm_orientation_samples':len(palm_errors)},
        'render_cameras':sorted({r['render_camera'] for r in rows if 'render_camera' in r}),
        'occlusion_measured':bool(mouth_checks),'samples':len(rows),
        'torso_clearance_model':'skinned torso convex slabs / finite-radius arm links / shoulder socket exemption',
        'torso_proxy_intersection_frames':body_intersections,
        'quality_passed':not issues,'quality_issues':issues,
        'intermediate_samples':sum(r.get('intermediate',False) for r in rows),
        'finger_articulation_review_frames':finger_fail,'unsupported_elbow_fold_frames':prior_folds,'unsupported_acceleration_frames':unsupported,
        'unsupported_elbow_raise_frames':unsupported_raises,
        'image_motion_review_frames':image_motion,
        'unobserved_stop_start_frames':stop_start,
        'curve_velocity_change_p95_mps':float(np.percentile(velocity_changes,95)) if velocity_changes else None,
        'curve_velocity_change_max_mps':max(velocity_changes) if velocity_changes else None,
        'contradictory_forearm_frames':direction_fail,
        'palm_orientation_review_frames':palm_fail,
        'palm_normal_median_error_degrees':float(np.median(palm_errors)) if palm_errors else None,
        'palm_normal_mean_error_degrees':float(np.mean(palm_errors)) if palm_errors else None,
        'arm_speed_max_degrees_per_second':max(speeds) if speeds else None,
        'almost_entire_observed_hand_outside_samples':sum(h.get('source_visible_joints',0)>=15 and h.get('rendered_visible_joints',21)<=4 for h in measured),
        'fist_silhouette_support_median_px':float(np.median([s['support_error_px'] for s in silhouettes])) if silhouettes else None,
        'fist_silhouette_area_median_ratio':float(np.median([s['visible_area_ratio'] for s in silhouettes])) if silhouettes else None,
        'face_surface_width_median_ratio':float(np.median(widths)) if widths else None,
        'hand_projection_flagged_samples':sum(h.get('projection_flag',False) for r in rows for h in r['hands'].values()),
        'face_median_px':float(np.median(faces)) if faces else None,
        'face_mean_px':float(np.mean(faces)) if faces else None,
        'hand_mean_px':float(np.mean(hands)) if hands else None,
        'hand_median_px':float(np.median(hands)) if hands else None,
        'arm_rotation_max_step_degrees':max(steps) if steps else None,
        'arm_rotation_p95_step_degrees':float(np.percentile(steps,95)) if steps else None,
        'arm_steps_over_40_degrees':sum(v>40 for v in steps),
        'angular_acceleration_p95_degrees_per_second2':float(np.percentile(accelerations,95)) if accelerations else None,
        'angular_acceleration_max_degrees_per_second2':max(accelerations) if accelerations else None,
        'wrist_bend_max_degrees':max(wrists) if wrists else None,
        'wrist_bends_over_75_degrees':sum(v>75.1 for v in wrists),
        'combined_extreme_fold_samples':len(folded),
        'combined_extreme_fold_frames':folded,
        'forearm_direction_median_error_degrees':float(np.median(directions)) if directions else None}

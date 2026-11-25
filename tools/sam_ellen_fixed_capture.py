"""Fixed camera, grounded horizontal root, conservative unobserved pelvis.

All upper-body solvers operate in the untranslated skeleton frame. Their view
matrix explicitly includes the same displacement that is committed to root
and pelvis channels. Camera animation is never used as subject motion.
"""
import numpy as np
from mathutils import Matrix, Vector


def hold_missing_motion(times,values,observed,max_gap=.3):
    """Missing observations cannot establish a return to the starting point."""
    times=np.asarray(times);values=np.asarray(values).copy();ids=np.flatnonzero(observed)
    if not len(ids):return np.zeros_like(values)
    for i in np.flatnonzero(~np.asarray(observed,bool)):
        before=ids[ids<i];after=ids[ids>i]
        if len(before) and len(after) and times[after[0]]-times[before[-1]]<=max_gap:
            a,b=before[-1],after[0];t=(times[i]-times[a])/(times[b]-times[a]);values[i]=values[a]*(1-t)+values[b]*t
        else:values[i]=values[before[-1] if len(before) else after[0]]
    return values


def fit_subject(scene,target_rig,scratch,rig,channels,data,options,cameras,report):
    from sam_ellen_screen import project,unproject
    from sam_ellen_camera_fit import reference_indices
    from sam_ellen_frame_body import relative_face_depth
    from sam_ellen_temporal import trajectory
    from sam_ellen_arm_policy import activation
    root=rig.pose.bones.get('ParentNode');pelvis=rig.pose.bones.get('Center')
    if root is None or pelvis is None or root.parent or root.constraints:
        raise ValueError('固定机位目前需要无约束的 ParentNode 根控制骨和 Center 骨盆控制骨')
    # Preserve every existing bone basis; root motion is an additive channel.
    root_base=root.location.copy();pelvis_base=pelvis.location.copy()
    root_basis=root.bone.matrix_local.to_3x3()
    pelvis_basis=pelvis.parent.matrix.to_3x3() @ pelvis.parent.bone.matrix_local.to_3x3().inverted() @ pelvis.bone.matrix_local.to_3x3()
    times=np.array([r['time'] for r in data['frames']]);faces=[r.get('face') for r in data['screen']['frames']]
    refs=reference_indices(data);ratios,reliability=relative_face_depth(times,faces,refs)
    weights=activation(times,options)
    centers=[]
    for i in range(len(times)):
        for name,values in channels.items():rig.pose.bones[name].rotation_quaternion=values[i][1]
        scratch.view_layers[0].update()
        centers.append((rig.pose.bones['Eye_L'].head+rig.pose.bones['Eye_R'].head)*.5)
    anchor=float(np.median([-(cameras[i][0]@centers[i]).z for i in refs]))
    offsets=[];targets=[];valid=[]
    near=max(.16,anchor*.18)
    for i,(face,cam,center) in enumerate(zip(faces,cameras,centers)):
        target=project(center,cam)
        if face:
            target=Vector(face['center']);target.x*=cam[2]/data['screen']['aspect']
            if options.get('mirror'):target.x=cam[2]-target.x
        depth=max(near,anchor/ratios[i])
        offsets.append(np.array(unproject(target,depth,cam)-center))
        targets.append(target);valid.append(bool(face and reliability[i]>.1))
    offsets=hold_missing_motion(times,offsets,np.asarray(valid))
    offsets=trajectory(times,offsets,.065,np.maximum(.05,reliability),robust=.03)
    # Pelvis may lower with fixed foot IK, but a cropped head observation is
    # not evidence for jumping. Horizontal root motion retains ground height.
    offsets[:,2]=np.clip(offsets[:,2],-.12,0.)
    offsets*=weights[:,None]
    loc={root.name:[],pelvis.name:[]}
    height=data.get('video_orientation',{}).get('decoded_height_width',[1280])[0]
    for i,offset in enumerate(offsets):
        frame=1+(times[i]-times[0])*data['fps'];horizontal=Vector((offset[0],offset[1],0.))
        loc[root.name].append((frame,tuple(root_base+root_basis.inverted()@horizontal)))
        loc[pelvis.name].append((frame,tuple(pelvis_base+pelvis_basis.inverted()@Vector((0.,0.,offset[2])))))
        cam=cameras[i];effective=(cam[0]@Matrix.Translation(Vector(offset)),*cam[1:])
        cameras[i]=effective
        report.append({'time':float(times[i]),'model':'fixed-camera-subject-v2',
            'root_offset':list(horizontal),'pelvis_height':float(offset[2]),
            'relative_face_scale':float(ratios[i]),'confidence':float(reliability[i]),
            'after_center_px':float((project(centers[i],effective)-targets[i]).length*height),
            'source_height':height,'feet_policy':'grounded translation; no inferred walking cycle'})
    return loc


def fixed_camera_check(scene,data):
    """Verify the complete presentation hierarchy, not only its driver."""
    import bpy
    from sam_ellen_sequence import timeline_frame
    prior=scene.frame_current;sub=scene.frame_subframe;matrices=[];optics=[]
    try:
        for sample in data['frames']:
            f=timeline_frame(data,sample['time']);scene.frame_set(int(f),subframe=f-int(f))
            camera=scene.camera.evaluated_get(bpy.context.evaluated_depsgraph_get())
            matrices.append(np.array(camera.matrix_world));optics.append([camera.data.lens,camera.data.shift_x,camera.data.shift_y])
        return {'transform_max_delta':float(np.max(abs(np.array(matrices)-matrices[0]))),
                'optics_max_delta':float(np.max(abs(np.array(optics)-optics[0])))}
    finally:scene.frame_set(prior,subframe=sub)


def ground_deformed_feet(scene,rig,data,reference):
    """Keep the rendered MMD ankle at ground height, not only its IK target.

    MMD's additional-rotation deform chain may diverge slightly from its IK
    control chain during a crouch. Correct that measured residual through the
    existing foot IK; never scale a leg or move the upper-body solution.
    """
    import bpy
    from sam_ellen_sequence import timeline_frame,curves
    action=rig.animation_data.action;report=[]
    for sample in data['frames']:
        frame=timeline_frame(data,sample['time']);scene.frame_set(int(frame),subframe=frame-int(frame))
        for side,height in reference.items():
            ctrl=rig.pose.bones.get('LegIK_'+side);deform='AnkleD_'+side
            if ctrl is None or deform not in rig.pose.bones:continue
            initial=ctrl.location.copy();remaining=0.
            for iteration in range(8):
                ev=rig.evaluated_get(bpy.context.evaluated_depsgraph_get())
                remaining=float(height-ev.pose.bones[deform].head.z)
                if abs(remaining)<.00015:break
                desired=ctrl.matrix.copy();desired.translation.z+=float(np.clip(remaining,-.012,.012))
                kwargs={'parent_matrix':ctrl.parent.matrix,'parent_matrix_local':ctrl.parent.bone.matrix_local} if ctrl.parent else {}
                local=ctrl.bone.convert_local_to_pose(desired,ctrl.bone.matrix_local,invert=True,**kwargs)
                ctrl.location=local.translation
                if (ctrl.location-initial).length>.06:
                    raise ValueError('脚部落地修正超过保守范围，请检查腿部 IK 与变形骨绑定')
                ctrl.keyframe_insert('location',frame=frame)
                bpy.context.view_layer.update()
            if abs(remaining)>.002:
                raise ValueError('实际脚部变形未能保持地面高度，请检查腿部绑定')
            report.append({'frame':frame,'side':side,'adjustment':list(ctrl.location-initial),'height_error':remaining})
    for c in curves(action):
        if c.data_path in {rig.pose.bones['LegIK_'+side].path_from_id('location') for side in reference}:
            for point in c.keyframe_points:point.interpolation='LINEAR'
    return report

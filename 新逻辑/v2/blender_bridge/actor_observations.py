"""Dense facial geometry for root placement, independent of iris/gaze motion."""
import numpy as np


def dense_anchors(face_path,body_uv,width,height,fps=30.):
    data=np.load(face_path);p=np.asarray(data['landmarks'],float)[:,:,:2]*[width,height]
    if len(p)!=len(body_uv):raise ValueError('Face/root observations have different frame counts')
    valid=np.asarray(data['valid'],bool)&np.isfinite(p).all(axis=(1,2))
    eyes=np.stack([p[:,[362,263]].mean(1),p[:,[33,133]].mean(1)],1)
    # Match semantic sides using observed coordinates, not a mirrored-video assumption.
    ids=np.flatnonzero(valid)[:max(1,int(fps*.5))]
    if len(ids)<3:return np.asarray(body_uv).copy(),{'dense_face_used':False}
    if np.median(np.linalg.norm(eyes[ids]-body_uv[ids][:,[1,2]],axis=2))>np.median(np.linalg.norm(eyes[ids,::-1]-body_uv[ids][:,[1,2]],axis=2)):
        eyes=eyes[:,::-1]
    center=eyes.mean(1);span=np.linalg.norm(eyes[:,0]-eyes[:,1],axis=1)
    base_center=body_uv[:,[1,2]].mean(1);base_span=np.linalg.norm(body_uv[:,1]-body_uv[:,2],axis=1)
    # A single calibration preserves the accepted initial framing. Per-frame
    # calibration would reintroduce the body detector's scale fluctuations.
    ratio=float(np.clip(np.median(base_span[ids]/np.maximum(span[ids],1)),.7,1.3))
    offset=np.median(base_center[ids]-center[ids],axis=0)
    eyes=(eyes-center[:,None])*ratio+center[:,None]+offset
    nose=p[:,1]+np.median(body_uv[ids,0]-p[ids,1],axis=0)
    result=np.asarray(body_uv).copy()
    target=np.stack([nose,eyes[:,0],eyes[:,1]],1)
    # Short gaps are interpolated; longer gaps blend to body observations.
    supported=valid.copy();good=np.flatnonzero(valid)
    for a,b in zip(good[:-1],good[1:]):
        if b-a<=max(2,int(fps*.2)):supported[a:b+1]=True
    for j in range(3):
        for d in range(2):target[:,j,d]=np.interp(np.arange(len(p)),good,target[good,j,d])
    # Smooth support, not the pose itself, so a detector dropout cannot snap.
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'camera'))
    from joint_motion import smooth
    weight=np.clip(smooth(supported.astype(float),fps,.08),0,1)
    result[:,:3]=result[:,:3]*(1-weight[:,None,None])+target*weight[:,None,None]
    return result,{'dense_face_used':True,'valid_fraction':float(valid.mean()),'fixed_eye_span_ratio':ratio,
                   'fixed_center_offset_px':offset.tolist(),'iris_drives_body':False}


def refit_final_actor(scene,rig,uv,camera):
    """Solve after final head pose, in the shot camera before user navigation.

    This preserves the editable navigation offset and anatomical bone lengths.
    """
    import bpy
    from mathutils import Vector
    from hand_refinement import intrinsics
    from calibrated_bridge import fit_translation
    original=scene.frame_current;shifts=[]
    for t in range(len(uv)):
        scene.frame_set(t+1);bpy.context.view_layer.update()
        R=np.array(camera.matrix_world.to_3x3())@np.diag([1.,-1.,-1.])
        origin=np.array(camera.matrix_world.translation)
        points=np.array([rig.matrix_world@rig.pose.bones[n].head for n in ['eyeLf03Joint','eyeRt03Joint','faceMdNoseJoint']])
        pc=(points-origin)@R
        shift_cv=fit_translation(pc,uv[t,[1,2,0]],intrinsics(scene,camera),np.array([1.,1.,.25]))
        if np.min(pc[:,2]+shift_cv[2])<=.025:raise ValueError('Root face fit lies behind shot camera')
        shift=R@shift_cv
        local=Vector(shift)
        if rig.parent:local=(rig.parent.matrix_world@rig.matrix_parent_inverse).to_3x3().inverted()@local
        rig.location+=local;rig.keyframe_insert(data_path='location',frame=t+1);shifts.append(shift)
    scene.frame_set(original)
    return {'stage':'after final head/torso pose; before user camera offset and hands',
            'max_world_correction':float(np.linalg.norm(shifts,axis=1).max())}

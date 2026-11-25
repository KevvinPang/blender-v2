"""Stabilize the actor and focal/depth pair in the final camera's image space.

Filtering world translation alone breaks camera compensation. Denoise residual
motion relative to reliable background tracks in one final pass, after dense
facial placement and before hand fitting. Preserve measured global image motion,
the editable navigation camera, bone lengths and facial expression curves.
"""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'camera'))
from joint_motion import stabilize_relative


def smooth_trajectory(values,fps,seconds=.11):
    from temporal_denoise import intercept
    raw=np.asarray(values,float);out=raw.copy()
    sigma=max(.75,fps*seconds);radius=int(np.ceil(sigma*3))
    for t in range(len(raw)):
        ids=np.arange(max(0,t-radius),min(len(raw),t+radius+1));dt=ids-t
        w=np.exp(-.5*(dt/sigma)**2)
        out[t]=intercept(dt,raw[ids],w)
    return out


def stabilize_actor_projection(scene,rig,count,background_path=None,*,image_scale=1.,image_offset=(0.,0.),eye_target=None,eye_valid=None):
    import bpy
    from mathutils import Vector
    from hand_refinement import intrinsics
    action=rig.animation_data.action
    if action.get('v2_actor_projection_stability')==2:return {'already_applied':True}
    fps=scene.render.fps/scene.render.fps_base
    samples=[];frames=[];eyes=[];original=scene.frame_current
    for t in range(count):
        scene.frame_set(t+1);bpy.context.view_layer.update()
        cam=scene.camera;R=np.array(cam.matrix_world.to_3x3())@np.diag([1.,-1.,-1.])
        p=np.array(rig.matrix_world@rig.pose.bones['UpperBody2'].head)
        cv=R.T@(p-np.array(cam.matrix_world.translation));K=intrinsics(scene,cam)
        if cv[2]<=.025:raise ValueError('Torso anchor is behind the active camera')
        focal=np.sqrt(K[0,0]*K[1,1]);uv=cv[:2]/cv[2]*[K[0,0],K[1,1]]+K[:2,2]
        samples.append(np.r_[uv,np.log(focal/cv[2])])
        frames.append((cv,R,K,np.array(rig.location)))
        if eye_target is not None:
            eye=np.mean([np.array(rig.matrix_world@rig.pose.bones[name].head)
                for name in ('eyeLf03Joint','eyeRt03Joint')],axis=0)
            eyes.append(R.T@(eye-np.array(cam.matrix_world.translation)))
    raw=np.array(samples)
    background=None;confidence=None
    if background_path:
        data=np.load(background_path)
        if tuple(data['image_size_hw'])!=(scene.render.resolution_y,scene.render.resolution_x):
            raise ValueError('Background image dimensions differ from the render')
        background=data['image_from_reference'];confidence=data['confidence']
        mapping=np.eye(3);mapping[0,0]=mapping[1,1]=image_scale;mapping[:2,2]=image_offset
        background=mapping[None]@background@np.linalg.inv(mapping)[None]
    filtered,relative_report=stabilize_relative(raw,fps,background,confidence)
    delta=filtered-raw
    # Restrict presentation changes while retaining slow/constant-speed motion.
    length=np.linalg.norm(delta[:,:2],axis=1)
    delta[:,:2]*=np.minimum(1.,scene.render.resolution_y*.035/np.maximum(length,1e-9))[:,None]
    delta[:,2]=np.clip(delta[:,2],-.075,.075);filtered=raw+delta
    correction=np.zeros((count,2))
    if eye_target is not None:
        error=np.empty((count,2))
        for t,(cv,R,K,loc) in enumerate(frames):
            focal=K[[0,1],[0,1]]
            z=np.sqrt(focal.prod())/np.exp(filtered[t,2])
            chest=np.r_[(filtered[t,:2]-K[:2,2])/focal*z,z]
            eye=eyes[t]+chest-cv
            error[t]=eye_target[t]-(eye[:2]/eye[2]*focal+K[:2,2])
        good=np.flatnonzero(eye_valid & np.isfinite(error).all(1))
        if len(good)>2:
            residual=np.column_stack([np.interp(np.arange(count),good,error[good,k]) for k in (0,1)])
            # Preserve the composition offset and slow framing, correct the
            # residual motion after camera and body stabilization are composed.
            correction=(residual-smooth_trajectory(residual,fps,.60))*np.clip(
                smooth_trajectory(np.asarray(eye_valid,float)[:,None],fps,.08),0,1)
            correction*=np.minimum(1.,scene.render.resolution_y*.04/np.maximum(np.linalg.norm(correction,axis=1),1e-8))[:,None]
    world_changes=[]
    for t,(cv,R,K,loc) in enumerate(frames):
        scene.frame_set(t+1);bpy.context.view_layer.update()
        z=np.sqrt(K[0,0]*K[1,1])/np.exp(filtered[t,2])
        target=np.r_[(filtered[t,:2]-K[:2,2])/[K[0,0],K[1,1]]*z,z]
        if eye_target is not None:
            target[:2]+=correction[t]*(eyes[t][2]+z-cv[2])/K[[0,1],[0,1]]
        shift=R@(target-cv)
        if rig.parent:
            parent=rig.parent.matrix_world@rig.matrix_parent_inverse
            local_shift=parent.to_3x3().inverted()@Vector(shift)
        else:local_shift=Vector(shift)
        rig.location=Vector(loc)+local_shift;rig.keyframe_insert(data_path='location',frame=t+1)
        world_changes.append(shift)
    action['v2_actor_projection_stability']=2
    scene.frame_set(original)
    def acceleration(x):return float(np.sqrt(np.mean(np.diff(x[:,:2],n=2,axis=0)**2)))
    return {'before_acceleration_rms_px':acceleration(raw),'after_acceleration_rms_px':acceleration(filtered),
        'max_center_change_px':float(np.linalg.norm(delta[:,:2],axis=1).max()),
        'max_scale_change_fraction':float(abs(np.exp(delta[:,2])-1).max()),
        'max_world_translation_change':float(np.linalg.norm(world_changes,axis=1).max()),
        'camera_unchanged':True,'filter_seconds':.11,**relative_report}

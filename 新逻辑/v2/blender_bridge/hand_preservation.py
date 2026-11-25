"""Preserve an accepted take's hand presentation during body-only refinement.

The caller must verify identical source observations before using a reference.
Finger articulation is copied, wrist presentation is retained, and the two
arm segments are solved to it without stretching either bone.
"""
import numpy as np


def observations_match(previous,current):
    if previous.keys()!=current.keys():return False
    for side in current:
        if int(previous[side].get('solver_revision',0))!=int(current[side].get('solver_revision',0)):return False
        for key in ('uv','world','valid','observed'):
            a,b=previous[side][key],current[side][key]
            if a.shape!=b.shape or not np.allclose(a,b,atol=1e-7,rtol=0,equal_nan=True):return False
    return True


def capture(scene,rig,count):
    import bpy
    from hand_refinement import intrinsics,STEMS
    action=rig.animation_data.action;frame=scene.frame_current
    names=[f'{stem}{k if stem=="Thumb" else k+1}_{side}' for side in ('L','R') for stem,_ in STEMS for k in range(3)]
    rows=[];q=[];cams=[];Ks=[];wr=[]
    try:
        rig.animation_data.action=rig.get('mocap_v2_base_action',action)
        for t in range(count):
            scene.frame_set(t+1);bpy.context.view_layer.update()
            cams.append(np.array(scene.camera.matrix_world));Ks.append(intrinsics(scene,scene.camera))
            rows.append([[list(rig.matrix_world@rig.pose.bones[n+'_'+side].head) for n in ('Arm','Elbow','Wrist')] for side in ('L','R')])
            wr.append([np.array(rig.matrix_world.to_3x3()@rig.pose.bones['Wrist_'+side].matrix.to_3x3()) for side in ('L','R')])
            q.append([list(rig.pose.bones[n].rotation_quaternion) for n in names])
    finally:
        rig.animation_data.action=action;scene.frame_set(frame);bpy.context.view_layer.update()
    return dict(camera=np.array(cams),K=np.array(Ks),arm_points=np.array(rows),wrist_rotation=np.array(wr),finger_names=np.array(names),finger_q=np.array(q))


def restore(scene,rig,reference,tracks):
    import bpy
    from mathutils import Matrix,Quaternion
    from hand_refinement import intrinsics,between,unit
    from calibrated_bridge import key_quat
    previous={};errors=[];limited=[]
    for t in range(len(reference['camera'])):
        scene.frame_set(t+1);bpy.context.view_layer.update();cam=scene.camera
        # Also key unseen frames. Leaving gaps lets newly inserted keys change
        # interpolation through entry/exit, despite identical visible poses.
        for name,q in zip(reference['finger_names'],reference['finger_q'][t]):
            key_quat(rig.pose.bones[name],Quaternion(q),t+1,previous)
        C=np.array(cam.matrix_world);R=C[:3,:3]@np.diag([1.,-1.,-1.]);K=intrinsics(scene,cam)
        C0=reference['camera'][t];R0=C0[:3,:3]@np.diag([1.,-1.,-1.]);K0=reference['K'][t]
        for side_index,side in enumerate(('L','R')):
            fade=max(4,round(scene.render.fps/scene.render.fps_base*.4))
            if not tracks[side]['valid'][max(0,t-fade):t+fade+1].any():continue
            pts=(reference['arm_points'][t,side_index]-C0[:3,3])@R0
            scale=np.sqrt(K[0,0]*K[1,1]/(K0[0,0]*K0[1,1]))
            uv=pts[:,:2]/pts[:,2:]*K0[[0,1],[0,1]]+K0[:2,2]
            depth=pts[:,2]*scale
            cv=np.c_[(uv-K[:2,2])/K[[0,1],[0,1]]*depth[:,None],depth]
            desired=C[:3,3]+cv@R.T
            A,E,W=[rig.pose.bones[n+'_'+side] for n in ('Arm','Elbow','Wrist')]
            a,e,w=[np.array(rig.matrix_world@b.head) for b in (A,E,W)]
            L1,L2=np.linalg.norm(e-a),np.linalg.norm(w-e)
            goal=desired[2];direction=unit(goal-a);raw_distance=np.linalg.norm(goal-a)
            distance=np.clip(raw_distance,abs(L1-L2)+1e-5,L1+L2-1e-5)
            goal=a+direction*distance
            pole=desired[1]-a;pole-=direction*(pole@direction)
            if np.linalg.norm(pole)<1e-7:pole=e-a;pole-=direction*(pole@direction)
            pole=unit(pole);along=(L1*L1-L2*L2+distance*distance)/(2*distance)
            elbow=a+direction*along+pole*np.sqrt(max(0,L1*L1-along*along))
            invR=np.array(rig.matrix_world.to_3x3().inverted())
            AR=between(e-a,elbow-a)@np.array(rig.matrix_world.to_3x3()@A.matrix.to_3x3())
            ER=between(w-e,goal-elbow)@np.array(rig.matrix_world.to_3x3()@E.matrix.to_3x3())
            WR=R@R0.T@reference['wrist_rotation'][t,side_index]
            for bone,D in ((A,invR@AR),(E,invR@ER),(W,invR@WR)):
                inherited=bone.parent.matrix.to_3x3()@bone.parent.bone.matrix_local.to_3x3().inverted()@bone.bone.matrix_local.to_3x3()
                key_quat(bone,(inherited.inverted()@Matrix(D)).to_quaternion(),t+1,previous);bpy.context.view_layer.update()
            bpy.context.view_layer.update();actual=(np.array(rig.matrix_world@W.head)-C[:3,3])@R
            output=actual[:2]/actual[2]*K[[0,1],[0,1]]+K[:2,2]
            errors.append(float(np.linalg.norm(output-uv[2])));limited.append(abs(raw_distance-distance)>1e-5)
    return {'method':'accepted hand camera-space pose; fixed bone lengths; unchanged finger articulation',
            'max_wrist_projection_change_px':float(max(errors,default=0)),
            'p95_wrist_projection_change_px':float(np.percentile(errors,95)) if errors else 0.,
            'reach_limited_frames':int(sum(limited))}

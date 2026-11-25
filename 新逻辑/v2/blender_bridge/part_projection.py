"""Place the final articulated hand on its measured image rays at a frame edge."""
import bpy
import numpy as np
from mathutils import Matrix
from calibrated_bridge import key_quat
from hand_refinement import STEMS, between, intrinsics, unit


def align_parts(scene, rig, tracks):
    previous={}
    for side,track in tracks.items():
        anchors=track.get('part_anchor')
        if anchors is None:continue
        partial=np.flatnonzero(anchors>=0)
        if not len(partial):continue
        arm,elbow,wrist=[rig.pose.bones[n+'_'+side] for n in ('Arm','Elbow','Wrist')]
        digits=[[rig.pose.bones[f'{stem}{j if stem=="Thumb" else j+1}_{side}'] for j in range(3)] for stem,_ in STEMS]
        for t in range(len(anchors)):
            near=np.min(abs(partial-t))
            if near>3 or not track['valid'][t]:continue
            uv=track['uv'][t]
            wt=track['digit_weights'][t].copy() if near==0 else ((uv>=0)&(uv<=1)).all(1).astype(float)
            if not wt.any() or (near and not track['observed'][t]):continue
            scene.frame_set(t+1)
            K=intrinsics(scene,scene.camera)
            size=np.array([scene.render.resolution_x,scene.render.resolution_y])
            C=np.array(scene.camera.matrix_world)
            R=C[:3,:3]@np.diag([1.,-1.,-1.])
            def camera(p):
                return (np.array(rig.matrix_world@p)-C[:3,3])@R
            pose=[wrist.head]
            for chain in digits:pose.extend([b.head for b in chain]+[chain[-1].tail])
            pos=np.array([camera(p) for p in pose])
            offset=pos-pos[0]
            tips=np.array([4,8,12,16,20])
            separation=np.linalg.norm((uv[tips,None]-uv[tips])*size,axis=2)
            np.fill_diagonal(separation,np.inf)
            isolated=tips[(separation.min(1)>.09*min(size)) & (wt[tips]>0)]
            wt[isolated]*=30
            rays=(uv*size-K[:2,2])/K[[0,1],[0,1]]
            a,e,w=[camera(b.head) for b in (arm,elbow,wrist)]
            upper,lower=np.linalg.norm(e-a),np.linalg.norm(w-e)
            current=np.linalg.norm(w-a)
            # Preserve the current bent-arm radius when the desired ray permits it.
            z=np.linspace(max(.05,w[2]-.5*(upper+lower)),w[2]+.5*(upper+lower),121)
            xy=np.average(rays[None,:,:]*(z[:,None,None]+offset[None,:,2:])-offset[None,:,:2],axis=1,weights=wt)
            goals=np.column_stack([xy,z])
            distance=np.linalg.norm(goals-a,axis=1)
            score=abs(distance-current)+.18*abs(z-w[2])+40*np.maximum(0,distance-.975*(upper+lower))
            goal=w+(goals[score.argmin()]-w)*(1-near/4)*track['confidence'][t]
            direction=unit(goal-a)
            reach=np.clip(np.linalg.norm(goal-a),abs(upper-lower)+1e-5,.975*(upper+lower))
            goal=a+direction*reach
            pole=e-a-direction*((e-a)@direction)
            if np.linalg.norm(pole)<1e-6:continue
            along=(upper**2-lower**2+reach**2)/(2*reach)
            bend=a+direction*along+unit(pole)*np.sqrt(max(0,upper**2-along**2))
            rotations=[np.array(rig.matrix_world.to_3x3()@b.matrix.to_3x3()) for b in (arm,elbow,wrist)]
            wanted=[between(R@(e-a),R@(bend-a))@rotations[0],
                    between(R@(w-e),R@(goal-bend))@rotations[1],rotations[2]]
            for bone,rotation in zip((arm,elbow,wrist),wanted):
                bind=bone.parent.matrix.to_3x3()@bone.parent.bone.matrix_local.to_3x3().inverted()@bone.bone.matrix_local.to_3x3()
                q=(bind.inverted()@rig.matrix_world.to_3x3().inverted()@Matrix(rotation)).to_quaternion()
                key_quat(bone,q,t+1,previous)
                bpy.context.view_layer.update()

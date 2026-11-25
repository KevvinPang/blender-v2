"""Match the visible extent of a departing hand in the final editable camera."""
import numpy as np


def blend_contour_handoff(scene, rig, side, track, frames, key_quat):
    """Join the fitted and contour-driven poses across a confirmed exit."""
    from mathutils import Quaternion
    from temporal_denoise import smooth_rotations
    boundary=track['boundary']
    selected=np.zeros(len(boundary),bool)
    selected[frames]=True
    ids=np.flatnonzero(np.isfinite(boundary)&selected)
    fps=scene.render.fps/scene.render.fps_base
    radius=max(2,round(fps*.10))
    bones=[rig.pose.bones[name+'_'+side] for name in ('Arm','Elbow','Wrist')]
    previous={}
    for run in np.split(ids,np.flatnonzero(np.diff(ids)>1)+1):
        if len(run)<3:continue
        first,last=run[0],run[-1]
        left=first
        while left>0 and first-left<2*radius and selected[left-1] and track['valid'][left-1] and not np.isfinite(boundary[left-1]):
            left-=1
        rows=[]
        for t in range(left,last+1):
            scene.frame_set(int(t+1))
            rows.append([list(b.rotation_quaternion) for b in bones])
        rows=np.array(rows)
        for j,b in enumerate(bones):
            clean=smooth_rotations(rows[:,j],fps,limit_degrees=24.,range_degrees=60.)
            for k,t in enumerate(range(left,last+1)):
                weight=min(1.,k/radius,(last-t)/radius)
                q=Quaternion(rows[k,j].tolist()).slerp(Quaternion(clean[k].tolist()),weight)
                key_quat(b,q,t+1,previous)


def apply_contours(scene,rig,tracks,frames):
    import bpy
    from mathutils import Matrix, Quaternion
    from calibrated_bridge import key_quat
    from hand_refinement import intrinsics,between,unit,STEMS
    previous={};report=[]
    for side,track in tracks.items():
        if 'boundary' not in track:continue
        arm,elbow,wrist=[rig.pose.bones[n+'_'+side] for n in ('Arm','Elbow','Wrist')]
        fingers=[rig.pose.bones[f'{stem}{i if stem=="Thumb" else i+1}_{side}'] for stem,_ in STEMS for i in range(3)]
        prev_t=-2
        for t in frames:
            target=track['boundary'][t]
            if not np.isfinite(target):continue
            scene.frame_set(t+1);bpy.context.view_layer.update()
            cam=scene.camera;K=intrinsics(scene,cam);C=np.array(cam.matrix_world)
            R=C[:3,:3]@np.diag([1.,-1.,-1.]);size=np.array([scene.render.resolution_x,scene.render.resolution_y])
            def cv(point):
                return (np.array(rig.matrix_world@point)-C[:3,3])@R
            def project(points):
                return (points[:,:2]/points[:,2:]*K[[0,1],[0,1]]+K[:2,2])/size
            side_id=int(track['edge'][t]);axis=side_id//2;sign=(-1,1)[side_id%2]
            if t!=prev_t+1:
                wrist_camera=R.T@np.array(rig.matrix_world.to_3x3()@wrist.matrix.to_3x3())
                finger_pose=[b.rotation_quaternion.copy() for b in fingers]
            elif track['observed'][t]:
                from hand_visibility import inside
                for j,(_,start) in enumerate(STEMS):
                    if inside(track['uv'][t,start:start+4]).all():
                        finger_pose[3*j:3*j+3]=[b.rotation_quaternion.copy() for b in fingers[3*j:3*j+3]]
            prev_t=t
            for b,q in zip(fingers,finger_pose):key_quat(b,q.copy(),t+1,previous)
            def orient(b,world):
                inherited=b.parent.matrix.to_3x3()@b.parent.bone.matrix_local.to_3x3().inverted()@b.bone.matrix_local.to_3x3()
                q=(inherited.inverted()@rig.matrix_world.to_3x3().inverted()@Matrix(world)).to_quaternion()
                key_quat(b,q,t+1,previous);bpy.context.view_layer.update()
            orient(wrist,R@wrist_camera)
            def points():
                return np.array([cv(wrist.head)]+[cv(b.head) for b in fingers]+[cv(b.tail) for b in fingers[2::3]])
            pts=points();offsets=pts-pts[0];i=np.argmin(sign*project(pts)[:,axis])
            a,e,w=[cv(b.head) for b in (arm,elbow,wrist)]
            upper,lower=np.linalg.norm(e-a),np.linalg.norm(w-e)
            # Search a feasible depth for a bent two-link arm. Keep the
            # current depth unless reach actually requires changing it.
            best=None
            for depth in np.linspace(max(.065,w[2]-.25),w[2]+.25,101):
                goal=w*(depth/max(.025,w[2]))
                goal[axis]=(target*size[axis]-K[axis,2])/K[axis,axis]*(depth+offsets[i,2])-offsets[i,axis]
                distance=np.linalg.norm(goal-a)
                score=abs(depth-w[2])+.04*abs(distance-np.linalg.norm(w-a))+100*max(0.,distance-(upper+lower)*.985,abs(upper-lower)+.005-distance)
                if best is None or score<best[0]:best=score,goal
            goal=best[1];direction=unit(goal-a)
            distance=np.clip(np.linalg.norm(goal-a),abs(upper-lower)+1e-5,(upper+lower)*.985)
            goal=a+direction*distance;pole=e-a;pole-=direction*(pole@direction)
            if np.linalg.norm(pole)<1e-7:pole=np.cross(direction,np.eye(3)[np.argmin(abs(direction))])
            along=(upper**2-lower**2+distance**2)/(2*distance)
            new_elbow=a+along*direction+unit(pole)*np.sqrt(max(0,upper**2-along**2))
            rotations=[np.array(rig.matrix_world.to_3x3()@b.matrix.to_3x3()) for b in (arm,elbow)]
            orient(arm,between(R@(e-a),R@(new_elbow-a))@rotations[0])
            orient(elbow,between(R@(w-e),R@(goal-new_elbow))@rotations[1])
            orient(wrist,R@wrist_camera)
            uv=project(points());actual=float(sign*np.min(sign*uv[:,axis]))
            report.append((int(t+1),float(target),actual))
            if t+1<len(track['boundary']) and not np.isfinite(track['boundary'][t+1]):
                # Even outside frame the forearm or sleeve may remain visible.
                # Release over a short interval, stopping at the next observation.
                bones=[arm,elbow,wrist,*fingers]
                departed=[b.rotation_quaternion.copy() for b in bones]
                for k in range(1,7):
                    index=t+k
                    if index>=len(track['valid']) or track['observed'][index] or np.isfinite(track['boundary'][index]):break
                    scene.frame_set(index+1);bpy.context.view_layer.update()
                    u=k/6.;u=u*u*(3-2*u)
                    for b,q in zip(bones,departed):
                        key_quat(b,q.slerp(b.rotation_quaternion,u),index+1,previous)
        blend_contour_handoff(scene,rig,side,track,frames,key_quat)
    return report

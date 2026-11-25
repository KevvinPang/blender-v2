"""Apply finger segment directions in the current wrist frame."""
import bpy
import numpy as np
from mathutils import Matrix, Vector
from hand_refinement import HandModel, palm


def apply_hamer_pose(scene, rig, points, valid, tracks=None):
    if points.shape != (scene.frame_end, 2, 21, 3) or valid.shape != points.shape[:2]:
        raise ValueError('HaMeR frame count does not match the animation')
    if not np.isfinite(points[valid]).all():
        raise ValueError('Invalid HaMeR hand geometry')
    points=points.copy()
    valid=valid.copy()
    blend=np.ones(valid.shape)
    for i,side in enumerate(('L','R')):
        anchors=(tracks or {}).get(side,{}).get('part_anchor')
        if anchors is None:continue
        original=valid[:,i].copy()
        for t in np.flatnonzero(anchors>=0):
            a=anchors[t]
            if original[a]:
                points[t,i]=points[a,i]
                valid[t,i]=True
        duration=max(2,round(scene.render.fps/scene.render.fps_base*.16))
        ends=np.flatnonzero(valid[:,i] & ~np.r_[valid[1:,i],True])
        for end in ends:
            if not np.any(anchors[max(0,end-duration):end+1]>=0):continue
            for dt in range(1,duration):
                t=end+dt
                if t>=len(valid) or valid[t,i]:break
                u=dt/duration
                blend[t,i]=1-u*u*(3-2*u)
                points[t,i]=points[end,i]
                valid[t,i]=True
    models = {i:HandModel(rig, side) for i, side in enumerate(('L','R')) if valid[:, i].any()}
    for t in np.flatnonzero(valid.any(axis=1)):
        scene.frame_set(int(t+1))
        for side, model in models.items():
            if not valid[t,side]:
                continue
            wrist = rig.pose.bones['Wrist_'+model.side]
            rotation = np.asarray(wrist.matrix.to_3x3().normalized())
            source = points[t,side]
            alignment = rotation @ model.wrist_rest.T @ model.rest_palm @ palm(source).T
            for name, (a,b) in zip(model.names, model.segments):
                bone = rig.pose.bones[name]
                target = Vector(alignment @ (source[b]-source[a]))
                swing = (bone.tail-bone.head).rotation_difference(target)
                mat = bone.matrix.copy()
                previous = bone.rotation_quaternion.copy()
                bone.matrix = Matrix.LocRotScale(mat.translation, swing @ mat.to_quaternion(), mat.to_scale())
                q = bone.rotation_quaternion.copy()
                q.make_compatible(previous)
                if blend[t,side]<1.:
                    q=previous.slerp(q,blend[t,side])
                bone.rotation_quaternion = q
                bone.keyframe_insert(data_path='rotation_quaternion', frame=int(t+1))
                bpy.context.view_layer.update()
        if t % 60 == 0:
            print(f'V2_PROGRESS HaMeR 指形重定向 {t+1}/{len(valid)}', flush=True)
    return dict(zip(('L','R'), map(int,valid.sum(axis=0))))

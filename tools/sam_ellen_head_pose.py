"""Camera-relative facial pose; preserve yaw instead of aiming at the lens."""
import math

import numpy as np
from mathutils import Matrix, Quaternion, Vector


def correct_head_pose(scratch, rig, channels, data, options, cameras, report):
    from sam_ellen_retarget import gaze_bones, set_global_rotation
    from sam_ellen_temporal import quaternions
    head = gaze_bones(rig)['頭']
    faces = data.get('screen', {}).get('frames', [])
    times = np.array([r['time'] for r in data['frames']])
    poses, indices = [], []
    mirror = np.diag([-1., 1., 1.])
    for i, row in enumerate(faces):
        rotation = (row.get('face') or {}).get('rotation')
        if rotation is None:
            continue
        rotation = np.asarray(rotation, dtype=float)
        if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
                or not np.allclose(rotation.T@rotation, np.eye(3), atol=.01)
                or abs(np.linalg.det(rotation)-1) > .01):
            raise ValueError('Invalid camera-relative facial rotation')
        if options.get('mirror'):
            rotation = mirror@rotation@mirror
        poses.append(Matrix(rotation.tolist()).to_quaternion())
        indices.append(i)
    filtered = {}
    # Never smooth across a long loss of face tracking.
    for group in np.split(np.arange(len(indices)),
                          np.flatnonzero(np.diff(times[indices]) > .2)+1):
        if not len(group):
            continue
        selected = [indices[j] for j in group]
        values = quaternions(times[selected], [poses[j] for j in group], options.get('smoothing', .025))
        filtered.update(zip(selected, values))
    valid = np.array(indices, dtype=int)
    # Character forward -Y / up +Z maps to camera forward +Z / up +Y.
    basis = Matrix(((1, 0, 0), (0, 0, 1), (0, -1, 0))).to_quaternion()
    rest = head.bone.matrix_local.to_quaternion()
    previous = None
    for i, row in enumerate(data['frames']):
        for name, samples in channels.items():
            rig.pose.bones[name].rotation_quaternion = samples[i][1]
        scratch.view_layers[0].update()
        pose, weight = filtered.get(i), 1.
        if pose is None and len(valid):
            right = int(np.searchsorted(valid, i))
            if 0 < right < len(valid):
                a, b = valid[right-1], valid[right]
                if times[b]-times[a] <= .8:
                    pose = filtered[a].slerp(filtered[b], float((times[i]-times[a])/(times[b]-times[a])))
                    weight = 1.
            if pose is None:
                # Long losses release the last measured orientation gradually;
                # reacquisition also fades in instead of switching to/from SAM.
                candidates=[]
                if right>0:
                    a=valid[right-1];x=float(np.clip((times[i]-times[a]-.2)/.45,0,1))
                    candidates.append((1-x*x*(3-2*x),filtered[a]))
                if right<len(valid):
                    b=valid[right];x=float(np.clip((times[b]-times[i])/.18,0,1))
                    candidates.append((1-x*x*(3-2*x),filtered[b]))
                if candidates:
                    weight,pose=max(candidates,key=lambda item:item[0])
                    if weight<=0:pose=None
        if pose is None:
            report.append({'time': row['time'], 'active': False, 'source': 'SAM fallback'})
        else:
            desired = cameras[i][0].to_quaternion().inverted()@pose@basis@rest
            captured = head.matrix.to_quaternion()
            desired = captured.slerp(desired, weight*options.get('strength', 1.))
            set_global_rotation(head, desired)
            scratch.view_layers[0].update()
            q = head.rotation_quaternion.copy()
            if previous is not None and q.dot(previous) < 0:
                q.negate()
            previous = q.copy()
            channels[head.name][i] = channels[head.name][i][0], tuple(q)
            direction = pose@Vector((0, 0, 1))
            report.append({'time': row['time'], 'active': True, 'weight': weight,
                           'source': 'MediaPipe camera-space face rotation',
                           'yaw_degrees': math.degrees(math.atan2(direction.x, direction.z)),
                           'smoothing_seconds': options.get('smoothing', .025)})
        yield i+1, len(times)

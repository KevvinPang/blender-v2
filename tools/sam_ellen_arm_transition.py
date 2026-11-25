"""Image-owned wrist trajectories and temporally fitted elbow-circle phases."""
import math
import numpy as np
from mathutils import Vector,Quaternion


def observed_goals(scratch,rig,channels,data,side,cameras,observations,observed,protected,floors,mirror=False,image_weights=None):
    """Fit measured wrists first; filter only the ambiguous elbow phase.

    The offline fit uses future direction evidence only inside a currently
    observed hand segment. It never animates an unseen hand before detection.
    """
    from sam_ellen_screen_fit import hand_points
    from sam_ellen_motion_continuity import translation_goal
    from sam_ellen_arm_policy import camera_reachable_goal,front_circle,transport_bend
    from sam_ellen_arm_chain import torso_front_plane
    from sam_ellen_temporal import trajectory
    n=len(data['frames']);times=np.array([r['time'] for r in data['frames']]);result={};segments=[];segment=[]
    pose_rows=data['screen'].get('pose_arms',[]);view=scratch.view_layers[0]
    base=None;old_axis=None;last=None;phase=0.
    for i in range(n):
        o=observations[i]
        if not observed[i] or protected[i] or o is None or o.get('held'):
            if segment:segments.append(segment);segment=[]
            base=None;old_axis=None;last=None;phase=0.;continue
        for name,rows in channels.items():rig.pose.bones[name].rotation_quaternion=rows[i][1]
        view.update();a,e,w=[rig.pose.bones[p+'_'+side] for p in ('Arm','Elbow','Wrist')]
        q=w.matrix.to_quaternion();origin=w.head.copy();upper=(e.head-a.head).length;lower=(w.head-e.head).length
        local=np.array([q.inverted()@(p-origin) for p in hand_points(rig,side)])
        goal=translation_goal(local,q,origin,o,cameras[i],data['screen']['aspect'],mirror)
        if image_weights is not None:goal=origin.lerp(goal,float(image_weights[i]))
        plane=torso_front_plane(rig)
        goal=camera_reachable_goal(a.head,goal,upper,lower,floors[i],cameras[i],plane)
        ee,goal,bend=front_circle(a.head,goal,upper,lower,e.head-a.head,plane,floors[i])
        axis=(goal-a.head).normalized()
        base=bend.copy() if base is None else transport_bend(base,old_axis,axis)
        value=math.atan2(axis.dot(base.cross(bend)),base.dot(bend))
        if last is not None:value+=math.tau*round((phase-value)/math.tau)
        pose=pose_rows[i].get(side) if len(pose_rows)==n else None
        confidence=pose.get('confidence',0.) if pose else 0.
        result[i]={'goal':goal.copy(),'axis':axis.copy(),'base':base.copy(),'phase':value,
                   'weight':6. if confidence>=.5 else 1.}
        segment.append(i);old_axis=axis.copy();phase=value;last=i
    if segment:segments.append(segment)
    for ids in segments:
        phases=np.array([result[i]['phase'] for i in ids]);weights=np.array([result[i]['weight'] for i in ids])
        # No smoothing of the measured wrist image position, and no coupling
        # across absence/protected intervals. Only depth-ambiguous elbow roll.
        if len(ids)>2:phases=trajectory(times[ids],phases,.10,weights,robust=math.radians(35.))
        for i,value in zip(ids,phases):
            r=result[i]
            if image_weights is not None:value=r['phase']+(value-r['phase'])*float(image_weights[i])
            r['bend']=Quaternion(r['axis'],float(value))@r['base']
    return result

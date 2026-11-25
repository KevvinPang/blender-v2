"""Final arm ownership: evidence -> coupled path -> bounded curves -> twist.

The earlier image/finger fit supplies proposals. This stage alone owns arm
loss transitions, safe hinge choice, palm compatibility and temporal fitting.
Legacy projection closure, loss repair and arm stabilization do not run here.
"""
import numpy as np


def finish_arms(scene,scratch,rig,target_rig,channels,data,options,cameras,contact,punch,report):
    from sam_ellen_arm_visibility import protected_frames
    from sam_ellen_arm_policy import activation,PREFIXES
    from sam_ellen_arm_trajectory import solve_trajectory
    from sam_ellen_arm_stability import stabilize_twist
    from sam_ellen_semantic import prepare
    sides=('L','R') if options['side']=='BOTH' else (options['side'],)
    excluded={contact['side']} if contact else set()
    if punch and not punch.get('automatic'):excluded.add(punch['side'])
    sides=tuple(s for s in sides if s not in excluded)
    protected={s:protected_frames(data,s,contact,punch) for s in sides}
    allowed=activation([r['time'] for r in data['frames']],options)>0
    frames=[1+(r['time']-data['frames'][0]['time'])*data['fps'] for r in data['frames']]
    for side in sides:
        for prefix in PREFIXES:
            name=prefix+side
            if name not in rig.pose.bones:raise ValueError('整臂求解缺少骨骼：'+name)
            if name not in channels:
                q=tuple(rig.pose.bones[name].rotation_quaternion)
                channels[name]=[(frame,q) for frame in frames]
    prepare(data,options)
    from sam_ellen_gesture_layout import prepare as prepare_layout
    prepare_layout(scene,scratch,rig,channels,data,cameras,options)
    for side,done,total in solve_trajectory(scratch,rig,channels,data,cameras,options,
            protected=protected,allowed=allowed,sides=sides,report=report):yield done,total
    from sam_ellen_upper_twist import distribute
    for side,done,total in distribute(scratch,rig,channels,data,protected,allowed,sides,report):yield done,total
    for side,done,total in stabilize_twist(scratch,rig,channels,data,
            protected=protected,allowed=allowed,sides=sides,report=report):yield done,total
    from sam_ellen_finger_constraints import constrain
    report.extend(constrain(scratch,rig,channels,data,sides,smooth=True,protected=protected,allowed=allowed))
    return sides,protected,allowed

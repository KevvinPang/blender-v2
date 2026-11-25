"""Distribute humeral roll through the rig's existing upper-arm helpers."""
import math
import numpy as np
from mathutils import Quaternion


def distribute(scratch,rig,channels,data,protected,allowed,sides,report):
    from sam_ellen_retarget import set_global_rotation
    from sam_ellen_arm_stability import bounded_roll_curve
    from sam_ellen_temporal import trajectory
    from sam_ellen_arm_policy import PREFIXES
    times=np.array([r['time'] for r in data['frames']]);view=scratch.view_layers[0]
    original={name:list(rows) for name,rows in channels.items()}
    def load(i):
        for name,rows in original.items():rig.pose.bones[name].rotation_quaternion=rows[i][1]
        view.update()
    for side in sides:
        a,e,w=[rig.pose.bones[n+'_'+side] for n in ('Arm','Elbow','Wrist')]
        controller=rig.pose.bones['ArmTwist_'+side]
        locked=np.asarray(protected[side],bool)|~allowed
        rolls=[];existing=[]
        for i in range(len(times)):
            load(i);q=a.rotation_quaternion.copy()
            if q.w<0:q.negate()
            rolls.append(q.to_swing_twist('Y')[1]);q=controller.rotation_quaternion.copy()
            if q.w<0:q.negate()
            existing.append(q.to_swing_twist('Y')[1])
        rolls=np.unwrap(rolls);desired=np.clip(np.asarray(existing)+rolls*.75,-math.radians(150),math.radians(150))
        desired[locked]=np.asarray(existing)[locked]
        desired=trajectory(times,desired,.065,weights=np.where(locked,1e9,12.))
        desired[locked]=np.asarray(existing)[locked]
        fitted=bounded_roll_curve(desired,np.radians(550)*np.diff(times),locked)
        maximum=0.
        for i in range(len(times)):
            if locked[i]:continue
            load(i);ep=e.head.copy();wp=w.head.copy();er=e.matrix.to_quaternion().copy();wr=w.matrix.to_quaternion().copy()
            swing,roll=a.rotation_quaternion.to_swing_twist('Y')
            moved=float(fitted[i]-existing[i])
            a.rotation_quaternion=swing@Quaternion((0,1,0),roll-moved)
            controller.rotation_quaternion=Quaternion((0,1,0),float(fitted[i]));view.update()
            # Helper axes are not exactly collinear in this MMD bind pose.
            # Reconcile direction after distributing roll; no scale/translation.
            for _ in range(2):
                correction=(e.head-a.head).rotation_difference(ep-a.head)
                set_global_rotation(a,correction@a.matrix.to_quaternion());view.update()
            set_global_rotation(e,er);view.update();set_global_rotation(w,wr);view.update()
            maximum=max(maximum,(e.head-ep).length,(w.head-wp).length)
            if max((e.head-ep).length,(w.head-wp).length)>.0005:raise ValueError('上臂扭转分配改变了关节位置，已取消生成')
            for prefix in PREFIXES:
                name=prefix+side;channels[name][i]=channels[name][i][0],tuple(rig.pose.bones[name].rotation_quaternion)
            if i%80==0:yield side,i,len(times)
        report.append({'model':'distributed-upper-arm-roll-v1','side':side,'endpoint_max_error_m':maximum})

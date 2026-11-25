"""Final native finger joint limits and time-coherent articulation.

Positions are never edited. Flexion is fitted in each bone's rest hinge frame;
the distal joints cannot use axial roll to explain a noisy image landmark.
"""
import math
import numpy as np
from mathutils import Quaternion, Vector


def joint_frame(bone, across):
    axis=bone.bone.matrix_local.to_quaternion().inverted()@(-across)
    axis.y=0.
    axis.normalize()
    return axis,axis.cross(Vector((0,1,0))).normalized()


def parameters(q,axis,normal):
    q=Quaternion(q).normalized();direction=q@Vector((0,1,0))
    flex=math.atan2(direction.dot(normal),direction.y)
    spread=math.asin(float(np.clip(-direction.dot(axis),-1,1)))
    swing=Quaternion(axis,flex)@Quaternion(normal,spread)
    twist=(swing.inverted()@q).to_swing_twist('Y')[1]
    return np.array([flex,spread,twist])


def bounded_parameters(values,joint):
    lo=np.radians([-12 if joint==0 else -5,-25 if joint==0 else -5,-12 if joint==0 else -3])
    hi=np.radians([95 if joint==0 else 105 if joint==1 else 80,25 if joint==0 else 5,12 if joint==0 else 3])
    return np.clip(values,lo,hi)


def rotation(values,axis,normal):
    f,s,t=map(float,values)
    return Quaternion(axis,f)@Quaternion(normal,s)@Quaternion((0,1,0),t)


def constrain(scratch,rig,channels,data,sides=('L','R'),smooth=False,protected=None,allowed=None):
    from sam_ellen_articulation import finger_bones
    from sam_ellen_temporal import trajectory,quaternions
    from sam_ellen_arm_visibility import screen_hand
    from sam_ellen_motion_evidence import motion_evidence
    times=np.array([r['time'] for r in data['frames']]);report=[]
    for side in sides:
        chains=finger_bones(rig,side)
        across=(chains[1][0].bone.head_local-chains[4][0].bone.head_local).normalized()
        evidence=motion_evidence(data,side)
        reliability=np.array([(screen_hand(data.get('screen',{}),i,side) or {}).get('tracking_reliability',.5) for i in range(len(times))])
        weights=(2.+10*reliability)*(1+30*np.array([e['fast_motion'] for e in evidence]))
        locked=np.array(protected.get(side,np.zeros(len(times),bool)) if protected else np.zeros(len(times),bool))
        if allowed is not None:locked|=~np.asarray(allowed,bool)
        weights[locked]=1e9
        changed=0;maximum=0.
        for k,chain in enumerate(chains):
            for j,b in enumerate(chain):
                if b.name not in channels:continue
                samples=channels[b.name];original=[Quaternion(q) for _,q in samples]
                if k==0:
                    # Thumb opposition is not the four-finger hinge. Retain it,
                    # with a bounded local swing/twist and its own smooth curve.
                    fitted=quaternions(times,original,.065,weights) if smooth else original
                    target=[]
                    for q in fitted:
                        swing,twist=q.to_swing_twist('Y');axis,angle=swing.to_axis_angle()
                        target.append(Quaternion(axis,min(angle,math.radians(110)))@Quaternion((0,1,0),float(np.clip(twist,-.65,.65))))
                else:
                    axis,normal=joint_frame(b,across)
                    values=bounded_parameters(np.array([parameters(q,axis,normal) for q in original]),j)
                    if smooth:values=bounded_parameters(trajectory(times,values,.085,weights,robust=.25),j)
                    target=[rotation(v,axis,normal) for v in values]
                for i,(old,q) in enumerate(zip(original,target)):
                    if locked[i]:q=old.copy()
                    if i and q.dot(Quaternion(samples[i-1][1]))<0:q.negate()
                    angle=2*math.acos(float(np.clip(abs(old.normalized().dot(q.normalized())),0,1)))
                    maximum=max(maximum,math.degrees(angle));changed+=angle>1e-4
                    samples[i]=(samples[i][0],tuple(q))
        report.append({'side':side,'model':'native-finger-hinges-v1','smoothed':smooth,
                       'changed_joint_samples':int(changed),'max_correction_degrees':maximum})
    return report


def measure(rig,side):
    from sam_ellen_articulation import finger_bones
    chains=finger_bones(rig,side);bends=[];twists=[]
    for chain in chains[1:]:
        vectors=[(b.tail-b.head).normalized() for b in chain]
        bends.append([math.degrees(vectors[j].angle(vectors[j+1])) for j in (0,1)])
        twists.extend(abs(math.degrees(b.rotation_quaternion.to_swing_twist('Y')[1])) for b in chain[1:])
    return {'pip_max_degrees':max(v[0] for v in bends),'dip_max_degrees':max(v[1] for v in bends),
            'distal_twist_max_degrees':max(twists),
            'invalid':bool(any(v[0]>120 or v[1]>100 for v in bends) or max(twists)>25)}

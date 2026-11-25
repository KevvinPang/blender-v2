"""Bounded, symmetric denoising: no hold-last-frame state and no causal lag."""
import numpy as np


def multiply(a,b):
    return np.concatenate((a[...,:1]*b[...,:1]-(a[...,1:]*b[...,1:]).sum(-1,keepdims=True),
        a[...,:1]*b[...,1:]+b[...,:1]*a[...,1:]+np.cross(a[...,1:],b[...,1:])),axis=-1)


def inverse(q):
    result=q.copy();result[...,1:]*=-1
    return result


def log_rotation(q):
    q=np.where((q[...,:1]<0),-q,q)
    v=q[...,1:];length=np.linalg.norm(v,axis=-1,keepdims=True)
    angle=2*np.arctan2(length,np.clip(q[...,:1],0,1))
    return v*angle/np.maximum(length,1e-12)


def exp_rotation(v):
    angle=np.linalg.norm(v,axis=-1,keepdims=True)
    return np.concatenate((np.cos(angle/2),v*np.sin(angle/2)/np.maximum(angle,1e-12)),axis=-1)


def intercept(offsets,values,weights):
    """Local-linear fit preserves constant-speed motion, even at clip ends."""
    s0=weights.sum();s1=(weights*offsets).sum();s2=(weights*offsets**2).sum()
    y0=(values*weights[:,None]).sum(0)
    y1=(values*(weights*offsets)[:,None]).sum(0)
    determinant=s0*s2-s1*s1
    return y0/s0 if determinant<1e-10 else (s2*y0-s1*y1)/determinant


def smooth_rotations(values,fps=30.,limit_degrees=6.,range_degrees=18.):
    raw=np.asarray(values,dtype=float).copy()
    raw/=np.linalg.norm(raw,axis=-1,keepdims=True)
    output=raw.copy();sigma=max(.75,fps*.085);radius=int(np.ceil(sigma*3))
    maximum=np.radians(limit_degrees);range_angle=np.radians(range_degrees)
    for t in range(len(raw)):
        indices=np.arange(max(0,t-radius),min(len(raw),t+radius+1));dt=indices-t
        delta=log_rotation(multiply(inverse(raw[t]),raw[indices]))
        weights=np.exp(-.5*(dt/sigma)**2-.5*(np.linalg.norm(delta,axis=1)/range_angle)**2)
        correction=intercept(dt,delta,weights)
        correction*=min(1.,maximum/max(np.linalg.norm(correction),1e-12))
        output[t]=multiply(raw[t],exp_rotation(correction))
    for t in range(1,len(output)):
        if output[t-1]@output[t]<0:output[t]*=-1
    return output


def smooth_positions(values,fps=30.,max_offset=.012):
    raw=np.asarray(values,dtype=float);result=raw.copy()
    sigma=max(.75,fps*.085);radius=int(np.ceil(sigma*3))
    for t in range(len(raw)):
        indices=np.arange(max(0,t-radius),min(len(raw),t+radius+1));dt=indices-t
        delta=raw[indices]-raw[t]
        weights=np.exp(-.5*(dt/sigma)**2-.5*(np.linalg.norm(delta,axis=1)/.035)**2)
        correction=intercept(dt,delta,weights)
        correction*=min(1.,max_offset/max(np.linalg.norm(correction),1e-12))
        result[t]=raw[t]+correction
    return result


def angular_jitter(q):
    q=np.asarray(q);q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    if len(q)<3:return 0.
    a=log_rotation(multiply(inverse(q[1:-1]),q[:-2]))
    b=log_rotation(multiply(inverse(q[1:-1]),q[2:]))
    return float(np.degrees(np.sqrt(np.mean(np.sum(((a+b)/2)**2,axis=1)))))


def stabilize_final_hands(scene,rig,tracks):
    """Filter the final FK curves once all image fitting and MANO edits are present."""
    from mathutils import Quaternion
    from calibrated_bridge import key_quat
    from hand_refinement import STEMS
    fps=scene.render.fps/scene.render.fps_base
    frame=scene.frame_current
    previous={}
    for side,track in tracks.items():
        if not np.any(track['valid']):continue
        names=['Arm_'+side,'Elbow_'+side,'Wrist_'+side]
        names.extend(f'{stem}{k if stem=="Thumb" else k+1}_{side}' for stem,_ in STEMS for k in range(3))
        bones=[rig.pose.bones[n] for n in names]
        count=len(track['valid'])
        raw=np.empty((count,len(bones),4))
        for t in range(count):
            scene.frame_set(t+1)
            raw[t]=[list(b.rotation_quaternion.normalized()) for b in bones]
        # A coherent sweep seen over several frames should retain its turning speed.
        uv=track['uv'][:,[0,5,9,13,17],:]
        movement=np.zeros(count)
        for t in range(1,count-1):
            if track['observed'][t-1:t+2].all():
                movement[t]=np.median(np.linalg.norm(uv[t+1]-uv[t-1],axis=1))*.5
        for j,bone in enumerate(bones):
            q=raw[:,j].copy()
            # Isolated flips require agreement from both neighbours.
            for t in range(1,count-1):
                a,b,c=raw[t-1:t+2,j]
                if abs(a@c)>.965 and max(abs(a@b),abs(c@b))<.91:
                    q[t]=a+c*np.sign(a@c)
                    q[t]/=np.linalg.norm(q[t])
            filtered=smooth_rotations(q,fps,limit_degrees=42.,range_degrees=60.)
            for t in range(count):
                maintain=.68*movement[t]**2/(movement[t]**2+.045**2)
                local=Quaternion(filtered[t]).slerp(Quaternion(q[t]),maintain)
                key_quat(bone,local,t+1,previous)
    scene.frame_set(frame)


def apply_to_scene(scene,rig,count):
    from mathutils import Quaternion
    from calibrated_bridge import MAP,key_quat
    # Hand fitting owns its single temporal pass after the body is finalized.
    names=[n for n in MAP.values() if n!='Neck' and not n.startswith(('Arm_', 'Elbow_', 'Wrist_', 'Thumb', 'IndexFinger', 'MiddleFinger', 'RingFinger', 'LittleFinger'))]
    samples={n:[] for n in names};positions=[];world_positions=[]
    for t in range(count):
        scene.frame_set(t+1)
        for n in names:samples[n].append(list(rig.pose.bones[n].rotation_quaternion.normalized()))
        positions.append(list(rig.location));world_positions.append(list(rig.matrix_world.translation))
    fps=scene.render.fps/scene.render.fps_base;previous={};report={}
    for n,raw in samples.items():
        limit=4. if n=='Head' else (8. if n.startswith(('Wrist','Thumb','Index','Middle','Ring','Little')) else 4.)
        filtered=smooth_rotations(raw,fps,limit)
        differences=log_rotation(multiply(inverse(np.asarray(raw)),filtered))
        report[n]={'jitter_before_deg':angular_jitter(raw),'jitter_after_deg':angular_jitter(filtered),
                   'max_pose_change_deg':float(np.degrees(np.linalg.norm(differences,axis=1).max()))}
        for t,q in enumerate(filtered):key_quat(rig.pose.bones[n],Quaternion(q.tolist()),t+1,previous)
    # Root is solved together with the final camera in actor_stability. Do not
    # independently smooth away the translation compensating camera motion.
    shifts=[]
    for t in range(count):
        scene.frame_set(t+1)
        shifts.append(np.array(rig.matrix_world.translation)-world_positions[t])
    report['root']={'max_world_change_m':float(np.linalg.norm(shifts,axis=1).max())}
    report['method']='symmetric local-linear SO(3) filter; bounded pose change; no hold-last-frame'
    return report,np.asarray(shifts)

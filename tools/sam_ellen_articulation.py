"""Bounded elbow-plane and finger hinge fitting; native lengths stay fixed."""
import math
import numpy as np
from mathutils import Matrix, Quaternion, Vector


def elbow_plane(arm, elbow, wrist, goal, bend, row, side, camera, data, options):
    from sam_ellen_screen import project
    source_height=data.get('video_orientation',{}).get('decoded_height_width',[0])[0]
    points=row.get('image_points')
    if not source_height or not points: return bend, {'elbow_reference':'capture'}
    target=np.array(points[7 if side=='L' else 8][:2])/source_height
    aspect=data['screen']['aspect']
    if not np.isfinite(target).all() or not (0<target[0]<aspect and 0<target[1]<1):
        return bend, {'elbow_reference':'capture / inferred point outside frame'}
    target[0]*=camera[2]/aspect
    if options.get('mirror'): target[0]=camera[2]-target[0]
    direction=(goal-arm.head).normalized()
    distance=max(1e-5,(goal-arm.head).length)
    upper,lower=(elbow.head-arm.head).length,(wrist.head-elbow.head).length
    along=(distance**2+upper**2-lower**2)/(2*distance)
    radius=math.sqrt(max(0.,upper**2-along**2))
    def image(angle):
        v=Quaternion(direction,float(angle))@bend
        return np.array(project(arm.head+direction*along+v*radius,camera))
    # SAM points project a 3D prediction, not an independent observed elbow.
    # Keep them weak and bound swivel to 35 degrees around captured motion.
    angles=np.linspace(-math.radians(35),math.radians(35),29)
    span=max(.04,np.linalg.norm(np.array(project(goal,camera))-np.array(project(arm.head,camera))))
    costs=[.15*np.sum(((image(a)-target)/span)**2)+.4*a*a for a in angles]
    angle=float(angles[int(np.argmin(costs))])*options['strength']
    return Quaternion(direction,angle)@bend, {'elbow_reference':'SAM projection / weak prior',
        'elbow_swivel_degrees':math.degrees(angle),
        'elbow_before_px':float(np.linalg.norm(image(0)-target)*options.get('height',1920)),
        'elbow_after_px':float(np.linalg.norm(image(angle)-target)*options.get('height',1920))}


def finger_bones(rig,side):
    result=[]
    for finger in ('Thumb','Index','Middle','Ring','Little'):
        bones=[b for b in rig.pose.bones if b.name.startswith(finger+'Finger') and b.name.endswith('_'+side)]
        if finger=='Thumb':
            bones=[b for b in rig.pose.bones if b.mmd_bone.name_j.startswith(('左' if side=='L' else '右')+'親指')]
        bones.sort(key=lambda b:len(b.parent_recursive))
        bones=bones[:3]
        if len(bones)!=3: raise ValueError('Missing finger chain: '+finger+side)
        result.append(bones)
    return result


def fit_fingers(scratch,rig,channels,data,options,cameras,metadata,report):
    from sam_ellen_frame_body import rotation
    from sam_ellen_screen_fit import projected
    from sam_ellen_retarget import set_global_rotation
    from sam_ellen_temporal import trajectory
    side=options['side']
    chains=finger_bones(rig,side)
    from sam_ellen_finger_constraints import constrain
    constrain(scratch,rig,channels,data,(side,))
    originals={name:list(samples) for name,samples in channels.items()}
    times=np.array([r['time'] for r in data['frames']])
    contexts={}
    values=np.zeros((len(times),5,4))
    weights=np.zeros((len(times),5))
    rest_across=(chains[1][0].bone.head_local-chains[4][0].bone.head_local).normalized()
    if side=='R': rest_across.negate()
    for i,row in enumerate(report):
        if i not in metadata or not row.get('active') or row.get('held') or row.get('weight',0)<=0:
            yield i+1,2*len(times)
            continue
        for name,samples in originals.items(): rig.pose.bones[name].rotation_quaternion=samples[i][1]
        scratch.view_layers[0].update()
        target,mask,_=metadata[i]
        for k,bones in enumerate(chains):
            indices=np.arange(1+4*k,5+4*k)
            # The coordinate mask says in-frame, never true visibility/confidence.
            if not mask[indices].all(): continue
            points=np.array([b.head[:] for b in bones]+[bones[-1].tail[:]])
            axes=[]
            for b in bones:
                axis=b.matrix.to_quaternion()@(b.bone.matrix_local.to_quaternion().inverted()@rest_across)
                axes.append(np.array(axis.normalized()))
            forward=(bones[0].tail-bones[0].head).normalized()
            spread=np.array(forward.cross(Vector(axes[0])).normalized())
            camera=cameras[i]
            selected=target[indices]
            scale=max(.015,float(np.linalg.norm(selected[-1]-selected[0])))
            def rotations(v):
                return [rotation(axes[0]*v[0]+spread*v[3]),rotation(axes[1]*v[1]),rotation(axes[2]*v[2])]
            def geometry(v):
                p=points.copy()
                for j,r in reversed(list(enumerate(rotations(v)))):
                    p[j+1:]=(p[j+1:]-points[j])@r.T+points[j]
                return p
            def residual(v):
                p=geometry(v)
                image=projected((np.c_[p,np.ones(4)]@np.asarray(camera[0]).T)[:,:3],camera)
                # Fit directions/shape relative to MCP; rigid fitting owns the palm.
                error=(.6*(image-selected)+.4*((image-image[0])-(selected-selected[0])))/scale
                return np.r_[error[1:].ravel(),v*.22]
            v=np.zeros(4)
            bound=np.radians([30,40,35,12])
            for iteration in range(12):
                r=residual(v)
                jac=np.stack([(residual(v+np.eye(4)[j]*1e-4)-r)/1e-4 for j in range(4)],axis=1)
                step=np.linalg.solve(jac.T@jac+np.eye(4)*.03,-jac.T@r)
                step*=min(1.,.15/max(1e-9,np.linalg.norm(step)))
                trial=np.clip(v+step,-bound,bound)
                if np.sum(residual(trial)**2)>=np.sum(r*r): break
                v=trial
                if np.linalg.norm(step)<1e-5:break
            values[i,k]=v*row['weight']
            weights[i,k]=1.
            contexts[i,k]=(points,axes,spread,[b.matrix.to_quaternion() for b in bones],selected,scale)
        yield i+1,2*len(times)
    for k in range(5):
        values[:,k]=trajectory(times,values[:,k],.045,np.maximum(.01,weights[:,k]),robust=.15)
    for i,row in enumerate(report):
        for name,samples in originals.items():rig.pose.bones[name].rotation_quaternion=samples[i][1]
        scratch.view_layers[0].update()
        improved=0
        for k,bones in enumerate(chains):
            context=contexts.get((i,k))
            if context is None:continue
            points,axes,spread,global_q,selected,scale=context
            cam=cameras[i]
            def error():
                p=np.array([b.head[:] for b in bones]+[bones[-1].tail[:]])
                image=projected((np.c_[p,np.ones(4)]@np.asarray(cam[0]).T)[:,:3],cam)
                return float(np.linalg.norm(.6*(image-selected)+.4*((image-image[0])-(selected-selected[0])),axis=1)[1:].mean())
            before=error()
            saved=[b.rotation_quaternion.copy() for b in bones]
            limit=np.radians([30,40,35,12])*row['weight']
            v=np.clip(values[i,k],-limit,limit)
            cumulative=Quaternion()
            for j,b in enumerate(bones):
                axis=axes[j]*v[j]+(spread*v[3] if j==0 else 0)
                angle=float(np.linalg.norm(axis))
                q=Quaternion(Vector(axis/angle),angle) if angle>1e-9 else Quaternion()
                cumulative=cumulative@q
                set_global_rotation(b,cumulative@global_q[j])
                scratch.view_layers[0].update()
            after=error()
            # A hard accept/revert decision makes an otherwise filtered hinge
            # correction pop on and off when its image benefit crosses zero.
            gain=float(np.clip((before-after)/max(.0005,.05*before),0.,1.))
            improved+=int(gain>0)
            for b,original in zip(bones,saved):
                frame=channels[b.name][i][0]
                q=original.slerp(b.rotation_quaternion,gain)
                if i and q.dot(Quaternion(channels[b.name][i-1][1]))<0:q.negate()
                channels[b.name][i]=(frame,tuple(q))
        if row.get('active'):row['finger_fit']={'model':'bounded-hinges-v2-continuous','accepted_fingers':improved,
            'joint_correction_limits_degrees':[30,40,35], 'spread_limit_degrees':12,
            'in_frame_is_visibility':False}
        yield len(times)+i+1,2*len(times)

    constrain(scratch,rig,channels,data,(side,))

"""Recover the observed hand orientation from labelled image joints.

Approach to the lens changes translation and apparent scale; it does not imply
that the hand's longitudinal axis, palm or knuckles should face the camera.
"""
import math
import numpy as np
from mathutils import Matrix,Vector


def measured_rotation(local,world,camera,mirror=False):
    """Transfer wrist-anchored palm axes independently of hand proportions."""
    p=np.asarray(local)[[0,5,9,13,17]].copy();q=np.asarray(world)[[0,5,9,13,17]].copy()
    q*=np.array([-1. if mirror else 1.,-1.,-1.])
    def basis(points):
        along=points[1:].mean(0)-points[0];along/=max(1e-8,np.linalg.norm(along))
        across=points[1]-points[4];across-=along*np.dot(across,along)
        across/=max(1e-8,np.linalg.norm(across))
        return np.stack([across,along,np.cross(across,along)],axis=1)
    rotation=basis(q)@basis(p).T
    return (camera[0].to_quaternion().inverted()@Matrix(rotation.tolist()).to_quaternion()).normalized()


def hand_boundaries(data,segments,observations,side):
    """Exclude the forearm from masks fitted to a hand-only skinned mesh."""
    from sam_punch_evidence import fist_observation
    from sam_ellen_punch_mesh import hull
    from sam_ellen_arm_visibility import hand_landmark_mask
    outlines={};wrists={};modes={}
    def clip(p,wrist,axis,margin):
        result=[];previous=p[-1];last=float((previous-wrist)@axis+margin)
        for point in p:
            value=float((point-wrist)@axis+margin)
            if (value>=0)!=(last>=0):result.append(previous+(point-previous)*last/(last-value))
            if value>=0:result.append(point)
            previous,last=point,value
        return np.array(result)
    for segment in segments:
        baseline=None
        for i,row in enumerate(data['frames']):
            time=row['time']-data['frames'][0]['time']
            if not segment['start']<=time<segment['end']:continue
            obs=observations[i]
            if not obs or not obs.get('silhouette'):continue
            polygon=hull(obs['silhouette']['outline'])
            hand=data['screen']['frames'][i].get('hands',{}).get(side)
            measured=fist_observation(hand) is not None and hand_landmark_mask(hand)[0]
            if measured:
                p=np.array(hand['points']);wrist=p[0];axis=p[[5,9,13,17]].mean(0)-wrist
                axis/=max(1e-8,np.linalg.norm(axis));across=np.array([-axis[1],axis[0]])
                cut=clip(polygon,wrist,axis,0.)
                if len(cut)<3:continue
                width=max(.015,float(np.ptp(cut@across)))
                baseline=(wrist,axis,across,width,float((cut@axis).max()-wrist@axis),float((cut@across).min()+(cut@across).max())*.5)
                mode='measured-wrist'
            elif baseline is not None:
                w0,axis,across,width,length,center=baseline
                scale=float(np.clip(np.ptp(polygon@across)/width,.35,8.))
                along=float((polygon@axis).max()-length*scale)
                lateral=float(((polygon@across).min()+(polygon@across).max())*.5+(w0@across-center)*scale)
                wrist=axis*along+across*lateral;mode='mask-derived-wrist-boundary'
            else:continue
            cut=clip(polygon,wrist,axis,.003)
            if len(cut)>=3:outlines[i]=cut;wrists[i]=wrist;modes[i]=mode
    # Cropping changes mask extrema abruptly when the fist crosses the image
    # border. Filter only derived boundaries; retain real wrist observations.
    from sam_ellen_temporal import trajectory
    times=np.array([r['time'] for r in data['frames']])
    for segment in segments:
        ids=[i for i in wrists if segment['start']<=times[i]-times[0]<segment['end']]
        if len(ids)<3:continue
        values=trajectory(times[ids],[wrists[i] for i in ids],.055,
            weights=[4. if modes[i]=='measured-wrist' else 1. for i in ids],robust=.025)
        for j,i in enumerate(ids):
            if modes[i]=='measured-wrist':continue
            # Shift the already clipped polygon's cut towards the filtered
            # boundary; raw silhouette still supplies all visible upper edges.
            hand_ids=[k for k in ids if k<=i and modes[k]=='measured-wrist']
            if not hand_ids:continue
            hand=data['screen']['frames'][hand_ids[-1]]['hands'][side]
            p=np.array(hand['points']);axis=p[[5,9,13,17]].mean(0)-p[0]
            axis/=max(1e-8,np.linalg.norm(axis))
            polygon=hull(observations[i]['silhouette']['outline'])
            cut=clip(polygon,values[j],axis,.003)
            if len(cut)>=3:outlines[i]=cut;wrists[i]=values[j]
    return outlines,wrists,modes


def observed_pose(local,target,weights,camera,rotation,wrist):
    from sam_ellen_screen_fit import projected
    from sam_ellen_frame_body import rotation as rotvec
    from sam_ellen_image_refine import turn
    transform=np.asarray(camera[0]);local=np.asarray(local);target=np.asarray(target)
    weights=np.asarray(weights,float);weights/=max(1e-8,weights.sum())
    center=np.sum(target*weights[:,None],axis=0)
    size=max(.02,float(np.sqrt(np.sum(np.sum((target-center)**2,axis=1)*weights))))
    initial=np.asarray(camera[0].to_3x3()@rotation.to_matrix())
    p=np.asarray(camera[0]@wrist);best=None
    axes=[np.eye(3)]
    for axis in np.eye(3):
        axes.extend(rotvec(axis*angle) for angle in (math.pi,math.pi/2,-math.pi/2))
    candidates=[]
    for delta in axes:
        base=delta@initial
        native=projected(local@base.T+p,camera)
        mean=np.sum(native*weights[:,None],axis=0)
        span=float(np.sqrt(np.sum(np.sum((native-mean)**2,axis=1)*weights)))
        depth=float(np.clip(-p[2]*span/size,.07,2.))
        from sam_ellen_screen import unproject
        cp=camera[0]@unproject(Vector(center),depth,camera)
        origin=np.array(cp)-np.sum((local@base.T)*weights[:,None],axis=0)
        def residual(value):
            points=local@(rotvec(value[3:])@base).T+origin+value[:3]
            image=projected(points,camera)
            return np.r_[((image-target)*np.sqrt(weights[:,None])/size).ravel(),
                         np.maximum(0,points[:,2]+.025)*30]
        value=np.zeros(6)
        for _ in range(45):
            r=residual(value)
            jac=np.stack([(residual(value+np.eye(6)[j]*1e-5)-r)/1e-5 for j in range(6)],axis=1)
            step=np.linalg.solve(jac.T@jac+np.eye(6)*.002,-jac.T@r)
            step[:3]*=min(1.,.06/max(1e-8,np.linalg.norm(step[:3])))
            step[3:]*=min(1.,.3/max(1e-8,np.linalg.norm(step[3:])))
            improved=False
            for gain in (1.,.5,.2):
                trial=value+step*gain;rr=residual(trial)
                if rr@rr<r@r:value=trial;improved=True;break
            if not improved or np.linalg.norm(step)<1e-6:break
        score=float(residual(value)@residual(value))
        candidates.append((score,value,base,origin))
    score,value,base,origin=min(candidates,key=lambda x:x[0])
    q=camera[0].to_quaternion().inverted()@Matrix((rotvec(value[3:])@base).tolist()).to_quaternion()
    return q.normalized(),{'model':'labelled-joint-hand-orientation-v1','normalized_joint_error':math.sqrt(score)}

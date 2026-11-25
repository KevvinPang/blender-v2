"""Anatomical hand fitting against the final, editable Blender camera.

The wrist, palm and finger joints share one image objective. Bone lengths
never change. Finger zero poses explicitly cancel the rig's bind-pose curl.
No gesture labels or clip-specific frames are used.
"""
import math
import numpy as np
from mathutils import Matrix, Quaternion, Vector

STEMS = [('Thumb',1),('IndexFinger',5),('MiddleFinger',9),('RingFinger',13),('LittleFinger',17)]


def unit(x):
    return x / np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-10)


def skew(v):
    v=np.asarray(v);out=np.zeros((*v.shape[:-1],3,3))
    out[...,0,1]=-v[...,2];out[...,0,2]=v[...,1]
    out[...,1,0]=v[...,2];out[...,1,2]=-v[...,0]
    out[...,2,0]=-v[...,1];out[...,2,1]=v[...,0]
    return out


def rotvec(v):
    v=np.asarray(v);theta=np.linalg.norm(v,axis=-1)
    A=np.sinc(theta/np.pi);B=.5*np.sinc(theta/(2*np.pi))**2
    S=skew(v)
    return np.eye(3)+A[...,None,None]*S+B[...,None,None]*(S@S)


def between(a,b):
    a=unit(a);b=unit(b);v=np.cross(a,b);s=np.linalg.norm(v);c=np.clip(a@b,-1.,1.)
    if s<1e-9:
        if c>0:return np.eye(3)
        axis=unit(np.cross(a,np.eye(3)[np.argmin(abs(a))]))
        return rotvec(axis*np.pi)
    return rotvec(v/s*np.arctan2(s,c))


def palm(points):
    y=unit(points[9]-points[0]);x=points[5]-points[17];x=unit(x-y*(x@y))
    return np.stack([x,y,np.cross(x,y)],axis=1)


class HandModel:
    def __init__(self,rig,side,curl_sign=-1.):
        self.side=side;self.names=[];self.joints=[];self.lower=[];self.upper=[]
        wrist=rig.data.bones['Wrist_'+side];self.wrist_rest=np.array(wrist.matrix_local.to_3x3())
        rp=np.zeros((21,3));rp[0]=wrist.head_local
        for stem,start in STEMS:
            bones=[rig.data.bones[f'{stem}{k if stem=="Thumb" else k+1}_{side}'] for k in range(3)]
            for k,b in enumerate(bones):rp[start+k]=b.head_local
            rp[start+3]=bones[-1].tail_local
        self.rest_palm=palm(rp)
        self.rest_points=(rp-rp[0])@self.wrist_rest
        self.segments=[(start+k,start+k+1) for _,start in STEMS for k in range(3)]
        self.neutral_dirs=unit(np.array([self.rest_points[b]-self.rest_points[a] for a,b in self.segments]))
        for stem,start in STEMS:
            parent_zero=self.wrist_rest
            bones=[rig.data.bones[f'{stem}{k if stem=="Thumb" else k+1}_{side}'] for k in range(3)]
            d0=unit(np.array(bones[0].tail_local-bones[0].head_local))
            for k,b in enumerate(bones):
                B=np.array(b.matrix_local.to_3x3());P=np.array(b.parent.matrix_local.to_3x3())
                bind=P.T@B;offset=P.T@np.array(b.head_local-b.parent.head_local)
                zero=B if stem=='Thumb' else between(B[:,1],d0)@B
                qzero=(parent_zero@bind).T@zero
                direction=zero[:,1]
                flex=unit(self.rest_palm[:,0]-direction*(direction@self.rest_palm[:,0]))*curl_sign
                splay=unit(np.cross(direction,flex))
                axes=[zero.T@flex]
                limits=[(-15.,110.) if k==0 else (0.,115. if k==1 else 95.)]
                if stem=='Thumb':
                    axes=[zero.T@flex,zero.T@splay];limits=[(-75.,85.),(-65.,65.)]
                elif k==0:
                    axes.append(zero.T@splay);limits.append((-32.,32.))
                index=len(self.lower)
                self.lower.extend(np.radians([a for a,b in limits]));self.upper.extend(np.radians([b for a,b in limits]))
                self.joints.append(dict(bind=bind,offset=offset,zero=qzero,axes=np.array(axes),index=index,
                    anatomical_basis=self.wrist_rest.T@zero,
                    parent=-1 if k==0 else len(self.joints)-1,point=start+k,
                    tip=np.array([0.,b.length,0.]) if k==2 else None))
                self.names.append(b.name);parent_zero=zero
        self.lower=np.array(self.lower);self.upper=np.array(self.upper)

    def forward(self,angles,locals_out=False):
        angles=np.atleast_2d(angles);N=len(angles);points=np.zeros((N,21,3));Rs=[];positions=[];locals_=[]
        for joint in self.joints:
            parent=joint['parent'];P=np.broadcast_to(np.eye(3),(N,3,3)) if parent<0 else Rs[parent]
            pos=np.zeros((N,3)) if parent<0 else positions[parent]
            pos=pos+np.einsum('nij,j->ni',P,joint['offset'])
            local=np.broadcast_to(joint['zero'],(N,3,3)).copy()
            for k,axis in enumerate(joint['axes']):local=local@rotvec(angles[:,joint['index']+k,None]*axis)
            R=P@joint['bind']@local
            Rs.append(R);positions.append(pos);locals_.append(local)
            points[:,joint['point']]=pos
            if joint['tip'] is not None:points[:,joint['point']+1]=pos+np.einsum('nij,j->ni',R,joint['tip'])
        return (points,locals_) if locals_out else points

    def seed(self,source):
        result=np.zeros(len(self.lower))
        source_R=palm(source)@self.rest_palm.T@self.wrist_rest
        for f,(_,start) in enumerate(STEMS):
            if f==0:continue
            v=unit(np.diff(source[start:start+4],axis=0))
            joint=self.joints[f*3];index=joint['index']
            wanted=joint['anatomical_basis'].T@(source_R.T@v[0])
            # Calibrate the MCP flexion/splay in its own anatomical frame.
            # Distal straightness alone cannot tell the MCP where to point.
            aa=np.linspace(self.lower[index],self.upper[index],25)
            bb=np.linspace(self.lower[index+1],self.upper[index+1],17)
            grid=np.array(np.meshgrid(aa,bb,indexing='ij')).reshape(2,-1).T
            rotations=rotvec(grid[:,0,None]*joint['axes'][0])@rotvec(grid[:,1,None]*joint['axes'][1])
            best=np.argmin(np.sum((rotations[:,:,1]-wanted)**2,axis=1))
            result[index:index+2]=grid[best]
            bends=np.arccos(np.clip(np.sum(v[:-1]*v[1:],axis=1),-1,1))
            result[self.joints[f*3+1]['index']]=bends[0]
            result[self.joints[f*3+2]['index']]=bends[1]
        return np.clip(result,self.lower,self.upper)


def least_squares(fun,x,lower,upper,iterations=25,project=None):
    """Batched finite-difference LM; Blender needs only its bundled NumPy."""
    x=np.clip(np.array(x,dtype=float),lower,upper)
    if project is not None:x=project(x)
    damping=.01
    steps=np.full(len(x),1e-4);best=np.sum(fun(x[None])[0]**2)
    for _ in range(iterations):
        batch=np.repeat(x[None],len(x)+1,axis=0)
        batch[np.arange(1,len(x)+1),np.arange(len(x))]+=steps
        values=fun(batch);r=values[0];J=((values[1:]-r)/steps[:,None]).T
        normal=J.T@J;diagonal=np.maximum(np.diag(normal),1e-5)
        step=np.linalg.solve(normal+np.diag(diagonal*damping),-J.T@r)
        step[:3]=np.clip(step[:3],-.08,.08);step[3:]=np.clip(step[3:],-.45,.45)
        candidate=np.clip(x+step,lower,upper)
        if project is not None:candidate=project(candidate)
        cost=np.sum(fun(candidate[None])[0]**2)
        if cost<best:
            movement=np.max(abs(candidate-x));x=candidate;best=cost;damping=max(1e-5,damping*.4)
            if movement<2e-5:break
        else:damping=min(1e5,damping*5.)
    return x,float(best)


def intrinsics(scene,camera):
    from bpy_extras.object_utils import world_to_camera_view
    M=camera.matrix_world
    pts=[world_to_camera_view(scene,camera,M@Vector(p)) for p in [(0,0,-1),(1,0,-1),(0,1,-1)]]
    w,h=scene.render.resolution_x,scene.render.resolution_y
    return np.array([[(pts[1].x-pts[0].x)*w,0,pts[0].x*w],
        [0,(pts[2].y-pts[0].y)*h,(1-pts[0].y)*h],[0,0,1.]])


def bounded_span_scale(reference_span,observed_span,limits=(.85,1.15)):
    return float(np.clip(reference_span/max(float(observed_span),1.),*limits))


def wrist_correction_limit(uv,image_size,predicted=False):
    if predicted:return np.radians(35.)
    points=np.asarray(uv[5:9])*image_size
    lengths=np.linalg.norm(np.diff(points,axis=0),axis=1)
    straightness=np.linalg.norm(points[-1]-points[0])/max(lengths.sum(),1e-6)
    return np.radians(35.+35.*np.clip((straightness-.72)/.23,0.,1.))


def supported_palm_rotations(world,valid,observed):
    """Reject an unsupported half-turn, including the last blurred exit frame.

    A sustained rotation with support in either of the following two observed
    frames is kept. Predicted tails inherit the corrected orientation instead
    of repeating the rejected last detection for another eight frames.
    """
    raw={t:palm(world[t]) for t in np.flatnonzero(valid)}
    result={t:R.copy() for t,R in raw.items()};correction=np.eye(3)
    def angle(a,b):
        return np.arccos(np.clip((np.trace(a.T@b)-1.)*.5,-1.,1.))
    for t in sorted(raw):
        if not observed[t]:
            if t-1 in raw:result[t]=correction@raw[t]
            else:correction=np.eye(3)
            continue
        correction=np.eye(3)
        if t-2 not in raw or not observed[t-2:t].all():continue
        jump=angle(result[t-1],raw[t])
        prior_speed=angle(result[t-2],result[t-1])
        following=[k for k in range(t+1,min(len(valid),t+3)) if observed[k] and k in raw]
        supported=any(angle(raw[t],raw[k])<np.radians(40.) for k in following)
        if jump>np.radians(100.) and prior_speed<np.radians(35.) and not supported:
            result[t]=result[t-1].copy()
            correction=result[t]@raw[t].T
    return result


def reachable_wrist(ray, shoulder, depth, radius, maximum):
    """Find a bent, reachable arm on the observed ray, near its reference depth."""
    direction=unit(ray)
    along=direction@shoulder
    lateral=np.linalg.norm(shoulder-along*direction)
    if lateral>=maximum:
        closest=direction*max(along,.055/direction[2])
        return shoulder+unit(closest-shoulder)*maximum
    extent=np.sqrt(maximum**2-lateral**2)
    near=max(.055/direction[2],along-extent)
    far=max(near,along+extent)
    preferred=np.clip(depth/direction[2],near,far)
    bend_extent=np.sqrt(max(0.,radius**2-lateral**2))
    # A soft bend prior leaves the depth cue continuous at tangency.
    bent=along+np.clip((preferred-along)/.04,-1.,1.)*bend_extent
    return direction*np.clip(.68*preferred+.32*bent,near,far)


def hinge_basis(axis,normal):
    x=unit(axis)
    z=unit(normal-x*(normal@x))
    return np.column_stack((x,np.cross(z,x),z))


def arm_reference_frames(scene,rig,side,track):
    """Calibrate the hinge roll once per episode in a clearly bent pose."""
    bones=[rig.pose.bones[n+'_'+side] for n in ('Arm','Elbow','Wrist')]
    valid=track['valid']
    result={}
    runs=np.split(np.flatnonzero(valid),np.where(np.diff(np.flatnonzero(valid))>1)[0]+1)
    for run in runs:
        if not len(run):continue
        seen=run[track['observed'][run]]
        if not len(seen):continue
        candidates=seen[min(4,len(seen)-1):min(16,len(seen))]
        best=0.
        for t in candidates:
            scene.frame_set(int(t)+1)
            a,e,w=[np.array(rig.matrix_world@b.head) for b in bones]
            axes=(unit(e-a),unit(w-e));normal=np.cross(*axes)
            score=np.linalg.norm(normal)
            if score<=max(best,.12):continue
            best=score
            bases=[]
            for bone,axis in zip(bones,axes):
                inverse=np.array((rig.matrix_world.to_3x3()@bone.matrix.to_3x3()).inverted())
                bases.append(hinge_basis(inverse@axis,inverse@normal).T)
        if best:
            result.update(dict.fromkeys(run,bases))
    return result


def ease_border_turns(rotations,run,track,fps):
    """Redistribute isolated, edge-ambiguous twist without filtering wrist travel."""
    values=[Quaternion(row.tolist()).normalized() for row in rotations]
    steps=np.array([np.degrees(2*np.arccos(np.clip(abs(a.dot(b)),0,1)))
                    for a,b in zip(values,values[1:])])
    last=-1
    radius=max(2,round(fps*.075))
    for k in np.flatnonzero(steps>1200./fps):
        t=run[k+1]
        if track['edge'][t]<0 and np.min(np.minimum(track['uv'][t,0],1-track['uv'][t,0]))>.07:
            continue
        lo=max(0,k-radius+1);hi=min(len(run)-1,k+radius)
        if lo<last or hi-lo<2:continue
        # A supported multi-frame turn retains its observed angular trajectory.
        if np.count_nonzero(steps[lo:hi]>.65*steps[k])>1:continue
        for j in range(lo+1,hi):
            rotations[j]=values[lo].slerp(values[hi],(j-lo)/(hi-lo))
        last=hi
    return rotations


def finger_activity(track, image_size):
    """Measure supported tip motion in a translating, rotating, scaled palm frame."""
    pixels = track['uv'] * image_size
    axis = pixels[:,9]-pixels[:,0]
    scale = np.maximum(np.linalg.norm(axis,axis=1),1.)
    axis = axis/scale[:,None]
    basis = np.stack((axis, np.stack((-axis[:,1],axis[:,0]),axis=1)),axis=2)
    starts = np.array([1,5,9,13,17])
    local = ((pixels[:,starts+3]-pixels[:,starts]) @ basis)/scale[:,None,None]
    reliable = track['observed'] & ~track['held'] & (scale>12)
    inside = ((track['uv'][:,starts+3]>=0)&(track['uv'][:,starts+3]<=1)).all(axis=2)
    strength = np.zeros((len(pixels),5))
    for t in range(3,len(pixels)-3):
        window = slice(t-3,t+4)
        if not reliable[window].all():
            continue
        # Adjacent three-frame medians ignore isolated landmark jumps.
        centers = np.stack([np.median(local[j:j+3],axis=0) for j in (t-3,t-1,t+1)])
        extent = np.linalg.norm(np.ptp(centers,axis=0),axis=1)
        strength[t] = np.clip((extent-.018)/.06,0,1)*inside[window].all(axis=0)
    return strength


def fit_frame(model,source,uv,K,shoulder,lengths,initial_wrist,flex_reference,previous_angles=None,predicted=False,reference_depth=None,reference_span=None,palm_reference=None,temporal_weight=0.,wrist_reference=None,previous_correction=None,activity=None,digit_weights=None,parts_only=False):
    source_palm=palm(source) if palm_reference is None else palm_reference
    # Keep the local finger geometry and accepted palm in the same frame.
    source=(source-source[0])@palm(source)@source_palm.T+source[0]
    source_rotation=source_palm@model.rest_palm.T@model.wrist_rest
    # Position/depth and elbow references preserve the accepted composition.
    # Wrist orientation must come from the SAME observation as the fingers.
    # The old bake may contain wrong-handed detections; using it as a moving
    # rotation anchor made neighbouring solves choose opposite limit edges.
    R0=source_rotation
    source_dirs=unit(np.array([source[b]-source[a] for a,b in model.segments]))
    local_dirs=source_dirs@source_rotation
    observed_seed=model.seed(source)
    seed=observed_seed.copy()
    # The warm start helps only shape convergence, never freezes observations.
    if previous_angles is not None:seed=.35*previous_angles+.65*observed_seed
    points=model.forward(seed)[0]@R0.T
    from calibrated_bridge import fit_translation
    inside=((uv>=0)&(uv<=1)).all(1)
    ids=np.array([0,5,9,13,17]);usable=ids[inside[ids]]
    if len(usable)<3:usable=np.flatnonzero(inside)
    if predicted and len(usable)<3:usable=ids
    pixels=uv*model.image_size
    depth=initial_wrist[2] if reference_depth is None else reference_depth
    wrist_limit=wrist_correction_limit(uv,model.image_size,predicted)
    if parts_only:
        usable=np.flatnonzero(digit_weights)
    if reference_span is not None and not parts_only:
        center=pixels[[5,9,13,17]].mean(0)
        span=max(np.linalg.norm(pixels[5]-pixels[17]),1.)
        # The palm landmark span collapses when a fist is foreshortened. An
        # unconditional correction turns that detector error into a huge hand.
        span_scale=bounded_span_scale(reference_span,span)
        pixels=center+(pixels-center)*span_scale
    try:translation=fit_translation(points[usable],pixels[usable],K,np.ones(len(usable)))
    except (ValueError,np.linalg.LinAlgError):translation=initial_wrist.copy()
    if not np.isfinite(translation).all() or translation[2]<.05:translation=initial_wrist.copy()
    # Keep a bent source elbow bent. A genuine straight-arm source may still
    # extend: this is relative to the immutable source, not a generic pose cap.
    minimum_flex=max(0.,flex_reference-np.radians(5.))
    maximum_flex=min(np.radians(150.),flex_reference+np.radians(5.))
    def reach(flex):return np.sqrt(lengths[0]**2+lengths[1]**2+2*lengths[0]*lengths[1]*np.cos(flex))
    maximum=min(sum(lengths)*.995,reach(minimum_flex));minimum=reach(maximum_flex)
    reference_distance=reach(flex_reference)
    delta=translation-shoulder;translation=shoulder+delta*min(1.,maximum/max(np.linalg.norm(delta),1e-8))
    translation[2]=depth
    x=np.r_[translation,np.zeros(3),seed]
    if previous_correction is not None:
        q=Matrix(previous_correction).to_quaternion().normalized()
        if q.w<0:q.negate()
        x[3:6]=np.array(q.axis)*q.angle
    lo=np.r_[[-10,-10,.04],[-1.2]*3,model.lower];hi=np.r_[[10,10,10],[1.2]*3,model.upper]
    # Preserve readable flexion state. Image/depth fitting cannot convert a
    # clearly extended index into a curled finger to reduce position error.
    for finger in range(1,5):
        root_index=model.joints[finger*3]['index']
        for index in [root_index,root_index+1]:
            lo[6+index]=max(lo[6+index],observed_seed[index]-np.radians(22.))
            hi[6+index]=min(hi[6+index],observed_seed[index]+np.radians(22.))
        for joint in (1,2):
            index=model.joints[finger*3+joint]['index']
            lo[6+index]=max(lo[6+index],observed_seed[index]-np.radians(18.))
            hi[6+index]=min(hi[6+index],observed_seed[index]+np.radians(18.))
    weights=np.ones(21);weights[[0,5,9,13,17]]=1.2;weights[[4,8,12,16,20]]=2.
    weights*=np.ones(21) if predicted else inside
    if digit_weights is not None and np.any(digit_weights):
        weights*=digit_weights if parts_only else np.where(inside,1.,.2)
        weights+=digit_weights*3.
    starts=np.array([1,5,9,13,17])
    tip_weight=np.zeros(5) if activity is None else activity
    local_weight=tip_weight*inside[starts]*inside[starts+3]
    if parts_only:
        local_weight*=digit_weights[starts]*digit_weights[starts+3]
    source_tip_vectors=(pixels[starts+3]-pixels[starts])
    height=model.image_size[1]
    # Relative finger spacing is distinct from absolute image placement.
    # Preserve observed adduction instead of opening fingers to fill pixels.
    pair_a=np.array([8,12,16]);pair_b=np.array([12,16,20])
    source_width=max(np.linalg.norm(source[5]-source[17]),1e-6)
    source_spacing=np.linalg.norm(source[pair_a]-source[pair_b],axis=1)/source_width
    source_prox=unit(np.array([source[k+1]-source[k] for k in [5,9,13,17]]))
    source_separation=np.sum(source_prox[:-1]*source_prox[1:],axis=1)
    image_prox=unit(np.array([pixels[k+1]-pixels[k] for k in [5,9,13,17]]))
    image_separation=np.sum(image_prox[:-1]*image_prox[1:],axis=1)
    image_axes=unit(np.array([pixels[k+3]-pixels[k] for k in [5,9,13,17]]))
    straight=np.array([np.linalg.norm(pixels[k+3]-pixels[k])/max(np.linalg.norm(np.diff(pixels[k:k+4],axis=0),axis=1).sum(),1e-6) for k in [5,9,13,17]])
    alignment=float(np.min(np.sum(image_axes[:-1]*image_axes[1:],axis=1)))
    closed_evidence=float(np.clip((alignment-.94)/.055,0,1)*np.clip((straight.min()-.9)/.08,0,1))
    if parts_only:closed_evidence=0.
    # Strong visible adduction must be allowed to override the unreliable
    # per-finger 3D splay seed. Previously those hard boxes could exclude every
    # closed-hand solution even when the adduction detector returned 1.0.
    for finger in range(1,5):
        index=model.joints[finger*3]['index']
        for j in (index,index+1):
            lo[6+j]=(1-closed_evidence)*lo[6+j]+closed_evidence*model.lower[j]
            hi[6+j]=(1-closed_evidence)*hi[6+j]+closed_evidence*model.upper[j]
    def residual(batch):
        shape=model.forward(batch[:,6:]);R=R0@rotvec(batch[:,3:6])
        pc=shape@np.swapaxes(R,1,2)+batch[:,None,:3]
        z=np.maximum(pc[:,:,2:3],.025)
        predicted=pc[:,:,:2]/z*np.array([K[0,0],K[1,1]])+K[:2,2]
        image=((predicted-pixels)/height*weights[None,:,None]*50).reshape(len(batch),-1)
        tip_vectors=predicted[:,starts+3]-predicted[:,starts]
        articulation=((tip_vectors-source_tip_vectors)/height*local_weight[None,:,None]*180).reshape(len(batch),-1)
        dirs=unit(np.stack([shape[:,b]-shape[:,a] for a,b in model.segments],axis=1))
        geometry=((dirs-local_dirs)*.65*(1.-.85*closed_evidence)).reshape(len(batch),-1)
        palm_width=np.linalg.norm(shape[:,5]-shape[:,17],axis=1)
        spacing=np.linalg.norm(shape[:,pair_a]-shape[:,pair_b],axis=2)/palm_width[:,None]
        finger_spacing=(spacing-source_spacing)*2.5*(1.-closed_evidence)
        prox=dirs[:,[3,6,9,12]]
        separation=(np.sum(prox[:,:-1]*prox[:,1:],axis=2)-source_separation)*3.*(1.-closed_evidence)
        projected_prox=unit(np.stack([predicted[:,k+1]-predicted[:,k] for k in [5,9,13,17]],axis=1))
        visible_separation=(np.sum(projected_prox[:,:-1]*projected_prox[:,1:],axis=2)-image_separation)*8.
        if parts_only:visible_separation*=0.
        long_axes=unit(np.stack([shape[:,k+3]-shape[:,k] for k in [5,9,13,17]],axis=1))
        mean_axis=unit(long_axes.mean(axis=1))
        gaps=shape[:,pair_a]-shape[:,pair_b]
        lateral=gaps-mean_axis[:,None,:]*np.sum(gaps*mean_axis[:,None,:],axis=2,keepdims=True)
        closed_gap=np.maximum(np.linalg.norm(lateral,axis=2)/palm_width[:,None]-.21,0)*closed_evidence*65.
        parallel=(long_axes-mean_axis[:,None,:]).reshape(len(batch),-1)*closed_evidence*15.
        closed_prox=(prox-unit(prox.mean(axis=1))[:,None,:]).reshape(len(batch),-1)*closed_evidence*12.
        distance=np.linalg.norm(batch[:,:3]-shoulder,axis=1)
        reach=(np.maximum(distance-maximum,0)+np.minimum(distance-minimum,0))[:,None]*200
        elbow_prior=(distance-reference_distance)[:,None]*15
        wrist_prior=batch[:,3:6]*(2.5*np.radians(35.)/wrist_limit)
        depth_prior=(batch[:,2:3]-depth)*100
        temporal=(batch[:,6:]-seed)*temporal_weight*(1.-.8*closed_evidence)
        # Smooth the solver's correction relative to the observed palm, not
        # the palm motion itself. Fast observed turns retain their timing.
        wrist_temporal=np.zeros((len(batch),9)) if previous_correction is None else (
            rotvec(batch[:,3:6])-previous_correction).reshape(len(batch),9)*2.5
        return np.concatenate([image,articulation,geometry,finger_spacing,separation,visible_separation,closed_gap,parallel,closed_prox,reach,elbow_prior,wrist_prior,depth_prior,temporal,wrist_temporal],axis=1)
    def project(candidate):
        delta=candidate[:3]-shoulder;distance=np.linalg.norm(delta)
        candidate[:3]=shoulder+delta*min(1.,sum(lengths)*.985/max(distance,1e-8))
        angle=np.linalg.norm(candidate[3:6])
        candidate[3:6]*=min(1.,wrist_limit/max(angle,1e-8))
        return candidate
    x,cost=least_squares(residual,x,lo,hi,iterations=40,project=project)
    shape,locals_=model.forward(x[None,6:],True);Rw=R0@rotvec(x[3:6]);goal=x[:3]
    # Solve wrist placement on the final camera ray. A fixed depth plane and
    # a hard flexion shell can intersect on opposite sides of the shoulder.
    offsets=shape[0]@Rw.T
    desired=(uv*model.image_size-K[:2,2])/K[[0,1],[0,1]]
    z=max(.055,goal[2])
    implied=(desired*(z+offsets[:,2:])-offsets[:,:2])/z
    xy=.75*desired[0]+.25*np.median(implied[ids],axis=0)
    if digit_weights is not None and np.any(digit_weights):
        weight=digit_weights.max()
        xy=(1-weight)*xy+weight*np.average(implied,axis=0,weights=digit_weights)
    goal=reachable_wrist(np.r_[xy,1.],shoulder,depth,reference_distance,
                         min(sum(lengths)*.985,reach(max(0.,flex_reference-np.radians(25.)))))
    return goal,Rw,[v[0] for v in locals_],x[6:],cost


def head_contact_motion(track, side, image_size, body, face):
    """Use short, head-relative wrist excursions only during supported contact."""
    uv=track['uv']
    strength=np.zeros(len(uv))
    depth_delta=np.zeros(len(uv))
    if body is None or face is None or face.get('landmarks') is None:
        return strength,depth_delta
    wrist,elbow=(62,7) if side=='L' else (41,8)
    xy=uv*image_size
    eyes=face['landmarks'][:,[33,263],:2]*image_size
    face_scale=np.linalg.norm(eyes[:,1]-eyes[:,0],axis=1)
    center=eyes.mean(axis=1)
    near=np.linalg.norm(xy[:,[5,9,13,17]].mean(axis=1)-center,axis=1)
    match=np.linalg.norm(body['keypoints_2d'][:,wrist]-xy[:,0],axis=1)
    good=track['observed'] & ~track['held'] & face['valid'] & (near<1.6*face_scale) & (match<.6*face_scale)
    relative=xy[:,0]-center
    axis=xy[:,[5,9,13,17]].mean(axis=1)-xy[:,0]
    axis=unit(axis)
    xyz=body['keypoints_3d']
    depth=xyz[:,wrist,2]-xyz[:,0,2]
    forearm=np.linalg.norm(xyz[:,wrist]-xyz[:,elbow],axis=1)
    for t in range(3,len(uv)-3):
        window=slice(t-3,t+4)
        if not good[window].all() or face_scale[t]<10:
            continue
        centers=np.stack([np.median(relative[j:j+3],axis=0) for j in (t-3,t-1,t+1)])
        directions=np.stack([np.median(axis[j:j+3],axis=0) for j in (t-3,t-1,t+1)])
        travel=np.linalg.norm(np.ptp(centers,axis=0))/face_scale[t]
        swing=np.linalg.norm(np.ptp(directions,axis=0))
        strength[t]=np.clip(max((travel-.012)/.035,(swing-.023)/.06),0,1)
        if np.isfinite(depth[window]).all() and forearm[t]>.02:
            depth_delta[t]=np.clip((depth[t]-np.median(depth[window]))/forearm[t],-.07,.07)
    return strength,depth_delta


def apply(scene,rig,tracks,frames=None,body=None,face=None):
    import bpy
    from calibrated_bridge import key_quat
    from temporal_denoise import smooth_rotations
    T=int(rig.get('mocap_v2_frame_count',scene.frame_end));frames=list(range(T)) if frames is None else list(frames)
    report={};prev={};samples={};model_by_side={};angles={};palm_tracks={};last_frame={};corrections={}
    activities={};bone_fingers={};contact={}
    for side,track in tracks.items():
        model_by_side[side]=HandModel(rig,side,-1. if side=='L' else 1.)
        model_by_side[side].image_size=np.array([scene.render.resolution_x,scene.render.resolution_y])
        activities[side]=finger_activity(track,model_by_side[side].image_size)
        contact[side]=head_contact_motion(track,side,model_by_side[side].image_size,body,face)
        bone_fingers.update({name:i//3 for i,name in enumerate(model_by_side[side].names)})
        track.setdefault('arm_flex_reference',np.full(T,np.nan))
        track.setdefault('arm_pole_reference',np.full((T,3),np.nan))
        track.setdefault('hand_depth_reference',np.full(T,np.nan))
        track.setdefault('hand_span_reference',np.full(T,np.nan))
        from hand_visibility import approach_depth
        if 'edge' in track:
            track['approach_depth']=approach_depth(track['uv'],track['world'],track['arm_flex_reference'],track['observed'])
        palm_tracks[side]=supported_palm_rotations(track['world'],track['valid'],track['observed'])
        for name in ['Arm_'+side,'Elbow_'+side,'Wrist_'+side]+model_by_side[side].names:samples[name]={}
    hinges={side:arm_reference_frames(scene,rig,side,track) for side,track in tracks.items()}
    last_pole={}
    for n,t in enumerate(frames):
        scene.frame_set(t+1);bpy.context.view_layer.update()
        if n%60==0:print(f'V2_PROGRESS 最终机位手部联合校正 {n+1}/{len(frames)}',flush=True)
        cam=scene.camera;K=intrinsics(scene,cam)
        cv_to_world=np.array(cam.matrix_world.to_3x3())@np.diag([1.,-1.,-1.]);camera_origin=np.array(cam.matrix_world.translation)
        for side,track in tracks.items():
            if not track['valid'][t]:continue
            source=track['world'][t];uv=track['uv'][t]
            predicted=not track['observed'][t]
            if not np.isfinite(source).all() or (not predicted and ((uv>=0)&(uv<=1)).all(1).sum()<3):continue
            A,E,W=[rig.pose.bones[p+'_'+side] for p in ('Arm','Elbow','Wrist')]
            sp,ep,wp=[np.array(rig.matrix_world@b.head) for b in (A,E,W)]
            L1,L2=np.linalg.norm(ep-sp),np.linalg.norm(wp-ep)
            if not np.isfinite(track['arm_flex_reference'][t]):
                track['arm_flex_reference'][t]=np.arccos(np.clip((ep-sp)@(wp-ep)/(L1*L2),-1,1))
            if not np.isfinite(track['arm_pole_reference'][t]).all():
                track['arm_pole_reference'][t]=np.array(rig.matrix_world.to_3x3().inverted())@(ep-sp)
            flex_reference=float(track['arm_flex_reference'][t])
            shoulder=(sp-camera_origin)@cv_to_world;wrist=(wp-camera_origin)@cv_to_world
            if not np.isfinite(track['hand_depth_reference'][t]):
                track['hand_depth_reference'][t]=wrist[2]
                palm_points=np.array([rig.matrix_world@rig.pose.bones[n+'1_'+side].head for n in ['IndexFinger','LittleFinger']])
                pc=(palm_points-camera_origin)@cv_to_world
                puv=pc[:,:2]/pc[:,2:3]*[K[0,0],K[1,1]]+K[:2,2]
                track['hand_span_reference'][t]=np.linalg.norm(puv[0]-puv[1])
            model=model_by_side[side]
            if last_frame.get(side)!=t-1:
                angles.pop(side,None);corrections.pop(side,None)
            image_motion=0.
            if t>0 and track['valid'][t-1]:
                image_motion=float(np.mean(np.linalg.norm(uv-track['uv'][t-1],axis=1)))
            temporal_weight=1.8/(1+(image_motion/.018)**2)
            follow,depth_delta=contact[side][0][t],contact[side][1][t]
            activity=activities[side][t]*(1.-follow)
            temporal_weight=temporal_weight*(1.-.85*np.repeat(activity,[6,4,4,4,4]))
            depth_factor=track['approach_depth'][t] if 'approach_depth' in track else 1.
            if 'relative_palm_scale' in track and track['relative_palm_scale'][t]!=1. and track['edge'][t]<0:
                # The base arm solve already carries part of the perspective change.
                depth_factor=track['relative_palm_scale'][t]**-.60
            goal,Rw,locals_,angles[side],cost=fit_frame(model,source,uv,K,shoulder,(L1,L2),wrist,flex_reference,angles.get(side),predicted,
                track['hand_depth_reference'][t]*depth_factor,track['hand_span_reference'][t],palm_tracks[side][t],temporal_weight,
                previous_correction=corrections.get(side),activity=activity,
                digit_weights=track['digit_weights'][t] if 'digit_weights' in track else None,
                parts_only='part_anchor' in track and track['part_anchor'][t]>=0)
            if follow>0.:
                # The palm base moves with the wrist. Fingertips can be concealed
                # in hair and must not determine wrist position or axial swing.
                bases=[5,9,13,17]
                ray=np.r_[(uv[0]*model.image_size-K[:2,2])/K[[0,1],[0,1]],1.]
                goal=(1-follow)*goal+follow*reachable_wrist(ray,shoulder,goal[2]+L2*depth_delta,
                    np.linalg.norm(goal-shoulder),(L1+L2)*.985)
                palm_points=model.rest_points[bases]@Rw.T+goal
                projected=(palm_points[:,:2]/palm_points[:,2:]-goal[:2]/goal[2])*K[[0,1],[0,1]]
                current=unit(projected.mean(axis=0))
                target=unit((uv[bases].mean(axis=0)-uv[0])*model.image_size)
                twist=np.arctan2(current[0]*target[1]-current[1]*target[0],current@target)
                Rw=rotvec(np.array([0.,0.,follow*np.clip(twist,-.18,.18)]))@Rw
            if 'occlusion_scale' in track and track['occlusion_scale'][t]>1.:
                scale=track['occlusion_scale'][t]
                ray=np.r_[(uv[0]*model.image_size-K[:2,2])/K[[0,1],[0,1]],1.]
                # A growing foreground arm can extend along the camera ray.
                # Keep the two physical bone lengths and a soft elbow bend.
                depth=track['hand_depth_reference'][t]/scale
                direction=unit(ray)
                along=direction@shoulder
                lateral=np.linalg.norm(shoulder-direction*along)
                reach=(L1+L2)*.989
                if lateral<reach:
                    extent=np.sqrt(reach**2-lateral**2)
                    target=direction*np.clip(depth/direction[2],max(.07,along-extent),along+extent)
                    goal+=(target-goal)*min(1.,(scale-1.)/.4)
            observation_rotation=palm_tracks[side][t]@model.rest_palm.T@model.wrist_rest
            corrections[side]=observation_rotation.T@Rw
            continuous=last_frame.get(side)==t-1
            last_frame[side]=t
            goal=camera_origin+cv_to_world@goal
            direction=unit(goal-sp);distance=np.clip(np.linalg.norm(goal-sp),abs(L1-L2)+1e-5,L1+L2-1e-5)
            reference_pole=np.array(rig.matrix_world.to_3x3())@track['arm_pole_reference'][t]
            pole=reference_pole-direction*(reference_pole@direction)
            if np.linalg.norm(pole)<1e-6:pole=np.array(rig.matrix_world.to_3x3())@np.array([0.,-1.,0.]);pole-=direction*(pole@direction)
            pole=unit(pole)
            if continuous and side in last_pole:
                prior=last_pole[side]-direction*(direction@last_pole[side])
                if np.linalg.norm(prior)>.1:
                    prior=unit(prior)
                    angle=np.arctan2(direction@np.cross(prior,pole),prior@pole)
                    pole=rotvec(direction*np.clip(angle*.55,-.18,.18))@prior
            along=(L1*L1-L2*L2+distance*distance)/(2*distance)
            if predicted and body is not None and track['edge'][t]==3:
                source_y=body['keypoints_2d'][t,7 if side=='L' else 8,1]/model.image_size[1]
                if source_y>1. and uv[0,1]>=.98:
                    # Pick an elbow on the feasible bend circle that remains
                    # below the image when the source forearm has departed.
                    plane=cv_to_world@np.array([0.,1.,-(model.image_size[1]-K[1,2])/K[1,1]])
                    below=plane-direction*(plane@direction)
                    radius=np.sqrt(max(0,L1*L1-along*along))
                    mid=sp+direction*along
                    if (mid+pole*radius-camera_origin)@plane<0 and np.linalg.norm(below)>1e-5:
                        pole=unit(below)
            last_pole[side]=pole
            elbow=sp+direction*along+pole*np.sqrt(max(0,L1*L1-along*along))
            invR=np.array(rig.matrix_world.to_3x3().inverted())
            if t in hinges[side]:
                normal=unit(np.cross(elbow-sp,goal-elbow))
                ar,er=hinges[side][t]
                AR=hinge_basis(elbow-sp,normal)@ar
                ER=hinge_basis(goal-elbow,normal)@er
            else:
                AR=between(ep-sp,elbow-sp)@np.array(rig.matrix_world.to_3x3()@A.matrix.to_3x3())
                ER=between(wp-ep,goal-elbow)@np.array(rig.matrix_world.to_3x3()@E.matrix.to_3x3())
            weight=float(track.get('confidence',np.ones(T))[t])
            for bone,D in [(A,invR@AR),(E,invR@ER),(W,invR@cv_to_world@Rw)]:
                parent=bone.parent;inherited=parent.matrix.to_3x3()@parent.bone.matrix_local.to_3x3().inverted()@bone.bone.matrix_local.to_3x3()
                q=(inherited.inverted()@Matrix(D)).to_quaternion().normalized()
                q=bone.rotation_quaternion.slerp(q,weight)
                key_quat(bone,q,t+1,prev);bpy.context.view_layer.update()
            for name,local in zip(model.names,locals_):
                bone=rig.pose.bones[name];q=Matrix(local).to_quaternion().normalized()
                q=bone.rotation_quaternion.slerp(q,weight);key_quat(bone,q,t+1,prev)
            for name in ['Arm_'+side,'Elbow_'+side,'Wrist_'+side]+model.names:samples[name][t]=list(rig.pose.bones[name].rotation_quaternion)
            report.setdefault(side,[]).append({'frame':t+1,'fit_cost':cost,
                'reference_elbow_flex_deg':float(np.degrees(flex_reference)),
                'elbow_flex_deg':float(np.degrees(np.arccos(np.clip((elbow-sp)@(goal-elbow)/(L1*L2),-1,1))))})
    # Exactly one bounded symmetric pass for fitted hands. Do not mix separate
    # observed episodes or pull an unseen/entering hand toward an unrelated pose.
    for name,rows in samples.items():
        ids=sorted(rows);runs=[]
        for t in ids:
            if not runs or t!=runs[-1][-1]+1:runs.append([])
            runs[-1].append(t)
        for run in runs:
            values=np.array([rows[t] for t in run])
            # Reject isolated pose flips only when both neighbouring poses
            # agree. Real sustained motion is retained at its original time.
            clean=values.copy()
            for k in range(1,len(run)-1):
                a,b,c=[Quaternion(values[j].tolist()) for j in [k-1,k,k+1]]
                if min(a.rotation_difference(b).angle,b.rotation_difference(c).angle)>np.radians(40.) and a.rotation_difference(c).angle<np.radians(30.):
                    clean[k]=a.slerp(c,.5)
            is_hand=not name.startswith(('Arm_','Elbow_'))
            filtered=smooth_rotations(clean,scene.render.fps/scene.render.fps_base,
                limit_degrees=28. if is_hand else 3.,range_degrees=75. if is_hand else 20.)
            if is_hand:
                track=tracks[name[-1]]
                for k,t in enumerate(run):
                    lo=max(run[0],t-1);hi=min(run[-1],t+1)
                    # Opposing steps must not cancel at the peak of a wave.
                    steps=np.diff(track['uv'][lo:hi+1],axis=0)
                    motion=float(np.max(np.percentile(np.linalg.norm(steps,axis=2),75,axis=1))) if len(steps) else 0.
                    # Fast image-supported motion retains its amplitude; when
                    # the image is stationary, depth/IK noise gets more cleanup.
                    amount=.65+.35/(1+(motion/.025)**2)
                    follow=contact[name[-1]][0][t]
                    if name in bone_fingers:
                        amount*=1.-.85*(1.-follow)*activities[name[-1]][t,bone_fingers[name]]
                    else:
                        amount*=1.-.90*follow
                    filtered[k]=Quaternion(clean[k].tolist()).slerp(Quaternion(filtered[k].tolist()),amount)
            else:
                for k,t in enumerate(run):
                    weight=1.-.90*contact[name[-1]][0][t]
                    filtered[k]=Quaternion(clean[k].tolist()).slerp(Quaternion(filtered[k].tolist()),weight)
            if name.startswith('Wrist_') and 'edge' in tracks[name[-1]]:
                filtered=ease_border_turns(filtered,run,tracks[name[-1]],scene.render.fps/scene.render.fps_base)
            for t,q in zip(run,filtered):key_quat(rig.pose.bones[name],Quaternion(q.tolist()),t+1,prev)
    from hand_exit_ik import apply_contours
    contour_report=apply_contours(scene,rig,tracks,frames)
    if contour_report:report['edge_contours']=contour_report
    if frames==list(range(T)):
        blend_hidden_transitions(scene,rig,tracks)
        from arm_entry import relax_entries,release_lower_exits
        report['relaxed_entry_frames']=relax_entries(scene,rig,tracks)
        release_lower_exits(scene,rig,tracks)
    scene.frame_set(scene.frame_current)
    return report


def blend_hidden_transitions(scene,rig,tracks):
    """Join the fitted pose to the unseen-arm animation on the hidden side."""
    from calibrated_bridge import key_quat
    length=max(4,round(scene.render.fps/scene.render.fps_base*.4))
    for side,track in tracks.items():
        valid=track['valid']
        bones=[rig.pose.bones[k+'_'+side] for k in ('Arm','Elbow','Wrist')]
        bones.extend(rig.pose.bones[f'{stem}{k if stem=="Thumb" else k+1}_{side}']
                     for stem,_ in STEMS for k in range(3))
        gaps=np.flatnonzero(~valid & ~np.r_[False,~valid[:-1]])
        for a in gaps:
            b=a
            while b+1<len(valid) and not valid[b+1]:
                b+=1
            if a==0 and b==len(valid)-1:
                continue
            if b-a+1<=2*length and a>0 and b+1<len(valid):
                segments=[(a-1,b+1)]
            else:
                segments=[]
                if a>0:segments.append((a-1,min(a+length,b)))
                if b+1<len(valid):segments.append((max(b-length,a),b+1))
            for start,end in segments:
                scene.frame_set(start+1)
                first=[bone.rotation_quaternion.copy() for bone in bones]
                scene.frame_set(end+1)
                last=[bone.rotation_quaternion.copy() for bone in bones]
                previous={}
                for t in range(start+1,end):
                    u=(t-start)/(end-start)
                    u=u*u*(3-2*u)
                    for bone,q,r in zip(bones,first,last):
                        key_quat(bone,q.slerp(r,u),t+1,previous)


def save_tracks(path,tracks):
    np.savez_compressed(path,**{side+'_'+key:value for side,track in tracks.items() for key,value in track.items()})


def load_tracks(path):
    with np.load(path) as data:
        return {side:{key[len(side)+1:]:data[key].copy() for key in data.files if key.startswith(side+'_')}
                for side in ('L','R') if side+'_uv' in data.files}


def evaluate(scene,rig,tracks,frames=None):
    """Measure actual final bones through the current presentation camera."""
    import bpy
    T=scene.frame_end;frames=range(T) if frames is None else frames
    report={side:[] for side in tracks}
    for t in frames:
        scene.frame_set(t+1);bpy.context.view_layer.update();cam=scene.camera;K=intrinsics(scene,cam)
        R=np.array(cam.matrix_world.to_3x3())@np.diag([1.,-1.,-1.]);origin=np.array(cam.matrix_world.translation)
        for side,track in tracks.items():
            if not track['observed'][t]:continue
            points=[rig.pose.bones['Wrist_'+side].head.copy()]
            for stem,start in STEMS:
                bones=[rig.pose.bones[f'{stem}{k if stem=="Thumb" else k+1}_{side}'] for k in range(3)]
                points.extend([b.head.copy() for b in bones]);points.append(bones[-1].tail.copy())
            points=np.array([rig.matrix_world@p for p in points]);pc=(points-origin)@R
            uv=pc[:,:2]/np.maximum(pc[:,2:3],1e-5)*[K[0,0],K[1,1]]+K[:2,2]
            normalized=uv/[scene.render.resolution_x,scene.render.resolution_y]
            source=track['uv'][t];inside=((source>=0)&(source<=1)).all(1)
            error=np.linalg.norm((normalized-source)*[scene.render.resolution_x,scene.render.resolution_y],axis=1)
            report[side].append(dict(frame=t+1,median_px=float(np.median(error[inside])) if inside.any() else None,
                mean_px=float(np.mean(error[inside])) if inside.any() else None,
                index_tip_error_px=float(error[8]),source_uv=source.tolist(),output_uv=normalized.tolist()))
    return report

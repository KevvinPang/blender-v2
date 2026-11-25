"""Absolute SMPL-H FK to Ellen, with checked rest frames and image calibration."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import bpy
from mathutils import Matrix,Vector,Quaternion
from face_controls import load_face, head_target, apply_face, correct_head_roll
from temporal_denoise import apply_to_scene as denoise_final_motion
from presentation_camera import ensure_presentation_camera

C=np.array([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])
F=np.diag([1.,-1.,-1.])
MAP={0:'Waist',3:'UpperBody',6:'UpperBody1',9:'UpperBody2',12:'Neck',15:'Head',
     13:'Shoulder_L',14:'Shoulder_R',16:'Arm_L',17:'Arm_R',18:'Elbow_L',19:'Elbow_R',20:'Wrist_L',21:'Wrist_R'}
OUT={3:6,6:9,9:12,12:15,13:16,14:17,16:18,17:19,18:20,19:21,20:25,21:40}
for side,start in [('L',22),('R',37)]:
    for fidx,name in enumerate(['IndexFinger','MiddleFinger','LittleFinger','RingFinger','Thumb']):
        for k in range(3):
            j=start+fidx*3+k;MAP[j]=f'{name}{k if name=="Thumb" else k+1}_{side}'
            if k<2:OUT[j]=j+1

def aa(v):
    a=float(np.linalg.norm(v))
    return np.eye(3) if a<1e-10 else np.asarray(Quaternion(Vector(v.tolist())/a,a).to_matrix())

def fk(motion,model):
    poses=motion['poses'];betas=motion['betas'].reshape(-1)
    verts=model['v_template']+np.einsum('vck,k->vc',model['shapedirs'][...,:len(betas)],betas)
    rest=model['J_regressor']@verts
    parents=model['kintree_table'][0].astype(np.int64);parents[0]=-1
    G=np.zeros((len(poses),52,3,3));J=np.zeros((len(poses),52,3))
    for t in range(len(poses)):
        for j in range(52):
            local=aa(poses[t,j]);p=int(parents[j])
            if p<0:G[t,j]=local;J[t,j]=rest[j]
            else:G[t,j]=G[t,p]@local;J[t,j]=J[t,p]+G[t,p]@(rest[j]-rest[p])
    return rest,parents,G,J

def rigid_rotation(src,dst):
    u,_,vt=np.linalg.svd((src-src.mean(0)).T@(dst-dst.mean(0)))
    return vt.T@np.diag([1.,1.,np.linalg.det(vt.T@u.T)])@u.T

def fit_translation(points,uv,K,weights):
    q=(uv-K[:2,2])/np.array([K[0,0],K[1,1]]);A=[];b=[]
    for (x,y,z),(u,v),w in zip(points,q,weights):
        A.extend([np.array([1,0,-u])*w,np.array([0,1,-v])*w]);b.extend([(u*z-x)*w,(v*z-y)*w])
    tr=np.linalg.lstsq(A,b,rcond=None)[0]
    if np.min((points+tr)[:,2])<=.03:raise ValueError('Calibration places anchors behind camera')
    return tr

def fit_reachable_hand(offsets,uv,K,shoulder,radius,initial,weights):
    """Fit visible fingers/palm, with a reach penalty; hidden wrist has no vote."""
    tr=initial.copy();fx,fy=K[0,0],K[1,1]
    for iteration in range(30):
        p=offsets+tr;z=np.maximum(p[:,2],.03)
        pred=p[:,:2]/z[:,None]*[fx,fy]+K[:2,2]
        residual=((pred-uv)*weights[:,None]).reshape(-1)
        jac=np.zeros((len(p),2,3));jac[:,0,0]=fx/z;jac[:,1,1]=fy/z
        jac[:,0,2]=-fx*p[:,0]/z**2;jac[:,1,2]=-fy*p[:,1]/z**2
        jac=(jac*weights[:,None,None]).reshape(-1,3)
        delta=tr-shoulder;distance=np.linalg.norm(delta)
        if distance>radius:
            residual=np.append(residual,(distance-radius)*10000)
            jac=np.vstack([jac,delta/distance*10000])
        step=np.linalg.solve(jac.T@jac+np.eye(3)*10,jac.T@residual)
        if np.linalg.norm(step)>.05:step*=.05/np.linalg.norm(step)
        tr-=step
        if np.linalg.norm(step)<1e-6:break
    delta=tr-shoulder;distance=np.linalg.norm(delta)
    if distance>radius:tr=shoulder+delta*radius/distance
    return tr

def upright_rotation(rotation):
    """Remove camera roll while preserving its viewing direction."""
    forward=Vector((rotation@np.array((0.,0.,-1.))).tolist()).normalized()
    up=Vector((0.,0.,1.))
    right=forward.cross(up)
    if right.length<1e-5:
        up=Vector((0.,1.,0.));right=forward.cross(up)
    right.normalize();up=right.cross(forward).normalized()
    return np.asarray(Matrix(((right.x,up.x,-forward.x),
                              (right.y,up.y,-forward.y),
                              (right.z,up.z,-forward.z))).to_3x3())

def display_intrinsics(calibration,width,height):
    lens=float(calibration.get('lens',50.))
    sensor=float(calibration.get('sensor_width',36.))
    fx=lens*width/sensor
    fy=fx
    sx=float(calibration.get('shift_x',0.));sy=float(calibration.get('shift_y',0.))
    return np.array([[fx,0.,width*.5-sx*width],
                     [0.,fy,height*.5+sy*height],[0.,0.,1.]])

def limited_direction(reference,target,limit_degrees):
    a=reference.normalized();b=target.normalized();q=a.rotation_difference(b)
    angle=q.angle
    if angle>np.radians(limit_degrees):
        q=Quaternion((1,0,0,0)).slerp(q,float(np.radians(limit_degrees)/angle))
    return (q@a).normalized()

def preserve_arm_shape(shoulder,elbow,wrist,goal,suggested_elbow,limit_degrees=30.):
    """Image fitting is a bounded correction, never a replacement for 3D pose.

    Preserve upper-arm direction, elbow flexion and elbow plane separately.
    A true punch in the source can remain straight; a bent source arm cannot
    become straight merely because the image-space target is far away.
    """
    u0=(elbow-shoulder).normalized();v0=(wrist-elbow).normalized()
    L1=(elbow-shoulder).length;L2=(wrist-elbow).length
    u=limited_direction(u0,suggested_elbow-shoulder,limit_degrees)
    new_elbow=shoulder+u*L1
    vref=u0.rotation_difference(u)@v0
    want=(goal-new_elbow).normalized()
    flex0=np.arccos(np.clip(u0.dot(v0),-1,1))
    flexwant=np.arccos(np.clip(u.dot(want),-1,1))
    flex=float(np.clip(flexwant,max(0.,flex0-np.radians(limit_degrees)),min(np.pi,flex0+np.radians(limit_degrees))))
    plane0=vref-u*vref.dot(u);plane1=want-u*want.dot(u)
    if plane0.length<1e-5:
        plane0=u.orthogonal()
    if plane1.length<1e-5:plane1=plane0.copy()
    plane=limited_direction(plane0,plane1,limit_degrees)
    v=(u*np.cos(flex)+plane*np.sin(flex)).normalized()
    return new_elbow,new_elbow+v*L2

def key_quat(b,q,frame,previous):
    if b.name in previous and q.dot(previous[b.name])<0:q.negate()
    previous[b.name]=q.copy();b.rotation_mode='QUATERNION';b.rotation_quaternion=q
    b.location=(0,0,0);b.scale=(1,1,1)
    b.keyframe_insert(data_path='rotation_quaternion',frame=frame,group=b.name)

def palm_basis(points,indices):
    wrist,index,middle,pinky=[points[i] for i in indices]
    y=middle-wrist;y/=np.linalg.norm(y)
    x=index-pinky;x-=y*np.dot(x,y);x/=np.linalg.norm(x)
    return np.stack([x,y,np.cross(x,y)],axis=1)


def stabilize_baked_hands(scene,rig,T):
    """Filter the final local rotations, including IK and detector handoffs.

    A symmetric three-frame filter has no causal lag on sustained movement.
    Only an isolated rotation excursion unsupported by both neighbors is
    replaced; sustained fast rotations are not treated as outliers.
    """
    names=[n for n in MAP.values() if n.startswith(('Arm_','Elbow_','Wrist_','Thumb','IndexFinger','MiddleFinger','RingFinger','LittleFinger'))]
    samples={n:[] for n in names}
    for t in range(T):
        scene.frame_set(t+1)
        for n in names:samples[n].append(rig.pose.bones[n].rotation_quaternion.copy().normalized())
    report={};previous={}
    for n,raw in samples.items():
        for i in range(1,T):
            if raw[i-1].dot(raw[i])<0:raw[i].negate()
        clean=[q.copy() for q in raw];rejected=0
        for i in range(1,T-1):
            ab=raw[i-1].rotation_difference(raw[i]).angle
            bc=raw[i].rotation_difference(raw[i+1]).angle
            ac=raw[i-1].rotation_difference(raw[i+1]).angle
            if min(ab,bc)>np.radians(45.) and ac<np.radians(25.):
                clean[i]=raw[i-1].slerp(raw[i+1],.5);rejected+=1
        filtered=[q.copy() for q in clean]
        for i in range(1,T-1):
            a=clean[i-1].copy();b=clean[i+1].copy()
            if clean[i].dot(a)<0:a.negate()
            if clean[i].dot(b)<0:b.negate()
            neighbors=a.slerp(b,.5)
            filtered[i]=clean[i].slerp(neighbors,.4).normalized()
        before=[np.degrees(raw[i-1].rotation_difference(raw[i]).angle) for i in range(1,T)]
        after=[np.degrees(filtered[i-1].rotation_difference(filtered[i]).angle) for i in range(1,T)]
        report[n]={'spikes_replaced':rejected,'before_max_step_deg':float(max(before,default=0)),
                   'after_max_step_deg':float(max(after,default=0)),
                   'after_p99_step_deg':float(np.percentile(after,99)) if after else 0.}
        for t,q in enumerate(filtered):key_quat(rig.pose.bones[n],q,t+1,previous)
    return report

def smooth_hand_track(track,uvtrack,valid,max_gap=10):
    """Reject single-frame detector spikes and fill only short gaps.

    The detector is allowed to miss a fast hand for a few frames, but a long
    occlusion must remain a model-driven fallback.  A median residual filter
    removes landmark jitter without averaging a fast movement into a visibly
    delayed one.
    """
    observed=valid.copy()
    # Only reject isolated, unsupported detector spikes. Do not average a
    # sustained fast gesture, and never read already-filtered neighbours.
    raw=track.copy();raw_uv=uvtrack.copy()
    for t in range(1,len(valid)-1):
        if not valid[t-1:t+2].all():continue
        midpoint=(raw[t-1]+raw[t+1])*.5
        uv_midpoint=(raw_uv[t-1]+raw_uv[t+1])*.5
        unsupported=(np.linalg.norm(raw_uv[t]-uv_midpoint,axis=1)>.08)&(np.linalg.norm(raw_uv[t+1]-raw_uv[t-1],axis=1)<.025)
        track[t,unsupported]=midpoint[unsupported]
        uvtrack[t,unsupported]=uv_midpoint[unsupported]
    valid_indices=np.flatnonzero(valid)
    for a,b in zip(valid_indices,valid_indices[1:]):
        gap=b-a-1
        if gap<=0 or gap>max_gap:
            continue
        for t in range(a+1,b):
            u=(t-a)/(b-a)
            track[t]=track[a]*(1.-u)+track[b]*u
            uvtrack[t]=uvtrack[a]*(1.-u)+uvtrack[b]*u
            valid[t]=True
    return observed

def hand_targets(path,rig,T,camera_R,K,width,height,body_uv=None,face=None,parts=None):
    """Use visible hand geometry; keep FlowHMR as the missing-hand fallback.

    Handedness scores are association scores, not landmark confidence. The
    geometric checks below reject tiny/degenerate detections separately.
    """
    if path is None:return {},{},{},{},{}
    data=np.load(path);targets={};stats={};blends={};wrist_goals={};tracks={}
    assigned={'L':[None]*T,'R':[None]*T}
    singleton=[]
    for t in range(T):
        candidates=[]
        for h in range(2):
            uv=data['normalized_landmarks'][t,h,:,:2];pts=data['world_landmarks'][t,h]
            if data['scores'][t,h]<.55 or not np.isfinite(pts).all():continue
            if np.ptp(uv,axis=0).max()<.08 or np.linalg.norm(np.cross(pts[5]-pts[0],pts[17]-pts[0]))<1e-5:continue
            candidates.append(h)
        singleton.append(candidates[0] if len(candidates)==1 else None)
        if len(candidates)==2 and body_uv is not None and np.isfinite(body_uv[t,[41,62]]).all():
            d=np.array([[np.linalg.norm(data['normalized_landmarks'][t,h,0,:2]*[width,height]-body_uv[t,w])
                         for w in [62,41]] for h in candidates])
            direct=d[0,0]+d[1,1];swapped=d[0,1]+d[1,0]
            if swapped<direct:
                assigned['L'][t]=candidates[1];assigned['R'][t]=candidates[0]
            else:
                assigned['L'][t]=candidates[0];assigned['R'][t]=candidates[1]
        elif candidates:
            if body_uv is not None and np.isfinite(body_uv[t,[41,62]]).all():
                h=min(candidates,key=lambda h:min(
                    np.linalg.norm(data['normalized_landmarks'][t,h,0,:2]*[width,height]-body_uv[t,62]),
                    np.linalg.norm(data['normalized_landmarks'][t,h,0,:2]*[width,height]-body_uv[t,41])))
                dl=np.linalg.norm(data['normalized_landmarks'][t,h,0,:2]*[width,height]-body_uv[t,62])
                dr=np.linalg.norm(data['normalized_landmarks'][t,h,0,:2]*[width,height]-body_uv[t,41])
                assigned['L' if dl<=dr else 'R'][t]=h
            else:
                h=candidates[0];label=str(data['labels'][t,h])
                assigned['L' if label=='Left' else 'R'][t]=h
    # Anatomical identity cannot switch midway through a continuous one-hand
    # track just because an occluded body wrist briefly moves closer to it.
    start=0
    while start<T:
        if singleton[start] is None:
            start+=1;continue
        end=start+1
        while end<T and singleton[end] is not None:
            previous=data['normalized_landmarks'][end-1,singleton[end-1],0,:2]
            current=data['normalized_landmarks'][end,singleton[end],0,:2]
            if np.linalg.norm(current-previous)>.25:break
            end+=1
        votes={side:sum(assigned[side][i] is not None for i in range(start,end)) for side in assigned}
        side=max(votes,key=votes.get)
        for i in range(start,end):
            assigned[side][i]=singleton[i];assigned['R' if side=='L' else 'L'][i]=None
        start=end
    for side,label in [('L','Left'),('R','Right')]:
        track=np.full((T,21,3),np.nan);uvtrack=np.full((T,21,2),np.nan);valid=np.zeros(T,dtype=bool)
        for t in range(T):
            h=assigned[side][t]
            if h is None:continue
            track[t]=data['world_landmarks'][t,h]
            uvtrack[t]=data['normalized_landmarks'][t,h,:,:2]
            valid[t]=True
        from hand_visibility import fuse,apply_contour
        from hand_occlusion import unsupported_face_track
        rejected=unsupported_face_track(uvtrack,face,body_uv,np.array([width,height]))
        valid[rejected]=False
        uvtrack[rejected]=np.nan
        track[rejected]=np.nan
        raw_uv=uvtrack.copy()
        from hand_occlusion import recover_contact_pose
        repaired=recover_contact_pose(track,uvtrack,valid,face,
            body_uv[:,62 if side=='L' else 41] if body_uv is not None else None,np.array([width,height]))
        smooth_hand_track(track,uvtrack,valid,max_gap=0)
        observed_valid=valid.copy()
        # Reject a newly appearing second hand supported only by an isolated
        # one/two-frame detection. The detector sometimes emits duplicates of
        # one visible hand; do not animate the other arm from such a fragment.
        starts=np.flatnonzero(valid & ~np.r_[False,valid[:-1]])
        ends=np.flatnonzero(valid & ~np.r_[valid[1:],False])
        for start,end in zip(starts,ends):
            if observed_valid[start:end+1].sum()<4:
                valid[start:end+1]=False;observed_valid[start:end+1]=False
        # Being outside the image reduces individual point weights in the
        # final solver. It must not freeze the still-visible hand as a whole.
        forward=data['flow_forward'] if 'flow_forward' in data else None
        backward=data['flow_backward'] if 'flow_backward' in data else None
        from hand_visibility import stabilize_cropped_observations
        stabilize_cropped_observations(track,uvtrack,valid & ~repaired,forward,backward)
        from hand_perspective import resolve_perspective
        propagated,relative_scale=resolve_perspective(track,uvtrack,valid,raw_uv,face,
            forward,backward,np.array([width,height]))
        from partial_digits import recover_digits
        video=next(Path(path).parent.glob('*_30fps.mp4'),None)
        digit_weights,partial_gap=recover_digits(video,track,uvtrack,valid,observed_valid & ~repaired,
                                     forward,backward,np.array([width,height]))
        partial=digit_weights.any(axis=1)
        digit_uv,digit_pose=uvtrack[partial].copy(),track[partial].copy()
        observed_valid,confidence,edge,held,corrected=fuse(track,uvtrack,valid,forward,backward)
        boundary=np.full(T,np.nan)
        if 'contour_edge' in data:
            boundary=apply_contour(track,uvtrack,valid,observed_valid,confidence,edge,held,
                data['contour_edge'],data['contour_boundary'],raw_uv)
        # A whole-hand exit prediction cannot replace a tracked visible digit
        # or the short interpolation that connects it to a measured hand.
        uvtrack[partial],track[partial]=digit_uv,digit_pose
        observed_valid[partial]=True
        held[partial]=False
        edge[partial]=-1
        boundary[partial]=np.nan
        confidence[partial]=1.
        observed_valid[repaired]=False
        edge_holds=int(held.sum())
        interpolated=valid & ~held
        starts=np.flatnonzero(observed_valid & ~np.r_[False,valid[:-1]])
        for start in starts:
            if start==0 or np.isfinite(boundary[start]):continue
            for k in range(4):
                t=start+k
                if t>=T or not interpolated[t]:break
                confidence[t]=min(confidence[t],(k+1)/4.)
        stats[side]={'observed_frames':int(observed_valid.sum()),
                     'filled_frames':int(np.sum(interpolated & ~observed_valid)),
                     'exit_transition_frames':int(np.sum(valid & ~observed_valid & held)),
                     'edge_articulation_held_frames':edge_holds}
        stats[side]['unsupported_face_frames']=int(rejected.sum())
        stats[side]['reconstructed_contact_frames']=int(repaired.sum())
        tracks[side]={'solver_revision':np.array(20 if parts is not None else 18),'world':track,'uv':uvtrack,'valid':valid.copy(),
                      'observed':observed_valid,'confidence':confidence.copy(),
                      'edge':edge,'boundary':boundary,'held':held,
                      'flow_supported':propagated,'relative_palm_scale':relative_scale,
                      'digit_weights':digit_weights,'digit_gap':partial_gap}
        if body_uv is not None:
            elbow=body_uv[:,7 if side=='L' else 8]/[width,height]
            tracks[side]['elbow_visible']=np.all((elbow>=0)&(elbow<=1),axis=1)
        from hand_visibility import edge_transitions
        edge_transitions(tracks[side],30.)
        observed_valid[propagated | partial | partial_gap]=False
        stats[side]['partial_digit_frames']=int((partial & ~partial_gap).sum())
        stats[side]['flow_supported_frames']=int(propagated.sum())
        stats[side]['observed_frames']=int(observed_valid.sum())
        from hand_occlusion import recover
        tracks[side]['occlusion_scale']=recover(tracks[side],raw_uv,face,forward,backward)
        if parts is not None:
            from partial_digits import apply_parts
            stats[side]['part_frames']=apply_parts(tracks[side],parts,assigned[side],raw_uv)
        valid=tracks[side]['valid']
        confidence=tracks[side]['confidence']
        for start in starts:
            if start>0 and valid[start-1] and edge[start-1]>=0:
                confidence[start:min(T,start+4)][observed_valid[start:min(T,start+4)]]=1.
        restpts=np.array([list(rig.data.bones[n].head_local) for n in
                          [f'Wrist_{side}',f'IndexFinger1_{side}',f'MiddleFinger1_{side}',f'LittleFinger1_{side}']])
        restframe=palm_basis(restpts,[0,1,2,3])
        for t in range(T):
            if not valid[t]:continue
            weight=float(confidence[t])
            pts=track[t]
            uv=uvtrack[t]*[width,height]
            palm_scale=np.linalg.norm(restpts[2]-restpts[0])/np.linalg.norm(pts[9]-pts[0])
            palm_ids=[0,5,9,13,17]
            pc=(pts[palm_ids]-pts[0])*palm_scale
            try:
                wrist_goals[t,side]=fit_translation(pc,uv[palm_ids],K[t],np.ones(5))
            except ValueError:
                pass
            world=pts@camera_R[t].T
            wrist=f'Wrist_{side}'
            D=palm_basis(world,[0,5,9,17])@restframe.T@np.asarray(rig.data.bones[wrist].matrix_local.to_3x3())
            targets[t,wrist]=D
            blends[t,wrist]=weight
            for stem,start in [('Thumb',1),('IndexFinger',5),('MiddleFinger',9),('RingFinger',13),('LittleFinger',17)]:
                for k in range(3):
                    name=f'{stem}{k if stem=="Thumb" else k+1}_{side}'
                    # The final anatomical solver calibrates joint zero poses.
                    # This provisional FK uses segment directions without the
                    # old projected-angle conversion that lost finger flexion.
                    targets[t,name]=world[start+k+1]-world[start+k]
                    blends[t,name]=weight
    return targets,stats,blends,wrist_goals,tracks

def main():
    p=argparse.ArgumentParser();p.add_argument('motion');p.add_argument('output')
    p.add_argument('--camera-track',required=True,help='Raw VGGT npz')
    p.add_argument('--observations',required=True)
    p.add_argument('--hands')
    p.add_argument('--rig-name',default='艾莲_arm')
    p.add_argument('--preserve-expressions',action='store_true',default=True)
    p.add_argument('--preserve-hand-presentation',action='store_true',
                   help='Body-only refinement: keep accepted hands if source observations match exactly')
    p.add_argument('--replace-face-animation',action='store_false',dest='preserve_expressions',
                   help='Explicitly replace existing face animation with observed blinks/jaw')
    p.add_argument('--face-observations')
    p.add_argument('--hamer-fingers')
    p.add_argument('--hand-parts')
    p.add_argument('--gaze-mode',choices=['video','legacy_camera'],default='video')
    p.add_argument('--gaze-reference-frame',type=int,help='User-labelled frame looking at the source lens')
    p.add_argument('--head-mode',choices=['body','face'],default='body')
    p.add_argument('--framing-calibration',help='JSON exported from a user-adjusted child camera')
    p.add_argument('--framing',choices=['face','chest_crop','portrait'],default='face')
    p.add_argument('--portrait-reference-frame',type=int,default=33)
    p.add_argument('--portrait-eye-span',type=float,default=.275,help='Eye anchor spacing / image width')
    p.add_argument('--portrait-eye-height',type=float,default=.46,help='Upper-eye anchor centre / image height from top')
    p.add_argument('--model',default='C:/mocap_v2_models/smplh/neutral/model.npz')
    p.add_argument('--frames',default='1,45,90');p.add_argument('--render',action='store_true')
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    output=Path(args.output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    if output==Path(bpy.data.filepath).resolve():raise ValueError('Output must be a separate project')
    motion=np.load(args.motion);model=np.load(args.model,allow_pickle=False)
    obs=np.load(args.observations);raw=np.load(args.camera_track)
    rest,parents,G,J=fk(motion,model);T=len(G)
    face=load_face(args.face_observations,T)
    source_indices=np.asarray(obs['frame_indices'],dtype=int)
    if len(source_indices)<2 or source_indices.min()<0 or source_indices.max()>=T:
        raise ValueError('Observation frame indices do not align with the motion')
    if not np.array_equal(source_indices,np.arange(T)):
        interpolated={'frame_indices':np.arange(T)}
        for key in ('keypoints_2d','keypoints_3d'):
            values=np.asarray(obs[key]);interpolated[key]=np.stack([
                np.interp(np.arange(T),source_indices,values[:,j,d])
                for j in range(values.shape[1]) for d in range(values.shape[2])
            ],axis=1).reshape((T,)+values.shape[1:])
        obs=interpolated
    K=raw['K'].copy();height,width=map(int,raw['image_size_hw'])
    # Video pixels are square; per-axis VGGT focal estimates contain fitting
    # noise. Use their geometric mean, rather than animating pixel aspect.
    intrinsics_anisotropy=float(np.max(np.abs(K[:,0,0]/K[:,1,1]-1)))
    focal=np.sqrt(K[:,0,0]*K[:,1,1]);K[:,0,0]=focal;K[:,1,1]=focal
    image_scale=1.;image_offset=np.zeros(2)
    if args.framing=='portrait':
        ref=int(np.clip(args.portrait_reference_frame-1,0,T-1))
        reference_eyes=obs['keypoints_2d'][ref,[1,2]]
        image_scale=float(args.portrait_eye_span*width/np.linalg.norm(reference_eyes[0]-reference_eyes[1]))
        image_offset=np.array([width*.5,height*args.portrait_eye_height])-reference_eyes.mean(0)*image_scale
    from actor_observations import dense_anchors,refit_final_actor
    placement_uv=np.array(obs['keypoints_2d']).copy();placement_report={'dense_face_used':False}
    if args.face_observations:
        placement_uv,placement_report=dense_anchors(args.face_observations,placement_uv,width,height,float(motion['mocap_framerate']))
    composition_uv=placement_uv*image_scale+image_offset
    s=bpy.context.scene;s.frame_set(1);rig=bpy.data.objects[args.rig_name]
    accepted_hands=None;accepted_tracks=None
    previous_tracks_path=Path(rig.get('mocap_v2_hand_tracks',''))
    if (args.preserve_hand_presentation and previous_tracks_path.is_file() and rig.animation_data and rig.animation_data.action and s.camera
            and (s.render.resolution_x,s.render.resolution_y)==(width,height)):
        from hand_refinement import load_tracks
        from hand_preservation import capture
        previous_tracks=load_tracks(previous_tracks_path)
        if previous_tracks and all(len(track['uv'])==T for track in previous_tracks.values()):
            accepted_tracks=previous_tracks;accepted_hands=capture(s,rig,T)
    # A snapshot may contain an older take of a different duration. All later
    # hand/assistance passes must use this take, including its final frames.
    rig['mocap_v2_frame_count']=T
    navigation_offset=s.camera.matrix_basis.copy() if s.camera and s.camera.get('mocap_v2_navigation_view') else Matrix.Identity(4)
    base_world=rig.matrix_world.copy();base_location=rig.location.copy()
    rig.animation_data_clear();rig.animation_data_create().action=bpy.data.actions.new('V2 Calibrated Absolute FK')
    active=set(MAP.values())
    if face is not None:active.update(['Eyes','Eye_L','Eye_R'])
    for name in list(active):
        b=rig.pose.bones.get(name)
        if b is None:raise ValueError(f'Missing mapped bone {name}')
        while b.parent:active.add(b.parent.name);b=b.parent
    for name in active:
        b=rig.pose.bones[name];b.matrix_basis=Matrix.Identity(4)
        for con in b.constraints:con.mute=True
    # Remove the old take's curves; optional observed face controls replace them.
    for obj in bpy.data.objects:
        if obj.type=='MESH' and obj.data.shape_keys and any(m.type=='ARMATURE' and m.object==rig for m in obj.modifiers):
            keys=obj.data.shape_keys
            if keys.animation_data and not args.preserve_expressions:keys.animation_data_clear()
    bpy.context.view_layer.update()
    restR={b.name:np.asarray(b.bone.matrix_local.to_3x3()) for b in rig.pose.bones}
    invmap={name:j for j,name in MAP.items()};corrections={}
    for j,name in MAP.items():
        b=rig.pose.bones[name]
        if j in (0,15):correction=np.eye(3)
        else:
            ds=rest[OUT[j]]-rest[j] if j in OUT else rest[j]-rest[int(parents[j])]
            d=b.bone.tail_local-b.bone.head_local
            correction=np.asarray(d.normalized().rotation_difference(Vector((C@ds).tolist()).normalized()).to_matrix())
        corrections[j]=correction
    # Estimate source-to-camera orientation from independent SAM 3D joints.
    si=[1,2,16,17,18,19,20,21];oi=[9,10,5,6,7,8,62,41]
    Rsc=rigid_rotation(J[0,si],obs['keypoints_3d'][0,oi])
    cv_rotation=Rsc@C.T@np.asarray(base_world.to_3x3()).T
    cd=bpy.data.cameras.new('V2 Video Intrinsics');cam=bpy.data.objects.new('V2 Video Camera',cd)
    s.collection.objects.link(cam);s.camera=cam
    framing_calibration=None
    framing_camera=None
    if args.framing_calibration:
        framing_calibration=json.loads(Path(args.framing_calibration).read_text(encoding='utf8'))
        child=bpy.data.objects.get(framing_calibration.get('camera_name','Camera · 视图调整'))
        if child is None or child.type!='CAMERA':
            child_data=bpy.data.cameras.new(framing_calibration.get('camera_name','V2 Framed Camera'))
            child=bpy.data.objects.new(framing_calibration.get('camera_name','V2 Framed Camera'),child_data)
            s.collection.objects.link(child)
        # Reuse the source blend's framing camera as the saved active camera.
        # Clear its old SAM/static parent first; otherwise Blender can reopen
        # the file on the legacy camera even though the render used the V2 cam.
        child.parent=None
        child.matrix_parent_inverse=Matrix.Identity(4)
        child.parent=cam
        child.matrix_local=Matrix(framing_calibration['matrix_local'])
        child.data.lens=float(framing_calibration.get('lens',cd.lens))
        child.data.shift_x=float(framing_calibration.get('shift_x',0.))
        child.data.shift_y=float(framing_calibration.get('shift_y',0.))
        child.name='相机V2 · 活动画面（近景）'
        framing_camera=child
        s.camera=child
        for screen in bpy.data.screens:
            for area in screen.areas:
                if area.type=='VIEW_3D':
                    area.spaces.active.region_3d.view_perspective='CAMERA'
    cam.rotation_mode='QUATERNION';cd.sensor_fit='HORIZONTAL';cd.sensor_width=36
    s.render.resolution_x=width;s.render.resolution_y=height;s.render.resolution_percentage=100
    s.render.pixel_aspect_x=1;s.render.pixel_aspect_y=float(K[0,0,0]/K[0,1,1])
    s.render.fps=int(motion['mocap_framerate']);s.render.fps_base=1;s.frame_start=1;s.frame_end=T
    source_rt=raw['RT_vggt'] if 'RT_vggt' in raw.files else raw['RT']
    c2w=np.linalg.inv(source_rt);relative=np.linalg.inv(c2w[0])[None]@c2w
    hand_R=np.array([cv_rotation.T@r[:3,:3] for r in relative])
    hand_rig_R=np.einsum('ij,tjk->tik',np.asarray(base_world.to_3x3()).T,hand_R)
    parts=None
    if args.hand_parts:
        with np.load(args.hand_parts) as archive:
            parts=dict(archive)
    hand_data,hand_stats,hand_blends,wrist_goals,hand_tracks=hand_targets(
        args.hands,rig,T,hand_rig_R,K,width,height,obs['keypoints_2d'],face,parts)
    anchors=['eyeLf03Joint','eyeRt03Joint','faceMdNoseJoint','Arm_L','Arm_R']
    # This close crop has shoulders partly outside the frame. Their inferred
    # width must not outweigh visible face anchors and force an extreme zoom.
    anchor_obs=[1,2,0,5,6];weights=np.array([1.,1.,.25,0.,0.])
    points=[];heads=[];chests=[];direction_errors=[];prev={};hand_prev={}
    ordered=sorted(rig.pose.bones,key=lambda b:len(b.parent_recursive))
    for t in range(T):
        if t%60==0:print(f'V2_PROGRESS 骨架重定向 {t+1}/{T}',flush=True)
        actualR={};desired={}
        for b in ordered:
            B=restR[b.name];parent=b.parent
            inherited=(actualR[parent.name]@restR[parent.name].T@B) if parent else B
            if b.name in invmap:
                j=invmap[b.name];D=C@G[t,j]@C.T@corrections[j]@B
                if b.name.startswith(('Wrist_','Thumb','IndexFinger','MiddleFinger','RingFinger','LittleFinger')):
                    # An unobserved hand has no reliable global wrist/finger
                    # twist. Use the parent-relative rest pose as the fallback;
                    # the track's exit tail retains the last visible gesture.
                    D=inherited.copy()
                if b.name.startswith(('Arm_','Elbow_')):
                    side=b.name[-1]
                    evidence=hand_blends.get((t,'Wrist_'+side),0.)
                    elbow_index=7 if side=='L' else 8
                    elbow_uv=obs['keypoints_2d'][t,elbow_index]/[width,height]
                    margin=float(np.min(np.r_[elbow_uv,1-elbow_uv]))
                    if np.isfinite(margin):
                        evidence=max(evidence,float(np.clip(margin/.10,0,1)))
                    # The close crop cannot constrain an unseen arm. A guessed
                    # bent arm must not keep an invisible hand on the abdomen.
                    torso=actualR['UpperBody2']@restR['UpperBody2'].T
                    # About 10 degrees of lateral clearance in the relaxed
                    # fallback (previously 4.6); observed gestures still win.
                    down=torso@np.array([.18 if side=='L' else -.18,
                                        -.04 if b.name.startswith('Arm_') else -.32,-1.])
                    natural=np.asarray(Vector(B[:,1].tolist()).rotation_difference(Vector(down.tolist())).to_matrix())@B
                    qa=Matrix(natural.tolist()).to_quaternion();qb=Matrix(D.tolist()).to_quaternion()
                    if qa.dot(qb)<0:qb.negate()
                    D=np.asarray(qa.slerp(qb,evidence).to_matrix())
                if j==12:
                    # Keep the neck on the torso chain. Head rotation is
                    # solved from the face track below; adding SMPL-H neck
                    # pitch here was what pushed Ellen's neck forward.
                    D=inherited.copy()
                if j==15 and args.head_mode=='face':
                    observed_head=head_target(face,t,hand_R[t],np.asarray(base_world.to_3x3()),B,C.T)
                    if observed_head is not None:
                        source_q=Matrix(D.tolist()).to_quaternion()
                        target_q=Matrix(observed_head.tolist()).to_quaternion()
                        if source_q.dot(target_q)<0:target_q.negate()
                        D=np.asarray(source_q.slerp(target_q,float(face['weights'][t])).to_matrix())
                if (t,b.name) in hand_data:
                    fallback=Matrix(D.tolist()).to_quaternion().normalized()
                    hand=hand_data[t,b.name]
                    if hand.shape==(3,3):D=hand
                    elif hand.shape==(4,):D=inherited@aa(hand[:3]*hand[3])
                    else:
                        # Child phalanges inherit the solved palm/parent roll.
                        # Rotating an unrelated SMPL global frame onto every
                        # observed segment independently leaves arbitrary twist
                        # and can make a distal joint flip by almost 180 degrees.
                        current=Vector(inherited[:,1].tolist()).normalized()
                        desired_dir=Vector(hand.tolist()).normalized()
                        swing=current.rotation_difference(desired_dir)
                        if swing.angle>np.radians(115.):
                            swing=Quaternion((1,0,0,0)).slerp(swing,float(np.radians(115.)/swing.angle))
                        D=np.asarray(swing.to_matrix())@inherited
                    candidate=Matrix(D.tolist()).to_quaternion().normalized()
                    if fallback.dot(candidate)<0:candidate.negate()
                    D=np.asarray(fallback.slerp(candidate,hand_blends[t,b.name]).normalized().to_matrix())
                q=Matrix((inherited.T@D).tolist()).to_quaternion().normalized()
                key_quat(b,q,t+1,prev)
                actualR[b.name]=inherited@np.asarray(q.to_matrix());desired[b.name]=D
            elif b.name in active:
                key_quat(b,Quaternion((1,0,0,0)),t+1,prev);actualR[b.name]=inherited
            else:actualR[b.name]=inherited@np.asarray(b.matrix_basis.to_3x3())
        bpy.context.view_layer.update()
        for name,D in desired.items():
            observed=np.asarray(rig.pose.bones[name].matrix.to_3x3().normalized())
            direction_errors.append(float(np.degrees(np.arccos(np.clip((np.trace(observed.T@D)-1)/2,-1,1)))))
        points.append([list(base_world@rig.pose.bones[n].head) for n in anchors])
        heads.append([list(base_world@rig.pose.bones[n].head) for n in MAP.values()])
        chests.append(list(base_world@rig.pose.bones['UpperBody2'].head))
    if max(direction_errors)>.25:raise RuntimeError(f'Retarget orientation check failed: {max(direction_errors):.3f} degrees')
    points=np.asarray(points);origin=np.asarray(base_world.translation)
    trans0=fit_translation((points[0]-origin)@cv_rotation.T,composition_uv[0,anchor_obs],K[0],weights)
    framing_depth_ratio=1.
    if args.framing=='chest_crop':
        eye=points[0,:2].mean(0);eye_uv=obs['keypoints_2d'][0,[1,2]].mean(0)
        crop_points=(np.array([eye,chests[0]])-origin)@cv_rotation.T
        crop_uv=np.array([eye_uv,[eye_uv[0],height*1.02]])
        crop_translation=fit_translation(crop_points,crop_uv,K[0],np.array([1.,1.]))
        old_eye_depth=crop_points[0,2]+trans0[2]
        framing_depth_ratio=float((crop_points[0,2]+crop_translation[2])/old_eye_depth)
        trans0=crop_translation
    cam0=np.eye(4);cam0[:3,:3]=cv_rotation.T@F;cam0[:3,3]=origin-cv_rotation.T@trans0
    F4=np.diag([1.,-1.,-1.,1.]);translation_scale=1.0
    errors=[];locations=[];camera_matrices=[];presentation_matrices=[];qprev=None
    framing_local=np.asarray(Matrix(framing_calibration['matrix_local'])) if framing_calibration else np.eye(4)
    parent_matrix=(rig.parent.matrix_world@rig.matrix_parent_inverse) if rig.parent else Matrix.Identity(4)
    world_to_location=parent_matrix.to_3x3().inverted()
    for t in range(T):
        rel=relative[t].copy();rel[:3,3]*=translation_scale;mat=cam0@F4@rel@F4;camera_matrices.append(mat)
        # Solve the actor against the same close-up camera that is shown to
        # the user. The old bridge solved in the hidden raw-camera frame and
        # then applied a local framing offset afterwards, pushing hands out
        # of the visible image.
        solve_mat=mat@framing_local
        solve_mat[:3,:3]=upright_rotation(solve_mat[:3,:3])
        presentation_matrices.append(solve_mat)
        cam.location=mat[:3,3];q=Matrix(mat[:3,:3].tolist()).to_quaternion()
        if qprev and q.dot(qprev)<0:q.negate()
        qprev=q.copy();cam.rotation_quaternion=q
        for path in ['location','rotation_quaternion']:cam.keyframe_insert(data_path=path,frame=t+1)
        cd.lens=float(K[t,0,0])*36/width;cd.shift_x=float((width/2-K[t,0,2])/width)
        aspect=float(K[t,0,0]/K[t,1,1]);cd.shift_y=float((K[t,1,2]-height/2)*aspect/width)
        s.render.pixel_aspect_y=aspect
        for path in ['lens','shift_x','shift_y']:cd.keyframe_insert(data_path=path,frame=t+1)
        Rc=F@solve_mat[:3,:3].T;pc=(points[t]-origin)@Rc.T
        tr=fit_translation(pc,composition_uv[t,anchor_obs],K[t],weights)
        if args.framing=='chest_crop':
            eye_relative=pc[:2].mean(0)
            eye_uv=obs['keypoints_2d'][t,[1,2]].mean(0)
            eye_depth=(eye_relative[2]+tr[2])*framing_depth_ratio
            eye_ray=np.array([(eye_uv[0]-K[t,0,2])/K[t,0,0],(eye_uv[1]-K[t,1,2])/K[t,1,1],1.])
            tr=eye_ray*eye_depth-eye_relative
        delta=solve_mat[:3,3]+Rc.T@tr-origin
        rig.location=base_location+world_to_location@Vector(delta.tolist());rig.keyframe_insert(data_path='location',frame=t+1)
        locations.append(delta);p3=pc+tr
        uv=p3[:,:2]/p3[:,2:3]*np.array([K[t,0,0],K[t,1,1]])+K[t,:2,2]
        errors.append(np.linalg.norm(uv-composition_uv[t,anchor_obs],axis=1))
    # Keep camera compensation intact. Joint background-relative stabilization
    # runs once against the final camera; independent root filtering here would
    # alter compensation and reintroduce image motion.
    locations=np.asarray(locations)
    errors=[]
    for t in range(T):
        rig.location=base_location+world_to_location@Vector(locations[t].tolist())
        rig.keyframe_insert(data_path='location',frame=t+1)
        mat=presentation_matrices[t];Rc=F@mat[:3,:3].T
        pc=(points[t]+locations[t]-mat[:3,3])@Rc.T
        uv=pc[:,:2]/pc[:,2:3]*np.array([K[t,0,0],K[t,1,1]])+K[t,:2,2]
        errors.append(np.linalg.norm(uv-composition_uv[t,anchor_obs],axis=1))
    if args.framing=='portrait':
        # Resize the displayed shot after actor translation has been solved.
        # Moving both camera and rig together preserves scale, which was why
        # the previous implementation still rendered an oversized face.
        for t in range(T):
            mat=presentation_matrices[t];Rc=F@mat[:3,:3].T
            pc=(points[t]+locations[t]-mat[:3,3])@Rc.T
            uv=pc[:,:2]/pc[:,2:3]*np.array([K[t,0,0],K[t,1,1]])+K[t,:2,2]
            current_span=float(np.linalg.norm(uv[0]-uv[1]))
            target_span=float(np.linalg.norm(composition_uv[t,0]-composition_uv[t,1]))
            if current_span>target_span*1.05:
                ratio=min(3.,current_span/max(target_span,1.))
                depth=max(.05,float(pc[:2,2].mean()))
                mat=mat.copy();mat[:3,3]+=mat[:3,2]*(depth*(ratio-1.))
                presentation_matrices[t]=mat
                # Moving the presentation camera changes the actor's
                # projected position too. Refit the root translation against
                # the same camera immediately; previously only the camera was
                # updated, leaving the character at the old solve location.
                Rc=F@mat[:3,:3].T
                pc=(points[t]-origin)@Rc.T
                tr=fit_translation(pc,composition_uv[t,anchor_obs],K[t],weights)
                locations[t]=mat[:3,3]+Rc.T@tr-origin
                rig.location=base_location+world_to_location@Vector(locations[t].tolist())
                rig.keyframe_insert(data_path='location',frame=t+1)
    # Hand image fitting runs once, after body/head and camera are final.
    # Keep source elbow flexion as an independent prior; earlier image-space
    # corrections must not become the reference for later corrections.
    for side,indices in [('L',(16,18,20)),('R',(17,19,21))]:
        if side not in hand_tracks:continue
        a,e,w=indices
        upper=J[:,e]-J[:,a];fore=J[:,w]-J[:,e]
        dot=np.sum(upper*fore,axis=1)/np.maximum(np.linalg.norm(upper,axis=1)*np.linalg.norm(fore,axis=1),1e-9)
        hand_tracks[side]['arm_flex_reference']=np.arccos(np.clip(dot,-1,1))
    hand_continuity={'method':'single bounded final-camera hand solve'}
    reach_errors=[];hand_projection_before=[];hand_projection_after=[];arm_shape_errors=[]
    # Bake the final unrolled presentation camera before face solving. The
    # eye-line/roll correction must see the same camera the user will render,
    # otherwise a legacy child-camera tilt is baked into the head pose.
    if framing_camera is not None:
        local_framing=Matrix(framing_calibration['matrix_local'])
        framing_camera.parent=None
        framing_camera.matrix_parent_inverse=Matrix.Identity(4)
        framing_camera.animation_data_clear()
        framing_camera.rotation_mode='QUATERNION'
        framing_camera.data.lens=float(framing_calibration.get('lens',framing_camera.data.lens))
        framing_camera.data.shift_x=float(framing_calibration.get('shift_x',0.))
        framing_camera.data.shift_y=float(framing_calibration.get('shift_y',0.))
        framing_camera['character_director_camera']=True
        for frame in range(1,T+1):
            s.frame_set(frame);bpy.context.view_layer.update()
            framing_camera.matrix_world=Matrix(presentation_matrices[frame-1].tolist())
            framing_camera.keyframe_insert(data_path='location',frame=frame)
            framing_camera.keyframe_insert(data_path='rotation_quaternion',frame=frame)
        s.camera=framing_camera
    print('V2_PROGRESS 头眼视线与连续性处理',flush=True)
    face_report=apply_face(s,rig,face,key_quat,K,(height,width),args.gaze_reference_frame,args.gaze_mode,args.head_mode,args.preserve_expressions)
    shape_owners=[o.data.shape_keys for o in s.objects if o.type=='MESH' and o.data.shape_keys
                  and any(m.type=='ARMATURE' and m.object==rig for m in o.modifiers)]
    if args.preserve_expressions:shape_owners=[]
    for obj in [rig,cam,cd,s]+shape_owners:
        if not obj.animation_data or not obj.animation_data.action:continue
        for layer in obj.animation_data.action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    for fc in bag.fcurves:
                        for k in fc.keyframe_points:k.interpolation='LINEAR'
    # The source blend contains a legacy framing-child hierarchy whose parent
    # can be restored by its add-on handlers on reload. Bake the calibrated
    # local framing onto the actual V2 camera transform so the saved scene has
    # one stable active camera and remains correct without that hierarchy.
    if framing_camera is not None:
        local_framing=Matrix(framing_calibration['matrix_local'])
        framing_camera.parent=None
        framing_camera.matrix_parent_inverse=Matrix.Identity(4)
        framing_camera.animation_data_clear()
        framing_camera.rotation_mode='QUATERNION'
        framing_camera.data.lens=float(framing_calibration.get('lens',framing_camera.data.lens))
        framing_camera.data.shift_x=float(framing_calibration.get('shift_x',0.))
        framing_camera.data.shift_y=float(framing_calibration.get('shift_y',0.))
        # Tell the legacy camera-navigation add-on that this is the complete
        # recorded shot camera. It must not wrap or reparent it on reload.
        framing_camera['character_director_camera']=True
        for frame in range(1,T+1):
            s.frame_set(frame);bpy.context.view_layer.update()
            framing_camera.matrix_world=Matrix(presentation_matrices[frame-1].tolist())
            framing_camera.keyframe_insert(data_path='location',frame=frame)
            framing_camera.keyframe_insert(data_path='rotation_quaternion',frame=frame)
        s.camera=framing_camera
    stationary_denoising,world_shifts=denoise_final_motion(s,rig,T)
    locations+=world_shifts
    # Apply the saved framing offset before fitting hands, so fitting and
    # rendering use exactly the same editable presentation camera.
    view=ensure_presentation_camera(s)
    view.matrix_basis=navigation_offset
    stationary_denoising['final_root_fit']=refit_final_actor(s,rig,composition_uv,view.parent)
    stationary_denoising['placement_observations']=placement_report
    from actor_stability import stabilize_actor_projection
    background_path=str(raw['background_motion']) if 'background_motion' in raw.files else None
    stationary_denoising['actor_projection']=stabilize_actor_projection(s,rig,T,background_path,
        image_scale=image_scale,image_offset=image_offset,
        eye_target=composition_uv[:,[1,2]].mean(axis=1) if face is not None and placement_report.get('dense_face_used') else None,
        eye_valid=face['valid'] if face is not None else None)
    s['mocap_v2_camera_track']=str(Path(args.camera_track).resolve())
    if background_path:s['mocap_v2_background_motion']=background_path
    from hand_refinement import apply as refine_hands, evaluate as evaluate_hands, save_tracks
    from hand_preservation import observations_match,restore
    if accepted_hands is not None and observations_match(accepted_tracks,hand_tracks):
        # Same observed take: retain accepted hand articulation/size during a
        # camera/body update. Different video or changed detections refit fresh.
        for side,track in hand_tracks.items():
            for key in ('arm_flex_reference','arm_pole_reference','hand_depth_reference','hand_span_reference'):
                if key in accepted_tracks[side]:track[key]=accepted_tracks[side][key].copy()
        hand_fit=restore(s,rig,accepted_hands,hand_tracks)
    else:
        hand_fit=refine_hands(s,rig,hand_tracks,body=obs,face=face)
    if args.hamer_fingers:
        from hamer_pose import apply_hamer_pose
        with np.load(args.hamer_fingers) as fingers:
            hand_fit['hamer_frames']=apply_hamer_pose(s,rig,fingers['points'],fingers['valid'],hand_tracks)
    from part_projection import align_parts
    align_parts(s,rig,hand_tracks)
    from temporal_denoise import stabilize_final_hands
    stabilize_final_hands(s,rig,hand_tracks)
    final_hand_metrics=evaluate_hands(s,rig,hand_tracks)
    from shoulder_frills import stabilize
    stabilize(rig)
    hand_track_path=output.with_suffix('.hand_tracks.npz')
    save_tracks(hand_track_path,hand_tracks)
    rig['mocap_v2_hand_tracks']=str(hand_track_path)
    for t in range(T):
        s.frame_set(t+1);bpy.context.view_layer.update()
        locations[t]=np.asarray(rig.matrix_world.translation)-origin
        heads[t]=[list(rig.matrix_world@rig.pose.bones[n].head-Vector(locations[t].tolist())) for n in MAP.values()]
    report={'frames':T,'orientation_max_deg':max(direction_errors),'anchor_names':anchors,
       'framing':args.framing,'framing_depth_ratio':framing_depth_ratio,
       'head_mode':args.head_mode,'framing_calibration':str(Path(args.framing_calibration).resolve()) if args.framing_calibration else None,
       'composition_image_scale':image_scale,'composition_image_offset':image_offset.tolist(),
       'anchor_error_reference':'composition-adjusted observations' if args.framing=='portrait' else 'source observations',
       'source_intrinsics_anisotropy_max':intrinsics_anisotropy,'hand_observations':hand_stats,
       'final_hand_continuity':hand_continuity,
       'stationary_denoising':stationary_denoising,
       'projection_metrics_stage':'final camera metrics in final_hand_projection',
       'final_hand_projection':final_hand_metrics,'hand_joint_fit':hand_fit,
       'arm_shape_priority':'source 3D posture before pixel placement',
       'upper_arm_change_max_deg':float(np.max(arm_shape_errors,axis=0)[0]) if arm_shape_errors else 0.,
       'elbow_flexion_change_max_deg':float(np.max(arm_shape_errors,axis=0)[1]) if arm_shape_errors else 0.,
       'hand_ik_reach_error_median_m':float(np.median(reach_errors)) if reach_errors else None,
       'hand_ik_reach_error_max_m':max(reach_errors) if reach_errors else None,
       'visible_hand_projection_before_median_px':float(np.median(hand_projection_before)) if hand_projection_before else None,
       'visible_hand_projection_after_median_px':float(np.median(hand_projection_after)) if hand_projection_after else None,
       'anchor_median_px':np.median(errors,axis=0).tolist(),'camera_translation_scale':translation_scale,
       'camera_translation_scale_status':'approximate_unverified','source':'exported poses only; unrotated auxiliary joints excluded',
       'face_animation':face_report,'limitations':['SAM observations are estimates, not ground truth',
       'Different proportions prevent all face/shoulder anchors matching simultaneously']}
    s['mocap_v2_validation']=json.dumps(report);s['mocap_v2_motion']=str(Path(args.motion).resolve())
    output.with_suffix('.validation.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    rotation_names=list(MAP.values())+(['Eyes','Eye_L','Eye_R'] if face is not None else [])
    rotation_samples=[]
    for t in range(T):
        s.frame_set(t+1)
        rotation_samples.append([list(rig.pose.bones[n].rotation_quaternion.normalized()) for n in rotation_names])
    np.savez_compressed(output.with_suffix('.diagnostics.npz'),source_joints=J,source_global_rotations=G,
       target_heads=heads,actor_displacement=locations,camera_to_world=camera_matrices,K=K,
       anchor_error_px=errors,bone_names=np.array(list(MAP.values())),R_source_camera=Rsc,
       rotation_bone_names=np.array(rotation_names),local_quaternions=rotation_samples,
       hand_visibility=np.column_stack([hand_tracks[side]['valid'] if side in hand_tracks else np.zeros(T,bool) for side in ['L','R']]))
    s.frame_set(1);s.render.image_settings.media_type='IMAGE';s.render.image_settings.file_format='PNG';s.render.engine='BLENDER_EEVEE'
    s.render.film_transparent=False;s.render.use_compositing=False;s.render.filepath=str(output.with_suffix('.png'))
    s.camera['character_director_camera']=True
    if framing_camera is None:s.camera.name='相机V2 · 活动画面'
    for marker in s.timeline_markers:
        if marker.camera is not None:marker.camera=s.camera
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=='VIEW_3D':
                space=area.spaces.active
                space.use_local_camera=False;space.camera=s.camera
                space.region_3d.view_perspective='CAMERA'
                space.region_3d.view_camera_zoom=0.
    ensure_presentation_camera(s)
    rig['mocap_v2_frame_count']=T
    rig['mocap_v2_diagnostics']=str(output.with_suffix('.diagnostics.npz'))
    s['mocap_v2_stationary_denoise_version']=1
    print('V2_PROGRESS 保存结果与校验数据',flush=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output));print('VALIDATED',json.dumps(report),flush=True)
    if args.render:
        for frame in map(int,args.frames.split(',')):
            s.frame_set(frame);s.render.filepath=str(output.with_suffix(f'.frame{frame:04d}.png'));bpy.ops.render.render(write_still=True)
if __name__=='__main__':main()

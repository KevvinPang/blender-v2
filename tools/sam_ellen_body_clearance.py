"""Final arm safety against a moving, skinned torso envelope.

Image-space crop constraints are proposals. They cannot justify an unsupported
elbow fold or a link through the torso. This pass runs after entry/direction
fitting, preserves link lengths, and changes only the five arm controls.
"""
import math
import numpy as np
from mathutils import Vector, Quaternion

PREFIXES=('Arm_','Elbow_','Wrist_','ArmTwist_','HandTwist_')
# Only the socket itself blends into the torso. The former .55 exemption
# concealed a cross-chest upper arm even when the elbow was outside the body.
SOCKET_FRACTION=.12


def torso_geometry(rig,include_bust=False):
    """Build overlapping convex slabs from torso-weighted mesh, in rig space."""
    import bpy
    names={'LowerBody','UpperBody','UpperBody1','UpperBody2'}
    if include_bust:names.update(b.name for b in rig.pose.bones if b.name.startswith(('Breast','Bust')))
    chest=rig.data.bones.get('UpperBody2');base=rig.data.bones.get('UpperBody')
    if chest is None or base is None:return []
    lo=base.head_local.z;hi=chest.tail_local.z
    blocks=[]
    for mesh in bpy.data.objects:
        if mesh.type!='MESH':continue
        owners=[m.object for m in mesh.modifiers if m.type=='ARMATURE' and m.object and m.object.data==rig.data]
        if not owners:continue
        groups={g.index:g.name for g in mesh.vertex_groups};transform=owners[0].matrix_world.inverted()@mesh.matrix_world
        vertices=[];weights=[]
        for v in mesh.data.vertices:
            if sum(g.weight for g in v.groups if groups.get(g.group) in names)<.8:continue
            point=transform@v.co
            if not lo-.03<point.z<hi-.025:continue
            vertices.append((*point,1.));weights.append({groups[g.group]:g.weight for g in v.groups if groups.get(g.group) in rig.pose.bones})
        if not vertices:continue
        vertices=np.asarray(vertices);z=vertices[:,2]
        # Overlap avoids artificial gaps at slab boundaries. Frills, arms and
        # hair are excluded by their skinning weights rather than visibility.
        for bottom,top in ((lo-.03,lo+(hi-lo)*.43),(lo+(hi-lo)*.30,lo+(hi-lo)*.76),(lo+(hi-lo)*.62,hi-.025)):
            ids=np.flatnonzero((z>=bottom)&(z<=top))
            if len(ids)<12:continue
            # Keep all extremal vertices; no arbitrary mesh subsampling.
            blocks.append((vertices[ids],{name:np.array([weights[i].get(name,0.) for i in ids]) for name in set().union(*(weights[i] for i in ids))}))
    return blocks


_normals=np.array([(x,y,z) for x in (-1,0,1) for y in (-1,0,1) for z in (-1,0,1) if (x,y,z)!=(0,0,0)],float)
_normals/=np.linalg.norm(_normals,axis=1)[:,None]


def torso_envelope(rig,geometry):
    """Supporting half-spaces contain the actual weighted torso vertices."""
    result=[]
    for rest,weights in geometry:
        points=np.zeros((len(rest),3));total=np.zeros(len(rest))
        for name,weight in weights.items():
            bone=rig.pose.bones[name];matrix=np.asarray(bone.matrix@bone.bone.matrix_local.inverted())
            points+=(rest@matrix.T)[:,:3]*weight[:,None];total+=weight
        points/=np.maximum(1e-8,total[:,None])
        result.append((points@_normals.T).max(0))
    return np.asarray(result)


def _link_margin(start,end,envelope,radius,begin=0.):
    """Minimum signed half-space clearance along a complete link.

    max(plane distances) is convex along a segment. Ternary minimization
    locates its minimum, including cases with both endpoints outside the body.
    Negative values denote penetration into the radius-expanded torso proxy.
    """
    if not len(envelope):return .1
    start=np.asarray(start);delta=np.asarray(end)-start
    p=_normals@start;v=_normals@delta
    low=np.full(len(envelope),begin);high=np.ones(len(envelope))
    for _ in range(22):
        left=(2*low+high)/3;right=(low+2*high)/3
        a=np.max(p+left[:,None]*v-envelope-radius,axis=1)
        b=np.max(p+right[:,None]*v-envelope-radius,axis=1)
        choose=a<b;high=np.where(choose,right,high);low=np.where(choose,low,left)
    t=(low+high)*.5
    return float(np.min(np.max(p+t[:,None]*v-envelope-radius,axis=1)))


def arm_clearance(a,e,w,envelope):
    upper=(Vector(e)-Vector(a)).length;lower=(Vector(w)-Vector(e)).length
    # The upper sleeve/shoulder socket blends into the torso. Exempt its
    # proximal portion; protect the elbow and whole forearm with finite radii.
    return min(_link_margin(a,e,envelope,upper*.065,SOCKET_FRACTION),
               _link_margin(e,w,envelope,lower*.085))


def _candidate_clearance(a,elbows,wrists,envelope,upper,lower):
    """Vectorized continuous-segment search for all candidate elbow branches."""
    elbows=np.asarray(elbows);wrists=np.asarray(wrists)
    if not len(envelope):return np.full(len(elbows),.1)
    def margins(starts,ends,radius,begin):
        p=starts@_normals.T;v=(ends-starts)@_normals.T
        low=np.full((len(starts),len(envelope)),begin);high=np.ones_like(low)
        for _ in range(22):
            left=(2*low+high)/3;right=(low+2*high)/3
            lv=(p[:,None,:]+left[:,:,None]*v[:,None,:]-envelope[None,:,:]-radius).max(2)
            rv=(p[:,None,:]+right[:,:,None]*v[:,None,:]-envelope[None,:,:]-radius).max(2)
            mask=lv<rv;high=np.where(mask,right,high);low=np.where(mask,low,left)
        return (p[:,None,:]+((low+high)*.5)[:,:,None]*v[:,None,:]-envelope[None,:,:]-radius).max(2).min(1)
    return np.minimum(margins(np.repeat(np.asarray(a)[None,:],len(elbows),axis=0),elbows,upper*.065,SOCKET_FRACTION),
                      margins(elbows,wrists,lower*.085,0.))


def chains_safe(a,elbows,wrists,envelope,upper,lower,margin=0.):
    """Exact segment/convex-slab intersection, batched for temporal edges.

    Intersect the intervals satisfying every supporting half-space. Checking
    endpoints alone misses a link that enters and leaves the chest.
    """
    elbows=np.asarray(elbows,float);wrists=np.asarray(wrists,float)
    if not len(envelope):return np.ones(len(elbows),bool)
    def clear(starts,ends,radius,begin):
        p=np.asarray(starts)@_normals.T
        v=(ends-starts)@_normals.T
        delta=envelope[None,:,:]+radius+margin-p[:,None,:]
        slope=v[:,None,:]
        limits=np.divide(delta,slope,out=np.zeros_like(delta),where=abs(slope)>1e-12)
        lo=np.maximum(begin,np.where(slope< -1e-12,limits,-np.inf).max(2))
        hi=np.minimum(1.,np.where(slope>1e-12,limits,np.inf).min(2))
        outside=((abs(slope)<=1e-12)&(delta<0)).any(2)
        return (~((lo<=hi)&~outside)).all(1)
    starts=np.broadcast_to(np.asarray(a),elbows.shape)
    return clear(starts,elbows,upper*.065,SOCKET_FRACTION)&clear(elbows,wrists,lower*.085,0.)


def support_envelope(data,side):
    from sam_ellen_arm_visibility import observed_frames
    times=np.array([r['time'] for r in data['frames']]);visible=observed_frames(data,side)
    support=visible.astype(float)
    for episode in data.get('screen',{}).get('hand_entry',{}).get('sides',{}).get(side,[]):
        for key,observation in episode.get('observations',{}).items():
            i=int(key)
            if 0<=i<len(times):support[i]=max(support[i],.65)
        for key in episode.get('regions',{}):
            i=int(key)
            if 0<=i<len(times):support[i]=max(support[i],.4)
    # Nearby evidence permits preparation/release, with no hard state switch.
    strength=support.copy()
    for i in np.flatnonzero(support):
        ramp=np.clip(1-np.abs(times-times[i])/.32,0,1);ramp=ramp*ramp*(3-2*ramp)
        strength=np.maximum(strength,support[i]*ramp)
    return visible,strength


def observed_elbows(data,side):
    """Hand detections and skin-color axes do not measure an elbow position."""
    n=len(data['frames']);rows=data.get('screen',{}).get('pose_arms',[])
    if len(rows)!=n:return np.zeros(n,bool)
    result=[]
    for row in rows:
        pose=row.get(side) or {};inside=pose.get('in_frame',[])
        result.append(bool(pose.get('source')=='independent-video-pose' and
                           pose.get('confidence',0)>=.5 and pose.get('elbow_observed') and
                           len(inside)==3 and inside[1]))
    return np.asarray(result,bool)


def guard_unsupported_elbows(scratch,rig,channels,data,side,allowed,protected=None,report=None):
    """Preserve measured hands without inventing an elevated, hidden elbow.

    A camera wrist ray admits many elbow branches. If no elbow is measured,
    prefer the lowered torso-relative capture prior on that same wrist circle.
    Only episodes containing an unsupported raise are repaired. Preparation
    and release join the same branch; visible/protected elbows remain anchors.
    """
    from sam_ellen_arm_policy import front_circle,transport_bend,wrist_limit
    from sam_ellen_arm_chain import torso_front_plane,anatomical_rotations,bounded_wrist_rotation
    from sam_ellen_arm_visibility import observed_frames
    from sam_ellen_retarget import set_global_rotation
    times=np.array([r['time'] for r in data['frames']]);n=len(times)
    allowed=np.asarray(allowed,bool);protected=np.zeros(n,bool) if protected is None else np.asarray(protected,bool)
    measured=observed_elbows(data,side)|protected;visible=observed_frames(data,side)
    original={name:list(rows) for name,rows in channels.items()};view=scratch.view_layers[0]
    a,e,w=[rig.pose.bones[p+'_'+side] for p in ('Arm','Elbow','Wrist')]
    policy=data.get('_sam_arm_policy',[]);geometry=[];bad=np.zeros(n,bool)
    def load(i):
        for name,rows in original.items():rig.pose.bones[name].rotation_quaternion=rows[i][1]
        view.update()
    for i in range(n):
        load(i);chest=rig.pose.bones.get('UpperBody2') or rig.pose.bones.get('UpperBody1')
        body=chest.matrix.to_quaternion()@chest.bone.matrix_local.to_quaternion().inverted() if chest else Quaternion()
        natural=Vector((.24 if side=='L' else -.24,-.06,-1)).normalized()
        reference=policy[i].get(side,{}).get('reference_upper') if len(policy)==n else None
        # A plausible lowered SAM prior can refine the rest direction, but an
        # unmeasured SAM raise must not become evidence for itself.
        if reference is not None:
            local=body.inverted()@Vector(reference)
            if local.angle(natural)<math.radians(55):natural=natural.lerp(local.normalized(),.35).normalized()
        prior=body@natural;upper=(e.head-a.head).normalized()
        raise_angle=math.degrees((body.inverted()@upper).angle(Vector((0,0,-1))))
        bad[i]=allowed[i] and not measured[i] and raise_angle>65. and upper.angle(prior)>math.radians(40)
        geometry.append((a.head.copy(),e.head.copy(),w.head.copy(),w.matrix.to_quaternion().copy(),prior))
    ids=np.flatnonzero(bad)
    if not len(ids):
        if report is not None:report.append({'side':side,'model':'independent-elbow-prior-v1','trigger_frames':[],'changes':[]})
        return
    # Repair the complete unsupported hand episode, not isolated threshold
    # crossings. This prevents the branch from switching as a score fluctuates.
    active=visible&allowed&~measured
    valid=np.flatnonzero(active)
    for left,right in zip(valid,valid[1:]):
        if times[right]-times[left]<=.24 and (allowed&~measured)[left:right+1].all():active[left:right+1]=True
    chosen=bad.copy();edges=np.diff(np.r_[False,active,False].astype(int))
    for first,end in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        if bad[first:end].any():chosen[first:end]=True
    distance=np.min(abs(times[:,None]-times[np.flatnonzero(chosen)][None,:]),axis=1)
    amount=np.clip(1-distance/.35,0,1);amount=amount*amount*(3-2*amount);amount[~allowed|measured]=0.
    anchors=np.flatnonzero(measured|~allowed)
    if len(anchors):
        distance=np.min(abs(times[:,None]-times[anchors][None,:]),axis=1)
        release=np.clip(distance/.35,0,1);amount*=release*release*(3-2*release)
    previous_axis=None;previous_bend=None;hinge={};last_roll=None;changes=[]
    for i in range(n):
        if amount[i]<=0:
            previous_axis=None;previous_bend=None;hinge={};last_roll=None;continue
        load(i);aa,ee,ww,hand,prior=geometry[i];upper=(ee-aa).length;lower=(ww-ee).length
        axis=(ww-aa).normalized();old_bend=transport_bend(ee-aa,None,axis)
        wanted=transport_bend(prior,None,axis)
        turn=math.atan2(axis.dot(old_bend.cross(wanted)),old_bend.dot(wanted))
        bend=Quaternion(axis,float(amount[i])*turn)@old_bend
        if previous_bend is not None:
            transported=transport_bend(previous_bend,previous_axis,axis)
            delta=math.atan2(axis.dot(transported.cross(bend)),transported.dot(bend))
            limit=math.radians(180*(times[i]-times[i-1]))
            bend=Quaternion(axis,float(np.clip(delta,-limit,limit)))@transported
        # Keep the wrist and both link lengths. The only ambiguity resolved
        # here is the elbow-circle phase, never the observed hand's position.
        floor=max(35.,min(150.,math.degrees((aa-ee).angle(ww-ee))-.01))
        se,sw,bend=front_circle(aa,ww,upper,lower,bend,torso_front_plane(rig),floor,previous_bend)
        hand=bounded_wrist_rotation(hand,sw-se,wrist_limit(math.degrees((aa-se).angle(sw-se))))
        anatomical_rotations(a,e,w,aa,se,sw,hand,set_global_rotation,view,hinge)
        controller=rig.pose.bones.get('HandTwist_'+side)
        if controller is not None:
            palm=w.matrix.to_quaternion().copy();q=controller.rotation_quaternion.copy()
            if q.w<0:q.negate()
            _,roll=q.to_swing_twist('Y')
            roll=float(np.clip(roll,-math.radians(170),math.radians(170)))
            if last_roll is not None:
                step=math.radians(480*(times[i]-times[i-1]));roll=last_roll+float(np.clip(roll-last_roll,-step,step))
            last_roll=roll;controller.rotation_quaternion=Quaternion((0,1,0),roll);view.update();set_global_rotation(w,palm);view.update()
        for prefix in PREFIXES:
            name=prefix+side
            if name in channels:channels[name][i]=channels[name][i][0],tuple(rig.pose.bones[name].rotation_quaternion)
        previous_axis=(w.head-a.head).normalized();previous_bend=transport_bend(e.head-a.head,None,previous_axis)
        changes.append({'frame':i+1,'weight':float(amount[i]),'elbow_measured':False,
                        'wrist_shift':(w.head-ww).length,'elbow_shift':(e.head-ee).length})
        if len(policy)==n and side in policy[i]:policy[i][side]['elbow_evidence']='lowered-prior / hand-only observation'
    if report is not None:report.append({'side':side,'model':'independent-elbow-prior-v1',
        'trigger_frames':(ids+1).tolist(),'changes':changes})


def choose_chain(a,e,w,envelope,plane,minimum,side,previous=None,unsupported=False):
    """Keep a visible wrist when possible; search its continuous elbow circle."""
    from sam_ellen_arm_policy import front_circle,transport_bend
    a,e,w=map(Vector,(a,e,w));upper=(e-a).length;lower=(w-e).length
    axis=(w-a).normalized();bend=transport_bend(e-a,None,axis)
    prior=transport_bend(previous[0]-a,None,axis) if previous is not None else bend
    sign=1 if side=='L' else -1
    lateral=Vector((sign,0,0))
    def select(candidates):
        elbows=np.array([x[0] for x in candidates]);wrists=np.array([x[1] for x in candidates])
        clearance=_candidate_clearance(a,elbows,wrists,envelope,upper,lower)
        inward=np.maximum(0.,-(elbows-np.array(a))@np.array(lateral))*.7 if unsupported else 0.
        unsafe=np.maximum(0.,.002-clearance)+inward
        change=np.sum((elbows-np.array(e))**2,axis=1)+3*np.sum((wrists-np.array(w))**2,axis=1)
        if previous is not None:change+=.30*np.sum((elbows-np.array(previous[0]))**2,axis=1)
        costs=[(float(u)>1e-5,float(u),float(c)) for u,c in zip(unsafe,change)]
        i=min(range(len(costs)),key=costs.__getitem__)
        return costs[i],Vector(elbows[i]),Vector(wrists[i]),float(clearance[i])
    # Search the original wrist first. Only infeasible proposals may move it.
    candidates=[]
    for wrist in (w,):
        for phase in np.linspace(-math.pi,math.pi,49):
            b=Quaternion(axis,float(phase))@bend
            ee,ww,bb=front_circle(a,wrist,upper,lower,b,plane,minimum,prior)
            candidates.append((ee,ww))
    best=select(candidates)
    if best[0][0]:
        # No safe fixed-wrist circle: relax the crop/translation proposal.
        # The fallback arm points down and out in the moving torso frame.
        forward=Vector(plane[1]).normalized();out=(a-Vector(plane[0]));out-=forward*out.dot(forward)
        out.z=0
        out=out.normalized() if out.length>1e-7 else lateral
        down=Vector((0,0,-1));idle_e=a+(down+out*.28+forward*.12).normalized()*upper
        idle_w=idle_e+(down+out*.22+forward*.38).normalized()*lower
        for amount in np.linspace(.08,1.,14):
            target=w.lerp(idle_w,float(amount));ax=(target-a).normalized()
            target_bend=transport_bend(prior,None,ax)
            candidates=[]
            for phase in np.linspace(-math.pi,math.pi,25):
                ee,ww,_=front_circle(a,target,upper,lower,Quaternion(ax,float(phase))@target_bend,plane,minimum,prior)
                candidates.append((ee,ww))
            candidate=select(candidates)
            if candidate[0]<best[0]:best=candidate
            if not best[0][0]:break
    return best[1],best[2],best[3]


def guard_body(scratch,rig,channels,data,side,allowed,protected=None,report=None,repair_idle=False):
    from sam_ellen_arm_chain import anatomical_rotations,torso_front_plane,bounded_wrist_rotation,frame_rotation
    from sam_ellen_retarget import set_global_rotation
    from sam_ellen_arm_policy import wrist_limit
    from sam_ellen_arm_visibility import neutral_pose
    from sam_ellen_temporal import quaternions,trajectory
    view=scratch.view_layers[0];geometry=torso_geometry(rig)
    if not geometry:return
    original={name:list(rows) for name,rows in channels.items()};n=len(data['frames'])
    visible,_=support_envelope(data,side);policy=data.get('_sam_arm_policy',[])
    protected=np.zeros(n,bool) if protected is None else np.asarray(protected,bool)
    a,e,w=[rig.pose.bones[p+'_'+side] for p in ('Arm','Elbow','Wrist')]
    times=np.array([row['time'] for row in data['frames']]);source=[]
    def load(i):
        for name,rows in original.items():rig.pose.bones[name].rotation_quaternion=rows[i][1]
        view.update()
    def body_rotation():
        b=rig.pose.bones.get('UpperBody2') or rig.pose.bones.get('UpperBody1')
        return b.matrix.to_quaternion()@b.bone.matrix_local.to_quaternion().inverted()
    def local_chain():
        body=body_rotation();u=e.head-a.head;v=w.head-e.head
        return body.inverted()@frame_rotation(u,u.cross(v)),u.angle(v),body.inverted()@w.matrix.to_quaternion()
    # Store a torso-relative natural idle and the original supported chains.
    load(0);idle_controls=neutral_pose(rig,side,view)
    for name,q in idle_controls.items():rig.pose.bones[name].rotation_quaternion=q
    view.update();idle=local_chain()
    triggers=[]
    for i in range(n):
        load(i);source.append(local_chain())
        margin=arm_clearance(a.head,e.head,w.head,torso_envelope(rig,geometry))
        # Migration can rebuild an old baked idle whose oscillation is already
        # collision-safe. New bakes only need the geometric trigger.
        triggers.append(bool(allowed[i] and not protected[i] and
                             (margin<-.002 or repair_idle and not visible[i])))
    ids=np.flatnonzero(triggers)
    if not len(ids):
        if report is not None:report.append({'model':'skinned-torso-clearance-v1','side':side,'changes':[], 'trigger_frames':[]})
        return
    distance=np.min(abs(times[:,None]-times[ids][None,:]),axis=1)
    envelope_weight=np.clip(1-distance/.50,0,1);envelope_weight=envelope_weight**2*(3-2*envelope_weight)
    anchors=np.flatnonzero(protected|~np.asarray(allowed,bool))
    if len(anchors):
        distance=np.min(abs(times[:,None]-times[anchors][None,:]),axis=1)
        ramp=np.clip(distance/.20,0,1)
        envelope_weight*=ramp*ramp*(3-2*ramp)
    # Unrelated visible gestures keep their existing channels exactly.
    trusted=np.flatnonzero(visible|protected);hidden=np.zeros(n,bool);proposal=[]
    for i in range(n):
        left=trusted[trusted<i];right=trusted[trusted>i]
        short=len(left) and len(right) and times[right[0]]-times[left[-1]]<=.24
        if visible[i] or protected[i] or short:proposal.append(source[i]);continue
        hidden[i]=True
        # An invisible arm uses the SAME physical hinge for preparation and
        # release. Crop fitting cannot override this conservative trajectory.
        frame,flex,hand=idle[0].copy(),idle[1],idle[2].copy()
        contributions=[]
        for j,window,hold in ((int(left[-1]) if len(left) else None,.50,.08),
                              (int(right[0]) if len(right) else None,.50,0.)):
            if j is None:continue
            amount=float(np.clip(1-(abs(times[j]-times[i])-hold)/window,0,1));amount=amount*amount*(3-2*amount)
            if amount>0:contributions.append((j,amount))
        total=sum(x[1] for x in contributions)
        for j,amount in contributions:
            gain=amount/max(1.,total)
            frame=frame.slerp(source[j][0],gain);flex=flex*(1-gain)+source[j][1]*gain;hand=hand.slerp(source[j][2],gain)
        proposal.append((frame,flex,hand))
    # Solve the interval first; temporal filtering is acausal, with no delayed
    # reaction at the first measured frame. Strong observations anchor it.
    weights=np.where(hidden,.7,8.);weights[protected]=10000.
    frames=quaternions(times,[p[0] for p in proposal],.075,weights)
    flexes=trajectory(times,[p[1] for p in proposal],.065,weights)
    hands=quaternions(times,[p[2] for p in proposal],.065,weights)
    previous=None;hinge={};changes=[];last_frame=None;last_flex=None;last_hand=None;last_roll=None
    for i in range(n):
        if i%150==0:print('BODY_GUARD_PROGRESS',side,i,len(changes),flush=True)
        if not allowed[i] or protected[i]:previous=None;hinge={};last_frame=None;last_flex=None;last_hand=None;last_roll=None;continue
        load(i);aa,old_e,old_w=[b.head.copy() for b in (a,e,w)];body=body_rotation()
        upper=(old_e-aa).length;lower=(old_w-old_e).length
        frame=body@frames[i];flex=float(flexes[i]);hand=body@hands[i]
        dt=times[i]-times[i-1] if i else 1/data.get('fps',30)
        # Body-relative speed bounds include the hinge normal itself, so skin
        # cannot flip 180 degrees while the elbow endpoint stays almost still.
        if last_frame is not None:
            angle=2*math.acos(min(1.,abs(last_frame.dot(frames[i]))));limit=math.radians(420*dt)
            local=last_frame.slerp(frames[i],min(1.,limit/max(angle,1e-8)))
            frame=body@local;flex=last_flex+float(np.clip(flex-last_flex,-limit,limit))
            angle=2*math.acos(min(1.,abs(last_hand.dot(hands[i]))));local_hand=last_hand.slerp(hands[i],min(1.,math.radians(660*dt)/max(angle,1e-8)))
            hand=body@local_hand
        # Follow the requested trajectory, not the last collision projection.
        # Feeding a projected elbow back into the follower pulls it toward an
        # unsafe idle every frame and quantizes the next collision search. A
        # constant input then oscillates at the angular speed limit.
        last_frame=body.inverted()@frame
        last_flex=flex
        last_hand=body.inverted()@hand
        if repair_idle:
            # An existing take already has reviewed palm orientations. Rebuild
            # its unsupported hinge without replacing that orientation by an
            # idle-to-observation palm blend (which can choose the long turn).
            hand=body@source[i][2]
        u=frame@Vector((0,1,0));v=Quaternion(frame@Vector((1,0,0)),flex)@u
        ee=aa+u*upper;ww=ee+v*lower
        envelope=torso_envelope(rig,geometry);before=arm_clearance(aa,ee,ww,envelope)
        angle=math.degrees((aa-ee).angle(ww-ee));floor=35.
        se,sw=ee,ww; margin=before
        if before<-.001 or angle<floor-.25:
            se,sw,margin=choose_chain(aa,ee,ww,envelope,torso_front_plane(rig),floor,side,previous,False)
        hand=bounded_wrist_rotation(hand,sw-se,wrist_limit(math.degrees((aa-se).angle(sw-se))))
        anatomical_rotations(a,e,w,aa,se,sw,hand,set_global_rotation,view,hinge)
        controller=rig.pose.bones.get('HandTwist_'+side)
        if controller is not None:
            palm=w.matrix.to_quaternion().copy();q=controller.rotation_quaternion.copy()
            if q.w<0:q.negate()
            _,roll=q.to_swing_twist('Y')
            roll=float(np.clip(roll,-math.radians(170),math.radians(170)))
            if last_roll is not None:
                roll=last_roll+float(np.clip(roll-last_roll,-math.radians(480*dt),math.radians(480*dt)))
            last_roll=roll;controller.rotation_quaternion=Quaternion((0,1,0),roll);view.update();set_global_rotation(w,palm);view.update()
        for prefix in PREFIXES:
            name=prefix+side
            if name in channels:
                original_q=Quaternion(original[name][i][1]);amount=float(envelope_weight[i])
                q=original_q.slerp(rig.pose.bones[name].rotation_quaternion,amount)
                channels[name][i]=channels[name][i][0],tuple(q) if amount>0 else original[name][i][1]
        changes.append({'frame':i+1,'before_clearance':arm_clearance(aa,old_e,old_w,envelope),'after_clearance':margin,
                        'wrist_shift':(sw-old_w).length,'observed':bool(visible[i]),'minimum_elbow':floor,'hidden_prior':bool(hidden[i])})
        previous=(e.head.copy(),w.head.copy())
        if len(policy)==n and side in policy[i]:policy[i][side]['body_guard_minimum_elbow']=floor
    if report is not None:report.append({'model':'skinned-torso-clearance-v1','side':side,'changes':changes,'trigger_frames':(ids+1).tolist(),
        'limits':'torso proxy / upper and forearm capsules; no fingers, head, cloth or full mesh collision'})

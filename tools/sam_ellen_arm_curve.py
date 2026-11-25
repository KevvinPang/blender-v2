"""Continuous acceleration fitting inside the whole-chain safety corridor."""
import math
import numpy as np
from mathutils import Quaternion


def fit_palm_curve(times, rotations, forearms, limits, locked, fast=None, reliability=None):
    """Bound the final palm itself, not just the two links ending at it.

    A continuous wrist position does not imply continuous palm orientation.
    Alternate anatomical cones and physical rotation budgets before converting
    to local MMD channels; never change fingers or protected palm anchors.
    """
    from sam_ellen_arm_chain import bounded_wrist_rotation
    from sam_ellen_temporal import quaternions
    times=np.asarray(times);locked=np.asarray(locked,bool)
    raw=[Quaternion(q).normalized() for q in rotations]
    def cone(q,i):
        return raw[i].copy() if locked[i] else bounded_wrist_rotation(q,forearms[i],limits[i])
    result=[cone(q,i) for i,q in enumerate(raw)]
    # Leave room for the inherited MMD motion and the C1 interpolation peak.
    # The committed scene is also checked at quarter-frame intervals.
    budget=np.radians(900.)*np.diff(times)
    def turn(a,b):return 2*math.acos(min(1.,abs(a.dot(b))))
    edges=[i for i in range(len(raw)-1) if not (locked[i] and locked[i+1])]
    violating=any(turn(result[i],result[i+1])>budget[i]+1e-5 for i in edges)
    reliable=np.ones(len(raw)) if reliability is None else np.clip(reliability,0.,1.)
    # Translation speed cannot establish a rapid palm rotation. Require
    # coherent angular motion over adjacent observations independently.
    angular=np.zeros(len(raw));velocities=[]
    for a,b,dt in zip(raw[:-1],raw[1:],np.diff(times)):
        delta=b@a.inverted()
        if delta.w<0:delta.negate()
        axis,angle=delta.to_axis_angle();velocities.append(np.asarray(axis)*angle/dt)
    for i in range(1,len(raw)-1):
        a,b=velocities[i-1:i+1];sa,sb=np.linalg.norm(a),np.linalg.norm(b)
        if min(reliable[i-1:i+2])>.5 and a@b/max(1e-8,sa*sb)>.8 and max(sa,sb)<12:
            angular[i]=float(np.clip((min(sa,sb)-2.)/5.,0,1))
    weights=(1.+11*reliable)*(1+20*angular)
    weights[locked]=1e9
    result=[cone(q,i) for i,q in enumerate(quaternions(times,result,.18,weights))]
    for _ in range(96):
        for forward,order in ((True,range(1,len(raw))),(False,range(len(raw)-2,-1,-1))):
            for i in order:
                if locked[i]:continue
                j=i-1 if forward else i+1;step=budget[min(i,j)]
                angle=turn(result[j],result[i])
                if angle>step:result[i]=cone(result[j].slerp(result[i],float(step/angle)),i)
        excess=max((turn(result[i],result[i+1])-budget[i] for i in edges),default=0.)
        if excess<1e-5:break
    if excess>1e-4:raise ValueError('手掌方向无法安全连续连接受保护动作，已保留原动作')
    return result,{'model':'final-palm-continuity-v2',
        'changed_frames':[i+1 for i,(a,b) in enumerate(zip(raw,result)) if turn(a,b)>1e-4]}


def fit_curve(times,rows,rotations,flexes,upper,lower):
    """Refine graph knots without quantizing the final trajectory to that graph.

    Observed fast motion is weighted more strongly. Protected poses are exact.
    Every accepted update includes the interior of each moving-torso edge.
    """
    from sam_ellen_temporal import quaternions,trajectory
    from sam_ellen_arm_trajectory import _rotate,_angle
    from sam_ellen_body_clearance import chains_safe
    from sam_ellen_motion_evidence import edge_limits
    times=np.asarray(times);q=np.array(rotations,copy=True);f=np.array(flexes,copy=True)
    locked=np.array([r.get('locked',False) for r in rows]);visible=np.array([r['visible'] for r in rows])
    max_flex=np.radians([180-r.get('minimum_elbow',35.) for r in rows])
    bodies=[r['body'] for r in rows];a=np.array([r['a'] for r in rows]);env=np.array([r['env'] for r in rows])
    dt=np.diff(times);fast=np.array([r.get('fast_motion',0.) for r in rows])
    # Equivalent to reducing the time scale from .105 to .04 seconds for
    # corroborated fast motion, without switching filters at a hard boundary.
    weights=np.where(visible,3.+9*np.array([r.get('tracking_reliability',1.) for r in rows]),1.)*(1+46*fast);weights[locked]=1e9
    def geometry(q,f):
        u=_rotate(q,[0,1,0]);v=u*np.cos(f)[:,None]+_rotate(q,[0,0,1])*np.sin(f)[:,None]
        e=a+u*upper;return e,e+v*lower
    reference_e,reference_w=geometry(q,f)
    reference_cost=[float(r['pose_objective'](reference_e[i:i+1],reference_w[i:i+1])[0])
        if r.get('pose_objective') else None for i,r in enumerate(rows)]
    def safe(q,f,affected=None):
        ids=np.arange(len(q)) if affected is None else np.asarray(affected)
        editable=ids[~locked[ids]]
        if np.any((f[editable]<math.radians(15)-1e-5)|(f[editable]>max_flex[editable]+1e-5)):return False
        # chains_safe has one socket; batch moving sockets explicitly.
        e,w=geometry(q,f)
        for i in editable:
            if reference_cost[i] is not None and visible[i]:
                if rows[i]['pose_objective'](e[i:i+1],w[i:i+1])[0]>reference_cost[i]+.25+2.5*(1-rows[i].get('tracking_reliability',1.)):return False
            if not chains_safe(a[i],e[i:i+1],w[i:i+1],env[i],upper,lower,margin=.0005)[0]:return False
        edges=range(len(q)-1) if affected is None else np.unique(np.clip(np.r_[ids-1,ids],0,len(q)-2))
        for i in edges:
            if locked[i] and locked[i+1]:continue
            rate,flex_rate=edge_limits(rows[i],rows[i+1])
            if _angle(q[i:i+1],q[i+1:i+2])[0,0]>math.radians(rate*dt[i])+1e-5:return False
            if abs(f[i+1]-f[i])>math.radians(flex_rate*dt[i])+1e-5:return False
            lo=_rotate(q[i],[0,1,0])*np.cos(f[i])+_rotate(q[i],[0,0,1])*np.sin(f[i])
            ln=_rotate(q[i+1],[0,1,0])*np.cos(f[i+1])+_rotate(q[i+1],[0,0,1])*np.sin(f[i+1])
            if np.arccos(np.clip(lo@ln,-1,1))>math.radians(rate*dt[i])+1e-5:return False
            for t in (.25,.5,.75):
                last=q[i+1].copy()
                if q[i]@last<0:last=-last
                qi=q[i]*(1-t)+last*t;qi/=np.linalg.norm(qi)
                fi=f[i]*(1-t)+f[i+1]*t;ai=a[i]*(1-t)+a[i+1]*t
                u=_rotate(qi,[0,1,0]);v=u*np.cos(fi)+_rotate(qi,[0,0,1])*np.sin(fi)
                ei=ai+u*upper;wi=ei+v*lower
                if not chains_safe(ai,ei[None,:],wi[None,:],env[i]*(1-t)+env[i+1]*t,upper,lower,margin=.0005)[0]:return False
        return True
    # Fit relative to the moving chest, so normal torso motion is not damped.
    local=[b.inverted()@Quaternion(x) for b,x in zip(bodies,q)]
    fitted=quaternions(times,local,.105,weights)
    target=np.array([tuple(b@x) for b,x in zip(bodies,fitted)])
    target[locked]=q[locked]
    tf=np.clip(trajectory(times,f,.105,weights),math.radians(15),max_flex);tf[locked]=f[locked]
    target[np.sum(q*target,axis=1)<0]*=-1
    accepted=0.
    for gain in (1.,.8,.6,.4,.25,.125,.0625):
        candidate=q*(1-gain)+target*gain;candidate/=np.linalg.norm(candidate,axis=1)[:,None]
        cf=f*(1-gain)+tf*gain
        if safe(candidate,cf):q,f=candidate,cf;accepted=gain;break
    # A single tight shoulder socket must not disable smoothing everywhere.
    # Every local acceptance also checks both edges adjoining the window.
    local_accepted=0
    if accepted<1. and len(q)>5:
        for start in range(0,len(q),12):
            ids=np.arange(max(0,start-6),min(len(q),start+19))
            if len(ids)<4:continue
            taper=np.sin(np.linspace(0,math.pi,len(ids)))**2
            taper[locked[ids]]=0.
            for gain in (1.,.6,.3,.125):
                alpha=taper*gain
                candidate=q.copy();cf=f.copy();local=target[ids].copy()
                local[np.sum(q[ids]*local,axis=1)<0]*=-1
                candidate[ids]=q[ids]*(1-alpha[:,None])+local*alpha[:,None]
                candidate[ids]/=np.linalg.norm(candidate[ids],axis=1)[:,None]
                cf[ids]=f[ids]*(1-alpha)+tf[ids]*alpha
                if safe(candidate,cf,ids):
                    q,f=candidate,cf;local_accepted+=1;break
    return q,f,{'model':'continuous-acceleration-fit-v2','gain':accepted,
        'local_windows_accepted':local_accepted,'fast_motion_samples':int(np.count_nonzero(fast>.3)),
        'observed_weight':12.,'time_scale_seconds':.105,'fast_time_scale_seconds':.04}


def smooth_keys(action,bone,locked):
    """C1 quaternion components with monotone Hermite tangents, arms only.

    Local bounds prevent component overshoot. Evaluated skeletal collision is
    still verified after commit, since a bounded component is not a safe limb.
    Authored/protected edges retain their exact original linear interpolation.
    """
    path=bone.path_from_id('rotation_quaternion');locked=np.asarray(locked,bool)
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for curve in bag.fcurves:
                    if curve.data_path!=path:continue
                    keys=curve.keyframe_points
                    if len(keys)!=len(locked) or len(keys)<3:continue
                    x=np.array([k.co.x for k in keys]);y=np.array([k.co.y for k in keys])
                    h=np.diff(x);slopes=np.diff(y)/h
                    tangent=np.zeros(len(keys));tangent[0]=slopes[0];tangent[-1]=slopes[-1]
                    for i in range(1,len(keys)-1):
                        if slopes[i-1]*slopes[i]>0:
                            tangent[i]=np.sign(slopes[i])*min(abs((slopes[i-1]+slopes[i])*.5),3*min(abs(slopes[i-1]),abs(slopes[i])))
                    for i,key in enumerate(keys):
                        key.handle_left_type=key.handle_right_type='FREE'
                        left=h[max(0,i-1)]/3;right=h[min(len(h)-1,i)]/3
                        key.handle_left=(x[i]-left,y[i]-tangent[i]*left)
                        key.handle_right=(x[i]+right,y[i]+tangent[i]*right)
                        if i+1<len(keys) and not (locked[i] or locked[i+1]):key.interpolation='BEZIER'
                    curve.update()


def linearize_unsafe_edges(action,rig,rows):
    """Remove a C1 overshoot only on failed subframe edges, then revalidate.

    This changes interpolation, never key poses or physical safety thresholds.
    Compound local rotations can overshoot the world-space speed budget even
    when every sampled world rotation satisfied it.
    """
    from sam_ellen_motion_evidence import HARD_ARM_RATE
    from sam_ellen_arm_policy import PREFIXES
    spans={}
    for row in rows:
        if not row.get('intermediate'):continue
        for side,arm in row.get('arms',{}).items():
            if any(v>HARD_ARM_RATE for v in arm.get('angular_speed_degrees_per_second',{}).values()):
                spans.setdefault(side,set()).add(math.floor(row['frame']))
    if not spans:return []
    from sam_ellen_sequence import curves
    paths={rig.pose.bones[p+side].path_from_id('rotation_quaternion'):side
           for side in spans for p in PREFIXES}
    changed=[]
    for curve in curves(action):
        side=paths.get(curve.data_path)
        if side is None:continue
        for key in curve.keyframe_points:
            if any(abs(key.co.x-start)<1e-5 for start in spans[side]) and key.interpolation!='LINEAR':
                key.interpolation='LINEAR';changed.append([side,float(key.co.x),curve.array_index])
        curve.update()
    return changed

"""Reject opposite elbow branches in the final, camera-evaluated arm owner."""
import math
import numpy as np
from mathutils import Vector,Quaternion


def align_forearms(scratch,rig,channels,data,side,cameras,allowed,protected,floors,mirror=False):
    from sam_ellen_arm_chain import direction_evidence,torso_front_plane,anatomical_rotations,bounded_wrist_rotation
    from sam_ellen_arm_policy import front_circle,transport_bend,wrist_limit
    from sam_ellen_image_chain import fit_direction
    from sam_ellen_screen import project,unproject
    from sam_ellen_arm_visibility import observed_frames
    from sam_ellen_temporal import trajectory
    from sam_ellen_retarget import set_global_rotation
    from sam_ellen_entry_solver import fit_translation
    from sam_ellen_entry import hand_at
    from sam_ellen_arm_visibility import hand_landmark_mask
    from sam_ellen_screen_fit import hand_points
    from sam_ellen_arm_policy import camera_reachable_goal
    times=np.array([r['time'] for r in data['frames']]);n=len(times);view=scratch.view_layers[0]
    original={name:list(rows) for name,rows in channels.items()}
    evidence=direction_evidence(data,side,{'mirror':mirror});visible=observed_frames(data,side)
    allowed=np.asarray(allowed,bool);protected=np.asarray(protected,bool)
    masks=data['screen'].get('punch_tracking',{}).get('frames',[])
    silhouettes=np.array([bool(row.get('hands',{}).get(side,{}).get('silhouette')) for row in masks]) if len(masks)==n else np.zeros(n,bool)
    allowed=allowed&(~protected|silhouettes)
    a,e,w=[rig.pose.bones[p+'_'+side] for p in ('Arm','Elbow','Wrist')]
    def load(i):
        for name,rows in original.items():rig.pose.bones[name].rotation_quaternion=rows[i][1]
        view.update()
    geometry=[];bad=np.zeros(n,bool)
    for i in range(n):
        load(i);aa,ee,ww=[b.head.copy() for b in (a,e,w)]
        axis=(ww-aa).normalized();bend=transport_bend(ee-aa,None,axis)
        geometry.append((aa,ee,ww,bend,w.matrix.to_quaternion().copy(),torso_front_plane(rig)))
        cue=evidence[i]
        if allowed[i] and visible[i] and cue is not None and cue[1]>=.5:
            v=np.array(project(ee,cameras[i])-project(ww,cameras[i]));v/=max(1e-8,np.linalg.norm(v))
            bad[i]=math.degrees(math.acos(float(np.clip(v@cue[0],-1,1))))>35
    # An observed episode is one continuous elbow branch. Repair its entry,
    # evidence gaps and release together, never splice a corrected single pose.
    active=visible&allowed
    valid=np.flatnonzero(active)
    for left,right in zip(valid,valid[1:]):
        if times[right]-times[left]<=.24 and allowed[left:right+1].all():active[left:right+1]=True
    edges=np.diff(np.r_[False,active,False].astype(int));reports=[]
    for start,stop in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        if not bad[start:stop].any():continue
        begin=max(0,int(np.searchsorted(times,times[start]-.60)))
        end=min(n-1,int(np.searchsorted(times,times[stop-1]+.35)))
        while begin<start and not allowed[begin:start].all():begin+=1
        while end>=stop and not allowed[stop:end+1].all():end-=1
        solved={};hands={};previous=None
        for i in range(start,stop):
            load(i)
            aa,ee,ww,bend,hand,plane=geometry[i];upper=(ee-aa).length;lower=(ww-ee).length
            cue=evidence[i]
            observed_hand=hand_at(data['screen'],int(i),side)
            local=np.array([hand.inverted()@(p-ww) for p in hand_points(rig,side)])
            rotation=hand.copy()
            for iteration in range(4):
                if observed_hand is not None and not protected[i]:
                    observation={'points':observed_hand['points'],'mask':hand_landmark_mask(observed_hand)}
                    target=fit_translation(local,rotation,ww,observation,cameras[i],data['screen']['aspect'],mirror)
                    target=camera_reachable_goal(aa,target,upper,lower,floors[i],cameras[i],plane)
                    ee,ww,bend=front_circle(aa,target,upper,lower,bend,plane,floors[i])
                if cue is not None and cue[1]>=.25:
                    de,dw,db,check=fit_direction(aa,ee,ww,upper,lower,cameras[i],cue,plane,floors[i],previous,
                                               lock_depth=bool(protected[i]))
                    if check.get('feasible') and check['direction_error_degrees']<=35:ee,ww,bend=de,dw,db
                elif previous is not None:
                    ee,ww,bend=front_circle(aa,ww,upper,lower,previous[0]-aa,plane,floors[i])
                rotation=bounded_wrist_rotation(hand,ww-ee,wrist_limit(math.degrees((aa-ee).angle(ww-ee))))
            solved[i]=(ee.copy(),ww.copy(),bend.copy());hands[i]=rotation;previous=(ee.copy(),ww.copy())
        ids=np.arange(begin,end+1);bases=[];phases=[];depths=[];axes=[];oldaxis=None;base=None;last=0.
        for i in ids:
            aa,ee,ww,bend,hand,plane=geometry[i]
            if i in solved:ee,ww,bend=solved[i]
            axis=(ww-aa).normalized()
            if i<start:
                target=transport_bend(solved[start][2],None,axis);idle=transport_bend(geometry[begin][3],None,axis)
                t=float(np.clip((times[i]-times[begin])/.34,0,1));t=t*t*(3-2*t)
                turn=math.atan2(axis.dot(idle.cross(target)),idle.dot(target));bend=Quaternion(axis,t*turn)@idle
            elif i>=stop:
                target=transport_bend(geometry[end][3],None,axis);prior=transport_bend(solved[stop-1][2],None,axis)
                t=float((times[i]-times[stop-1])/max(1e-8,times[end]-times[stop-1]));t=t*t*(3-2*t)
                turn=math.atan2(axis.dot(prior.cross(target)),prior.dot(target));bend=Quaternion(axis,t*turn)@prior
            base=bend.copy() if base is None else transport_bend(base,oldaxis,axis)
            phase=math.atan2(axis.dot(base.cross(bend)),base.dot(bend));phase+=math.tau*round((last-phase)/math.tau)
            bases.append(base.copy());axes.append(axis);phases.append(phase);last=phase;oldaxis=axis
            depths.append(math.log(max(.04,-(cameras[i][0]@ww).z)))
        weights=np.array([8. if start<=i<stop else 1. for i in ids]);weights[[0,-1]]=10000.
        phases=trajectory(times[ids],phases,.055,weights)
        depths=trajectory(times[ids],depths,.045,weights)
        hinge={}
        for k,i in enumerate(ids):
            if i in (begin,end):continue
            load(int(i));aa,ee,ww,bend,hand,plane=geometry[i];upper=(ee-aa).length;lower=(ww-ee).length
            target=solved[i][1] if i in solved else ww
            target=geometry[i][2].copy() if protected[i] else unproject(project(target,cameras[i]),math.exp(depths[k]),cameras[i])
            axis=(target-aa).normalized();bend=transport_bend(Quaternion(axes[k],float(phases[k]))@bases[k],axes[k],axis)
            ee,target,bend=front_circle(aa,target,upper,lower,bend,plane,float(floors[i]))
            hand=bounded_wrist_rotation(hands.get(i,hand),target-ee,wrist_limit(math.degrees((aa-ee).angle(target-ee))))
            anatomical_rotations(a,e,w,aa,ee,target,hand,set_global_rotation,view,hinge)
            for prefix in ('Arm_','Elbow_','Wrist_','ArmTwist_','HandTwist_'):
                name=prefix+side
                if name in channels:channels[name][i]=channels[name][i][0],tuple(rig.pose.bones[name].rotation_quaternion)
        before_errors=[];after_errors=[]
        for i in range(start,stop):
            cue=evidence[i]
            if cue is None or cue[1]<.5:continue
            for name,rows in channels.items():rig.pose.bones[name].rotation_quaternion=rows[i][1]
            view.update()
            for values,ee,ww in ((before_errors,geometry[i][1],geometry[i][2]),(after_errors,e.head,w.head)):
                direction=np.array(project(ee,cameras[i])-project(ww,cameras[i]));direction/=max(1e-8,np.linalg.norm(direction))
                values.append(math.degrees(math.acos(float(np.clip(direction@cue[0],-1,1)))))
        accepted=not after_errors or max(after_errors)<=max(before_errors)+1.
        if not accepted:
            for prefix in ('Arm_','Elbow_','Wrist_','ArmTwist_','HandTwist_'):
                name=prefix+side
                if name in channels:channels[name][begin:end+1]=original[name][begin:end+1]
        reports.append({'side':side,'begin':int(begin)+1,'end':int(end)+1,'observed':[int(start)+1,int(stop)],
                        'trigger_frames':(np.flatnonzero(bad[start:stop])+start+1).tolist(),
                        'accepted':accepted,'before_direction_error':max(before_errors,default=0.),
                        'after_direction_error':max(after_errors,default=0.)})
    data.setdefault('_sam_direction_repairs',[]).extend(reports)
    return reports


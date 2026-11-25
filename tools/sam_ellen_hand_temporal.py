"""Continuous final palm goals: image motion, depth, rotation, then arm geometry."""
import numpy as np
from mathutils import Quaternion, Vector


def runs(active):
    edges=np.diff(np.r_[False,np.asarray(active,dtype=bool),False].astype(int))
    return zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))


def smooth_goals(times,positions,rotations,bends,cameras,active,reliability):
    """Fit camera rays separately from weak monocular depth, without phase lag."""
    from sam_ellen_temporal import trajectory,quaternions
    p=list(positions);q=list(rotations);b=list(bends)
    for lo,hi in runs(active):
        if hi-lo<3:continue
        t=times[lo:hi];weights=np.maximum(.05,reliability[lo:hi])
        camera=np.array([cameras[i][0]@Vector(p[i]) for i in range(lo,hi)])
        perspective=np.array([cameras[i][3]=='PERSP' for i in range(lo,hi)])
        depth=np.where(perspective,np.maximum(.05,-camera[:,2]),1.)
        rays=camera[:,:2]/depth[:,None]
        rays=trajectory(t,rays,.055,weights,robust=.02)
        camera[:,2:]=trajectory(t,camera[:,2:],.10,weights,robust=.035)
        depth=np.where(perspective,np.maximum(.05,-camera[:,2]),1.)
        camera[:,:2]=rays*depth[:,None]
        filtered=quaternions(t,q[lo:hi],.11,weights)
        planes=trajectory(t,b[lo:hi],.10,weights,robust=.5)
        for k,i in enumerate(range(lo,hi)):
            p[i]=tuple(cameras[i][0].inverted()@Vector(camera[k]))
            q[i]=tuple(filtered[k]);b[i]=tuple(planes[k])
    return p,q,b


def smooth_fingers(channels,times,side,active):
    """Remove residual joint tremor after all per-frame articulation choices."""
    from sam_ellen_temporal import quaternions
    for name,values in channels.items():
        if not name.endswith('_'+side) or not name.startswith(('Thumb','IndexFinger','MiddleFinger','RingFinger','LittleFinger')):continue
        for lo,hi in runs(active):
            if hi-lo<3:continue
            filtered=quaternions(times[lo:hi],[q for _,q in values[lo:hi]],.055)
            for k,i in enumerate(range(lo,hi)):values[i]=values[i][0],tuple(filtered[k])


def refit_palm(scratch,rig,channels,data,options,cameras,observations,metadata,allowed,evidence,report):
    from sam_ellen_screen_fit import fit_hand,hand_points
    from sam_ellen_arm_chain import refine_goal,torso_front_plane
    from sam_ellen_retarget import correct_contacts
    side=options['side'];view=scratch.view_layers[0]
    times=np.array([r['time'] for r in data['frames']])
    active=np.array([bool(allowed[i] and observations[i] is not None and i in metadata) for i in range(len(times))])
    smooth_fingers(channels,times,side,active)
    originals={n:list(rows) for n,rows in channels.items()}
    positions=[];rotations=[];bends=[];reliability=[];previous_fit=None
    for i in range(len(times)):
        for n,rows in originals.items():rig.pose.bones[n].rotation_quaternion=rows[i][1]
        view.update()
        a,e,w=[rig.pose.bones[n+'_'+side] for n in ('Arm','Elbow','Wrist')]
        position=w.head.copy();q=w.matrix.to_quaternion();direction=(position-a.head).normalized()
        bend=e.head-a.head;bend-=direction*bend.dot(direction)
        obs=observations[i]
        if active[i]:
            model=hand_points(rig,side);target,mask,_=metadata[i]
            reach=(e.head-a.head).length+(w.head-e.head).length
            position,delta,previous_fit=fit_hand(model,target,mask,cameras[i],options['depth'],previous_fit,
                match_image=True,reach=(a.head.copy(),reach*.985),articulate=True,
                joint_weights=obs.get('joint_weights'))
            q=delta@q
            support=obs.get('joint_weights',mask)
            reliability.append((.15 if obs.get('held') and not obs.get('interpolated') else 1.)
                               *max(.15,float(np.mean(np.asarray(support)[[5,9,13,17]]))))
        else:
            previous_fit=None;reliability.append(.05)
        positions.append(tuple(position));rotations.append(tuple(q));bends.append(tuple(bend.normalized()))
        yield i+1,3*len(times)
    # Resolve anatomical/image depth ambiguity first. Smooth the complete final
    # goals afterwards, so no subsequent candidate search can undo continuity.
    previous=None;details={}
    for i in range(len(times)):
        if not active[i]:previous=None;continue
        for n,rows in originals.items():rig.pose.bones[n].rotation_quaternion=rows[i][1]
        view.update()
        a,e,w=[rig.pose.bones[n+'_'+side] for n in ('Arm','Elbow','Wrist')]
        target,mask,anchors=metadata[i];model=hand_points(rig,side)
        delta=Quaternion(rotations[i])@w.matrix.to_quaternion().inverted()
        offsets=np.array([delta@(model[j]-w.head) for j in anchors])
        position,bend,row=refine_goal(a,e,w,Vector(positions[i]),Vector(bends[i]),cameras[i],evidence[i],previous,
            depth_strength=.1,palm=(offsets,target[anchors],.15 if observations[i].get('held') else 1.),
            front_plane=torso_front_plane(rig),hand_axis=Quaternion(rotations[i])@Vector((0,1,0)))
        positions[i]=tuple(position);bends[i]=tuple(bend)
        previous=tuple(np.array(v) for v in (a.head,row['elbow_position'],position,row['normal']))
        details[i]=row
        yield len(times)+i+1,3*len(times)
    positions,rotations,bends=smooth_goals(times,positions,rotations,bends,cameras,active,np.array(reliability))
    def goal(i):
        if not active[i]:return None
        report[i].update(details[i],palm_refitted_after_fingers=True,
                         final_palm_trajectory=True,final_depth_seconds=.10,
                         final_rotation_seconds=.11,continuous_landmark_support=True)
        return Vector(positions[i]),Quaternion(rotations[i]),Vector(bends[i])
    for done,total in correct_contacts(scratch,rig,channels,data,
            {'side':side,'stable_ik':True,'anatomical':True},goal_provider=goal):
        yield 2*len(times)+done,3*len(times)

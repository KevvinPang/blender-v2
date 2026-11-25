"""Re-solve image-induced arm folds from a coherent capture checkpoint."""
import math
import numpy as np
from mathutils import Quaternion,Vector


PREFIXES=('Arm_','Elbow_','Wrist_','ArmTwist_','HandTwist_')


def repair_folds(scratch,rig,channels,reference,data,options,cameras,report,protected=None):
    """Keep all other motion intact; repair the entire affected hinge interval.

    A reference is captured before the image-only refinement passes. It is not
    a rest pose: it contains the current clip's capture, visibility and gesture
    corrections. Measured finger animation always comes from current channels.
    """
    if not reference or options.get('space')!='FRAME' or options.get('strength',1.)<.999:return
    from sam_ellen_screen_fit import landmark_observations,hand_points
    from sam_ellen_arm_chain import frame_rotation,anatomical_rotations,direction_evidence,torso_front_plane,front_safe_positions,bounded_wrist_rotation
    from sam_ellen_image_refine import fit_chain
    from sam_ellen_retarget import set_global_rotation
    from sam_ellen_temporal import quaternions,trajectory
    view=scratch.view_layers[0];times=np.array([r['time'] for r in data['frames']]);relative=times-times[0]
    original={n:list(rows) for n,rows in channels.items()}
    allowed=(relative>=options.get('start',0))&((relative<=options['end']) if options.get('end',0)>0 else True)

    def pose(i,side,ref=False,reference_index=None):
        for n,rows in original.items():rig.pose.bones[n].rotation_quaternion=rows[i][1]
        if ref:
            for prefix in PREFIXES:
                n=prefix+side
                if n in reference:rig.pose.bones[n].rotation_quaternion=reference[n][i if reference_index is None else reference_index][1]
        view.update()

    def geometry(side):
        a,e,w=[rig.pose.bones[n+'_'+side] for n in ('Arm','Elbow','Wrist')]
        u=e.head-a.head;v=w.head-e.head;hand=w.matrix.to_quaternion()
        return (a.head.copy(),e.head.copy(),w.head.copy(),frame_rotation(u,u.cross(v)),u.angle(v),hand,
                math.degrees(v.angle(hand@Vector((0,1,0)))))

    for side in (('L','R') if options['side']=='BOTH' else (options['side'],)):
        if side==options.get('exclude_side'):continue
        if not all(p+side in channels and p+side in reference for p in PREFIXES[:3]):continue
        current=[];safe=[]
        for i in range(len(times)):
            pose(i,side);current.append(geometry(side))
            pose(i,side,True);safe.append(geometry(side))
        # A tight elbow alone is a valid raise. Reject the combination of a
        # deeply folded elbow AND a highly bent wrist introduced by image fit.
        bad=np.array([180-math.degrees(c[4])<45 and c[6]>55 and r[6]<40 and 180-math.degrees(r[4])>42
                      for c,r in zip(current,safe)],dtype=bool)&allowed
        if protected is not None and side in protected:bad&=~np.asarray(protected[side],dtype=bool)
        ids=np.flatnonzero(bad)
        if not len(ids):continue
        obs=landmark_observations(data['screen'],side,space='FRAME')
        measured=np.array([o is not None and (not o.get('held') or o.get('interpolated')) for o in obs],dtype=bool)&allowed
        if protected is not None and side in protected:measured&=~np.asarray(protected[side],dtype=bool)
        core=bad.copy();edges=np.diff(np.r_[False,measured,False].astype(int))
        for lo,hi in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
            if bad[lo:hi].any():core[lo:hi]=True
        distance=np.min(abs(times[:,None]-times[np.flatnonzero(core)][None,:]),axis=1)
        x=np.clip(1-distance/.35,0,1);envelope=x*x*(3-2*x)
        envelope*=allowed
        if protected is not None and side in protected:envelope*=~np.asarray(protected[side],dtype=bool)
        active=np.flatnonzero(envelope>0);evidence=direction_evidence(data,side,options)
        # Fit a whole interval from the same reference branch, then regularize
        # hinge orientation and flex in time. No wrist-only endpoint clamp.
        trusted=np.array([j for j,r in enumerate(safe) if measured[j] and r[6]<35 and 180-math.degrees(r[4])>45])
        fitted={};hinge={};borrowed=[]
        for count,i in enumerate(active):
            anchor_index=i
            # The first few hand-entry frames can already be folded in the
            # checkpoint. Borrow a nearby measured branch rather than treating
            # that folded prior as an anatomical reference.
            if measured[i] and safe[i][6]>45 and 180-math.degrees(safe[i][4])<45 and len(trusted):
                nearest=int(trusted[np.argmin(abs(times[trusted]-times[i]))])
                if abs(times[nearest]-times[i])<.35:
                    anchor_index=nearest;borrowed.append(int(i)+1)
            pose(i,side,True,anchor_index);a,e,w=[rig.pose.bones[n+'_'+side] for n in ('Arm','Elbow','Wrist')]
            origin=a.head.copy();q=w.matrix.to_quaternion();o=obs[i]
            if o is not None and (not o.get('held') or o.get('interpolated')):
                offsets=np.array([q.inverted()@(p-w.head) for p in hand_points(rig,side)])
                target=o['points'].copy();target[:,0]*=cameras[i][2]/data['screen']['aspect']
                if options.get('mirror'):target[:,0]=cameras[i][2]-target[:,0]
                weights=np.asarray(o.get('joint_weights',o['mask']),float)*o['mask']
                ee,ww,qq,row=fit_chain(origin,e.head,w.head,q.to_matrix(),offsets,target,weights,cameras[i],
                    evidence[i],None,wrist_limit=math.radians(40),front_plane=torso_front_plane(rig),
                    close_range=True,frame_guard=True,pose_prior=(np.array(e.head),np.array(w.head),2.))
                if (i+1)%10==0 or bad[i] and (i+1)%10==7:
                    row['error_px']*=data.get('video_orientation',{}).get('decoded_height_width',[1920])[0]/1920
                    report.append({'model':'image-fold-fit-sample','side':side,'frame':int(i)+1,**row,
                        'reference_elbow':180-math.degrees(safe[i][4]),'reference_wrist':safe[i][6],
                        'elbow':math.degrees((origin-ee).angle(ww-ee)),
                        'wrist':math.degrees((ww-ee).angle(qq@Vector((0,1,0))))})
                anatomical_rotations(a,e,w,origin,ee,ww,qq,set_global_rotation,view,hinge)
            fitted[i]=geometry(side)
            if count%15==0:yield count+1,len(active)*2
        # Split disconnected repair intervals so filtering cannot borrow a
        # different gesture across a long gap.
        groups=np.split(active,np.flatnonzero(np.diff(active)>1)+1);filtered={}
        for group in groups:
            # Filter the elbow's bend direction separately from the wrist
            # trajectory. Filtering a whole hinge rotation pulls a newly
            # visible hand away from its measured image ray.
            t=times[group];wrists=trajectory(t,[fitted[i][2] for i in group],.025)
            whole_frames=quaternions(t,[fitted[i][3] for i in group],.065)
            whole_flex=trajectory(t,[fitted[i][4] for i in group],.065)
            support=np.array([float(np.clip((int(obs[i]['mask'].sum())-16)/5,0,1))
                              if measured[i] else 0. for i in group])
            support=np.clip(trajectory(t,support,.035),0.,1.)
            bends=[]
            for i in group:
                a,e,w=fitted[i][:3];axis=(w-a).normalized();bend=e-a;v=bend-axis*bend.dot(axis)
                bends.append(tuple(v.normalized()))
            bends=trajectory(t,bends,.065)
            h=quaternions(t,[fitted[i][5] for i in group],.045)
            for i,point,bend,hand,whole,flex,retain in zip(group,wrists,bends,h,whole_frames,whole_flex,support):
                a,e,w=fitted[i][:3];upper=(e-a).length;lower=(w-e).length
                axis=Vector(point)-a;distance=axis.length;axis.normalize()
                lo=math.sqrt(upper**2+lower**2-2*upper*lower*math.cos(math.radians(35)))
                hi=math.sqrt(upper**2+lower**2-2*upper*lower*math.cos(math.radians(165)))
                distance=float(np.clip(distance,lo+1e-7,hi-1e-7));ww=a+axis*distance
                along=(distance*distance+upper*upper-lower*lower)/(2*distance)
                bend=Vector(bend);bend-=axis*bend.dot(axis)
                if bend.length<1e-7:bend=(e-a)-axis*(e-a).dot(axis)
                ee=a+axis*along+bend.normalized()*math.sqrt(max(0.,upper*upper-along*along))
                u=ee-a;v=ww-ee
                frame=whole.slerp(frame_rotation(u,u.cross(v)),float(retain))
                filtered[i]=(frame,float(flex)*(1-float(retain))+u.angle(v)*float(retain),hand)
        hinge={}
        for i in active:
            pose(i,side);a,e,w=[rig.pose.bones[n+'_'+side] for n in ('Arm','Elbow','Wrist')]
            origin=a.head.copy();upper=(e.head-origin).length;lower=(w.head-e.head).length
            c=current[i];frame,flex,hand=filtered[i];weight=float(envelope[i])
            frame=c[3].slerp(frame,weight);flex=c[4]*(1-weight)+flex*weight
            u=frame@Vector((0,1,0));v=Quaternion(frame@Vector((1,0,0)),flex)@u
            ee=origin+u*upper;ww=ee+v*lower
            ee,ww=front_safe_positions(origin,ee,ww,torso_front_plane(rig))
            hand=c[5].slerp(hand,weight);hand=bounded_wrist_rotation(hand,ww-ee,72-32*weight)
            anatomical_rotations(a,e,w,origin,ee,ww,hand,set_global_rotation,view,hinge)
            for prefix in PREFIXES:
                n=prefix+side
                if n in channels:channels[n][i]=channels[n][i][0],tuple(rig.pose.bones[n].rotation_quaternion)
        report.append({'model':'image-fold-recovery-v1','side':side,'trigger_frames':(ids+1).tolist(),
            'modified_frames':(active+1).tolist(),'wrist_limit_degrees':40,'reference':'pre-image capture chain',
            'interval_smoothing_seconds':.065,'boundary_blend_seconds':.35,'borrowed_reference_frames':borrowed})
        yield len(active)*2,len(active)*2

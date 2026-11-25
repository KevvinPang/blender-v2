"""Source-only evidence for upper-arm posture; output crop is never evidence."""
import math
import numpy as np


def posture_evidence(data,side):
    from sam_ellen_arm_visibility import screen_hand,hand_landmark_mask
    n=len(data['frames']);poses=data['screen'].get('pose_arms',[])
    height=data.get('video_orientation',{}).get('decoded_height_width',[1280])[0]
    result=[];last=None
    for i,frame in enumerate(data['frames']):
        now=frame['time'];pose=poses[i].get(side) if len(poses)==n else None
        row={'limit_degrees':35.,'preferred_degrees':12.,'kind':'unobserved-natural-down',
             'upper_image_direction':None,'confidence':0.}
        if pose and pose.get('source')=='independent-video-pose' and pose.get('confidence',0)>=.65 and pose.get('elbow_observed') and all(pose.get('in_frame',[False]*3)[:2]):
            p=np.asarray(pose['points'],float);v=p[1]-p[0];length=float(np.linalg.norm(v))
            if length>.055:
                angle=math.degrees(math.atan2(abs(v[0]),v[1]))
                row.update(limit_degrees=min(170.,max(35.,angle+25.)),preferred_degrees=angle,
                           kind='visible-shoulder-and-elbow',upper_image_direction=(v/length).tolist(),confidence=float(pose['confidence']))
        hand=screen_hand(data['screen'],i,side)
        if hand and hand_landmark_mask(hand)[0]:
            image=np.asarray(frame.get('image_points',[]),float)
            if image.ndim==2 and len(image)>6:
                shoulders=image[[5,6],:2]/height;mid=shoulders.mean(0)
                span=max(.08,float(np.linalg.norm(shoulders[1]-shoulders[0])))
                wrist=np.asarray(hand['points'][0]);height_gain=float(np.clip((mid[1]-wrist[1]+span*.1)/(span*.65),0,1))
                lateral_gain=float(np.clip((abs(wrist[0]-mid[0])/span-.9)/.7,0,1))
                gain=max(height_gain,lateral_gain)
                if gain>0 and row['kind']=='unobserved-natural-down':
                    row.update(limit_degrees=35.+125.*gain,preferred_degrees=12.+45.*gain,
                               kind='visible-hand-requires-reach',confidence=gain)
        if row['kind']!='unobserved-natural-down':last=(now,dict(row))
        elif last is not None:
            age=now-last[0]
            if age<.55:
                gain=1. if age<=.20 else 1.-(age-.20)/.35
                row.update(limit_degrees=35.+max(0,last[1]['limit_degrees']-35.)*gain,
                           preferred_degrees=12.+(last[1]['preferred_degrees']-12.)*gain,
                           kind='brief-evidence-loss-hold' if age<=.20 else 'evidence-loss-release',
                           upper_image_direction=last[1]['upper_image_direction'],confidence=last[1]['confidence']*gain)
        result.append(row)
    return result


def reachable_posture(evidence,shoulder,goal,body,upper,lower,contact_weight):
    """Adapt only the missing-elbow prior to character-relative contact reach.

    A source wrist at shoulder height can still require a raised model elbow:
    human/anime neck, palm and forearm proportions are different. Compute the
    least raised two-link solution instead of weakening all posture penalties.
    Measured shoulder/elbow directions remain authoritative.
    """
    from mathutils import Vector
    if contact_weight<=0 or evidence.get('upper_image_direction') is not None:
        return evidence
    axis=np.asarray(goal,float)-np.asarray(shoulder,float)
    distance=float(np.linalg.norm(axis))
    if distance<1e-8 or min(upper,lower)<=0:return evidence
    axis/=distance
    down=np.asarray(body@Vector((0,0,-1)))
    toward=math.acos(float(np.clip(axis@down,-1.,1.)))
    cone=math.acos(float(np.clip((distance*distance+upper*upper-lower*lower)/
                                (2*distance*upper),-1.,1.)))
    minimum=math.degrees(max(0.,toward-cone))
    weight=float(np.clip(contact_weight/.85,0.,1.))
    adjusted=dict(evidence)
    adjusted['preferred_degrees']+=weight*max(0.,minimum-evidence['preferred_degrees'])
    adjusted['limit_degrees']+=weight*max(0.,minimum+15.-evidence['limit_degrees'])
    adjusted['contact_reach_minimum_degrees']=minimum
    return adjusted


def posture_cost(elbows,shoulder,body,upper,camera,evidence,mirror=False):
    from mathutils import Vector
    from sam_ellen_screen import project
    from sam_ellen_screen_fit import projected
    down=np.asarray(body@Vector((0,0,-1)))
    angle=np.arccos(np.clip((elbows-shoulder)@down/upper,-1,1))
    preferred=math.radians(evidence['preferred_degrees']);limit=math.radians(evidence['limit_degrees'])
    # A low elbow is a proper prior, not a competitor with an arbitrary crop.
    cost=35*np.maximum(0,angle-preferred)**2+2500*np.maximum(0,angle-limit)**2
    direction=evidence['upper_image_direction']
    if direction is not None and evidence['confidence']>0:
        ep=np.c_[elbows,np.ones(len(elbows))]@np.asarray(camera[0]).T
        actual=projected(ep[:,:3],camera)-np.asarray(project(Vector(shoulder),camera))
        actual/=np.maximum(1e-8,np.linalg.norm(actual,axis=1))[:,None]
        expected=np.array(direction);expected[0]*=-1 if mirror else 1
        cost+=60*evidence['confidence']*np.arccos(np.clip(actual@expected,-1,1))**2
    return cost

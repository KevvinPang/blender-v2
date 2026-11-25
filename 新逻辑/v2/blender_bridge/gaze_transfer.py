"""Transfer observed gaze relative to the source lens to one shared target.

The fixed reference frame is explicitly labelled by the user as lens contact.
It calibrates the estimator bias once; sustained glances are never recentered.
"""
import numpy as np
from mathutils import Vector,Quaternion


def angles(direction):
    d=np.asarray(direction,dtype=float)
    return np.array([np.arctan2(d[0],d[2]),np.arctan2(d[1],np.hypot(d[0],d[2]))])


def direction_from_angles(value):
    yaw,pitch=value
    return np.array([np.sin(yaw)*np.cos(pitch),np.sin(pitch),np.cos(yaw)*np.cos(pitch)])


def calibrated_offsets(directions,lens_rays,valid,reference):
    if reference<0 or reference>=len(valid) or not valid[reference]:
        raise ValueError('The labelled lens-contact frame has no reliable eye observation')
    relative=np.full((len(valid),2),np.nan)
    for t in np.flatnonzero(valid):
        delta=angles(directions[t])-angles(lens_rays[t])
        relative[t]=np.arctan2(np.sin(delta),np.cos(delta))
    bias=relative[reference].copy()
    # A constant side glance after calibration stays a constant side glance.
    result=relative-bias
    result=np.arctan2(np.sin(result),np.cos(result))
    return result,bias


def source_track(face,K,width,height,reference):
    T=len(face['valid']);valid=np.zeros(T,bool);directions=np.zeros((T,3));rays=np.zeros((T,3))
    for t in np.flatnonzero(face['valid']):
        landmarks=face['landmarks'][t];shapes=face['shapes'][t];eyes=[];centres=[]
        # MediaPipe iris indices: 468-472 are image-left eye, 473-477
        # image-right eye. Use the measured iris centre and eyelid frame;
        # blendshape eyeLook values are a prior, not the actual gaze.
        for side,(suffix,iris_ids,inner,outer,up,down) in enumerate([
                ('Left',range(468,473),33,133,159,145),
                ('Right',range(473,478),362,263,386,374)]):
            iris=np.nanmean(landmarks[list(iris_ids),:2],axis=0)
            uv=np.array([iris,landmarks[inner,:2],landmarks[outer,:2],landmarks[up,:2],landmarks[down,:2]])*[width,height]
            span=np.linalg.norm(uv[2]-uv[1])
            if not np.isfinite(uv).all() or span<10 or shapes.get('eyeBlink'+suffix,1)>.65:continue
            fraction=np.dot(uv[0]-uv[1],uv[2]-uv[1])/max(span*span,1e-6)
            vertical=np.dot(uv[0]-uv[3],uv[4]-uv[3])/max(np.linalg.norm(uv[4]-uv[3])**2,1e-6)
            if not -.1<fraction<1.1 or not -.2<vertical<1.2:continue
            # Relative to the labelled lens-contact frame, 0.5 is neutral.
            # The gains are deliberately bounded for anime eye travel.
            yaw=np.radians(32)*(fraction-.5)*(1 if side==0 else -1)
            pitch=np.radians(26)*(vertical-.5)
            eyes.append(np.asarray(face['rotations'][t] @ Vector(direction_from_angles([yaw,pitch]).tolist())))
            centres.append((uv[1]+uv[2])/2)
        if not eyes:continue
        # A single shared gaze prevents separate noisy detections producing
        # divergent left/right eyes. Obvious binocular disagreement is unknown.
        if len(eyes)==2 and np.dot(eyes[0],eyes[1])<np.cos(np.radians(12)):continue
        direction=np.mean(eyes,axis=0);direction/=np.linalg.norm(direction)
        centre=np.mean(centres,axis=0)
        # OpenGL camera coordinates: eyes to lens is +Z, with the off-axis
        # source eye centre included. Principal-axis forward is not lens contact.
        lens=np.array([-(centre[0]-K[t,0,2])/K[t,0,0],(centre[1]-K[t,1,2])/K[t,1,1],1.])
        directions[t]=direction;rays[t]=lens/np.linalg.norm(lens);valid[t]=True
    offsets,bias=calibrated_offsets(directions,rays,valid,reference)
    reliability=valid.astype(float)
    # Brief blinks keep the last measured direction, then fade to neutral.
    # This fallback is flagged and is never classified as observed eye contact.
    last=None;gap=0
    for t in range(T):
        if valid[t]:last=offsets[t].copy();gap=0
        else:
            gap+=1;offsets[t]=last if last is not None else np.zeros(2)
            reliability[t]=max(0.,1.-max(0,gap-6)/6) if last is not None else 0.
    return offsets,reliability,{'observed_eye_frames':int(valid.sum()),'reference_frame':reference+1,
        'calibration':'user-labelled lens contact; fixed single-frame bias',
        'calibration_bias_degrees':np.degrees(bias).tolist(),
        'observed_offset_range_degrees':[np.degrees(offsets[valid].min(0)).tolist(),np.degrees(offsets[valid].max(0)).tolist()],
        'estimator':'head rotation plus measured iris/eyelid geometry; angular gains approximate'}


def common_target(eye_centre,camera_position,camera_rotation,offset):
    eye_centre=np.asarray(eye_centre);camera_position=np.asarray(camera_position);R=np.asarray(camera_rotation)
    ray=camera_position-eye_centre;distance=np.linalg.norm(ray)
    if distance<1e-5:raise ValueError('Camera overlaps eye centre')
    local=R.T@(ray/distance)
    desired=direction_from_angles(angles(local)+offset)
    return eye_centre+distance*(R@desired)


def legacy_eye_rotation(head_delta,bone_rest,direction,limit_degrees=20.):
    """Same anatomical forward/rest-axis rule as sam_ellen_retarget.correct_gaze.

    Original legacy function uses the camera as target. V2 supplies a target
    inferred from the video instead. No iris projection or independent centering.
    """
    forward=head_delta @ Vector((0,-1,0))
    swing=forward.rotation_difference(direction.normalized())
    if swing.angle>np.radians(limit_degrees):
        swing=Quaternion((1,0,0,0)).slerp(swing,float(np.radians(limit_degrees)/swing.angle))
    return swing @ head_delta @ bone_rest,float(np.degrees(swing.angle))

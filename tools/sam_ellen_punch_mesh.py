"""Fit the character's skinned fist outline to video, with fixed bone lengths."""
import math
import numpy as np
from mathutils import Matrix, Quaternion, Vector


def hand_geometry(source, side):
    """Read only hand-weighted vertices; retain their actual skinning weights."""
    import bpy
    blocks = []
    for mesh in bpy.data.objects:
        if mesh.type != 'MESH' or not any(m.type == 'ARMATURE' and m.object == source for m in mesh.modifiers):
            continue
        names = {g.index: g.name for g in mesh.vertex_groups}
        hand = {i for i,n in names.items() if n.endswith('_'+side) and
                (n == 'Wrist_'+side or n.startswith(('Thumb', 'IndexFinger', 'MiddleFinger', 'RingFinger', 'LittleFinger')))}
        transform = source.matrix_world.inverted() @ mesh.matrix_world
        chosen = [v for v in mesh.data.vertices if sum(g.weight for g in v.groups if g.group in hand) > .98]
        if not chosen:
            continue
        points = np.array([tuple(transform @ v.co)+(1.,) for v in chosen])
        weights = {}
        for j,v in enumerate(chosen):
            for group in v.groups:
                name = names.get(group.group)
                if name in source.pose.bones and group.weight > 0:
                    weights.setdefault(name, np.zeros(len(chosen)))[j] = group.weight
        blocks.append((points, weights))
    if not blocks:
        raise ValueError('没有找到角色的手部蒙皮，无法校验拳头画面轮廓')
    return blocks


def posed_hand(blocks, rig, side):
    result = []
    for rest, weights in blocks:
        points = np.zeros((len(rest), 3)); total = np.zeros(len(rest))
        for name, weight in weights.items():
            bone = rig.pose.bones[name]
            skin = np.asarray(bone.matrix @ bone.bone.matrix_local.inverted())
            points += (rest @ skin.T)[:, :3]*weight[:, None]
            total += weight
        result.extend(points/np.maximum(1e-8, total[:, None]))
    wrist = rig.pose.bones['Wrist_'+side].matrix
    inverse = np.asarray(wrist.inverted())
    return (np.c_[result, np.ones(len(result))] @ inverse.T)[:, :3]


def hull(points):
    ordered = sorted(set(map(tuple, np.asarray(points, dtype=float))))
    if len(ordered) <= 2:
        return np.asarray(ordered)
    def cross(a,b,c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lower, upper = [], []
    for p in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1]+upper[:-1])


def clip_polygon(points, aspect):
    points = [np.array(p) for p in points]
    for axis, bound, sign in ((0,0.,1), (0,aspect,-1), (1,0.,1), (1,1.,-1)):
        output = []
        if not points:
            return np.empty((0,2))
        previous = points[-1]; inside = sign*(previous[axis]-bound) >= 0
        for current in points:
            good = sign*(current[axis]-bound) >= 0
            if good != inside:
                t = (bound-previous[axis])/(current[axis]-previous[axis])
                output.append(previous+(current-previous)*t)
            if good:
                output.append(current)
            previous, inside = current, good
        points = output
    return np.array(points)


ANGLES = np.arange(16)*math.tau/16
DIRECTIONS = np.c_[np.cos(ANGLES), np.sin(ANGLES)]


def supports(points):
    return (np.asarray(points) @ DIRECTIONS.T).max(0)


def project_mesh(points, camera):
    from sam_ellen_screen_fit import projected
    return projected(points, camera)


def image_shape(shape, camera, source_aspect, mirror):
    outline = np.asarray(shape['outline'], dtype=float).copy()
    outline[:,0] *= camera[2]/source_aspect
    if mirror:
        outline[:,0] = camera[2]-outline[:,0]
    return hull(outline)


def fit_outline(local, rotation, wrist, target, camera, previous=None, wrist_hint=None):
    """Fit visible silhouette support lines; cropped edges stay inequalities.

    Clipping the mesh hull before comparing means a cropped fist never becomes
    a smaller full fist. No hidden joints or finger motion are inferred here.
    """
    from sam_ellen_screen_fit import rotation_vector
    from sam_ellen_screen import unproject, project
    transform = camera[0]; inverse = transform.inverted()
    # A front punch has its knuckles nearer the camera than its wrist. The
    # mirrored-depth solution has a similar outline but hides the whole fist
    # inside/behind the forearm. Resolve that ambiguity before fitting.
    # Preserve the observed pose. Reflecting the hand through camera depth can
    # rotate it by nearly 180 degrees while leaving a similar fist silhouette.
    rot = transform.to_3x3() @ rotation.to_matrix()
    offsets = local @ np.asarray(rot).T
    target_support = supports(target)
    desired_span = np.ptp(target, axis=0)
    start = np.asarray(transform @ wrist)
    native = project_mesh(offsets+start, camera)
    scale = max(desired_span)/max(1e-7, max(np.ptp(native, axis=0)))
    depth = max(.09, -start[2]/max(.15, scale))
    center = (target.min(0)+target.max(0))*.5
    origin = transform @ unproject(Vector(center), depth, camera)
    origin -= Vector(offsets.mean(0))
    initial = np.r_[origin.x, origin.y, math.log(max(.08,-origin.z)), 0.,0.,0.]
    if previous is not None:
        last = np.asarray(transform @ previous[0])
        lastq = transform.to_quaternion() @ previous[1]
        delta = lastq @ (transform.to_quaternion() @ rotation).inverted()
        if delta.angle > math.pi:
            delta.negate()
        initial = np.r_[last[:2], math.log(max(.08,-last[2])), np.asarray(delta.axis)*delta.angle]
    # Use the image itself for lateral initialization even after a large move.
    state = initial.copy()
    clipped = target.min(0) < .003
    cropped = bool(clipped.any() or target[:,0].max() >= camera[2]-.003 or target[:,1].max() >= .997)
    def evaluate(value):
        translation = np.r_[value[:2], -np.exp(value[2])]
        oriented = offsets @ rotation_vector(value[3:]).T
        moved = oriented+translation
        image = project_mesh(moved, camera)
        polygon = clip_polygon(hull(image), camera[2])
        if len(polygon) < 3:
            # Recover an off-screen previous pose without a flat clipped cost.
            actual = supports(image)
        else:
            actual = supports(polygon)
        residual = actual-target_support
        # A weak pose prior resolves the monocular rotation/depth ambiguity.
        regular = np.r_[(value[3:]-initial[3:])*.18,value[3:]*.12,
                        (value[2]-initial[2])*(.003 if cropped else .0005)]
        # A silhouette alone can roll the fist onto its side. Retain the last
        # observed palm-to-wrist direction while joints are lost near the lens.
        direction_residual = []
        if wrist_hint is not None:
            direction = project_mesh(translation[None,:],camera)[0]-image.mean(0)
            direction /= max(1e-6,np.linalg.norm(direction))
            direction_residual = (direction-wrist_hint)*.08
        return np.r_[residual, regular, max(0.,.008-oriented[:,2].mean())*.8,direction_residual]
    damping = 1e-4
    for _ in range(24):
        residual = evaluate(state)
        jacobian = np.empty((len(residual),6))
        for j in range(6):
            trial = state.copy(); trial[j] += .0005
            jacobian[:,j] = (evaluate(trial)-residual)/.0005
        try:
            step = np.linalg.solve(jacobian.T@jacobian+np.eye(6)*damping, -jacobian.T@residual)
        except np.linalg.LinAlgError:
            break
        step[:2] = np.clip(step[:2], -.15,.15)
        step[2] = np.clip(step[2], -.25,.25)
        step[3:] = np.clip(step[3:], -.18,.18)
        trial = state+step
        trial[2] = np.clip(trial[2], math.log(.08), math.log(8.))
        norm = np.linalg.norm(trial[3:])
        if norm > .45:
            trial[3:] *= .45/norm
        if np.dot(evaluate(trial),evaluate(trial)) < residual@residual:
            state = trial; damping = max(1e-7,damping*.5)
            if np.linalg.norm(step) < 1e-5:
                break
        else:
            damping *= 5
    position = inverse @ Vector((*state[:2], -math.exp(state[2])))
    delta = Matrix(rotation_vector(state[3:])).to_quaternion()
    aimed = inverse.to_quaternion() @ delta @ transform.to_quaternion() @ rotation
    return position, aimed.normalized(), float(np.linalg.norm(evaluate(state)[:16])/4)


def image_elbow_bend(arm, elbow, wrist, goal, desired, camera, previous=None):
    """Select the elbow circle from its image, including off-screen elbows."""
    from sam_ellen_screen import project
    vector=goal-arm.head;distance=vector.length;axis=vector.normalized()
    upper=(elbow.bone.head_local-arm.bone.head_local).length
    lower=(wrist.bone.head_local-elbow.bone.head_local).length
    along=(distance*distance+upper*upper-lower*lower)/(2*max(1e-6,distance))
    radius=math.sqrt(max(0.,upper*upper-along*along))
    center=arm.head+axis*along
    a=axis.orthogonal().normalized();b=axis.cross(a).normalized()
    if previous is not None:
        previous=previous-axis*previous.dot(axis)
        previous=previous.normalized() if previous.length>1e-6 else None
    fist_depth=-(camera[0]@goal).z
    def cost(angle):
        bend=a*math.cos(angle)+b*math.sin(angle)
        point=center+bend*radius
        delta=project(point,camera)-Vector(desired)
        depth=-(camera[0]@point).z
        value=delta.length_squared+max(0.,fist_depth+.03-depth)**2*4
        if previous is not None:
            value+=(1-max(-1.,min(1.,bend.dot(previous))))*.002
            # A nearly straight arm has two equivalent image branches. Limit
            # movement of its bend plane instead of filtering the solved bones.
            value+=max(0.,math.cos(.25)-bend.dot(previous))*100
        return value,bend
    angles=np.arange(96)*math.tau/96
    angle=min(angles,key=lambda value:cost(value)[0])
    step=math.tau/96
    for _ in range(7):
        angle=min((angle-step,angle,angle+step),key=lambda value:cost(value)[0]);step*=.5
    return cost(angle)[1]


def fist_pose_weights(times, segment, frames, side):
    """Keep the observed fist through retraction, separately from position fade.

    A short landmark dropout is not evidence that the fingers opened. Release
    the prior on a visible open hand, or shortly after the tracked punch ends.
    """
    from sam_ellen_punch import ease
    release=segment['end'];duration=.15
    # Perspective and occlusion make 2-D wrist/tip ratios unreliable at the
    # lens. A single false open-hand detection must not undo an established
    # closed fist while its verified punch silhouette is still being tracked.
    return [min(ease((time-segment['start'])/.10),ease((release+duration-time)/duration))
            for time in times]


def correct_video_punch(scratch, rig, source_rig, channels, data, options, cameras, report):
    from sam_ellen_punch import ease, sample_at, ray_reach
    from sam_ellen_retarget import bone_map, correct_contacts
    from sam_ellen_screen import project, unproject
    from sam_ellen_sequence import timeline_frame
    from sam_punch_evidence import detect
    from sam_ellen_arm_chain import direction_evidence,refine_goal,torso_front_plane
    from sam_ellen_arm_visibility import observed_frames
    side = options['side']; view = scratch.view_layers[0]
    originals = {name:list(samples) for name,samples in channels.items()}
    segments, observations = detect(data['screen'], side); options['segments'] = segments
    times = np.array([r['time']-data['frames'][0]['time'] for r in data['frames']])
    if not segments or options.get('strength',0)<=0:
        report.extend({'time':float(time),'active':False,'model':'video-fist-mesh-v3'} for time in times)
        return
    arm, elbow, wrist = [rig.pose.bones[n+'_'+side] for n in ('Arm','Elbow','Wrist')]
    fingers = [name for name in channels if name.endswith('_'+side) and name.startswith(
        ('Thumb','IndexFinger','MiddleFinger','RingFinger','LittleFinger'))]
    blocks = hand_geometry(source_rig, side)
    reach = ((arm.bone.head_local-elbow.bone.head_local).length+
             (elbow.bone.head_local-wrist.bone.head_local).length)*.995
    goals, metadata, geometry, rotations = [], [], {}, {}
    arm_evidence=direction_evidence(data,side,options)
    hand_visible=observed_frames(data,side)
    def pose(index):
        for name,samples in channels.items():
            rig.pose.bones[name].rotation_quaternion = samples[index][1]
        view.update()
    for s in segments:
        index = s['reference_index']; pose(index)
        geometry[index] = posed_hand(blocks, rig, side), wrist.matrix.to_quaternion().normalized()
        good = [i for i in range(index,len(times)) if times[i]<s['end']
                and observations[i] and observations[i].get('source')=='landmarks']
        # A direct landmark sample after a long gap must not retrospectively
        # replace the visible pre-punch orientation.
        previous_index=index
        for i in good:
            if times[i]-times[previous_index]>.18:break
            pose(i);rotations[i]=wrist.matrix.to_quaternion().normalized();previous_index=i
    # Finger evidence has its own lifetime: fading the positional correction
    # must not open a fist that is still visibly closed during the return.
    for s in segments:
        weights=fist_pose_weights(times,s,data['screen']['frames'],side)
        for i,weight in enumerate(weights):
            if weight<=0:continue
            weight*=options['strength']
            for name in fingers:
                f,q=channels[name][i]
                channels[name][i]=f,tuple(Quaternion(q).slerp(
                    Quaternion(originals[name][s['reference_index']][1]),weight))
    previous, last_segment, bend_previous = None, None, None
    for i,time in enumerate(times):
        segment = next((s for s in segments if s['start'] < time < s['end']),None)
        obs = observations[i]
        row = {'time':float(time),'active':False,'model':'video-fist-mesh-v3'}
        if segment is None or not obs or not obs.get('silhouette'):
            goals.append(None); metadata.append(row); previous=None; continue
        if segment is not last_segment:
            previous=None; bend_previous=None; last_segment=segment
        weight = min(ease((time-segment['start'])/.10),ease((segment['end']-time)/.22))*options['strength']
        pose(i); before=wrist.head.copy(); original_rotation=wrist.matrix.to_quaternion().normalized()
        reference = segment['reference_index']
        # Segmentation supplies no finger pose. Keep the observed closed fist,
        # rather than the body's falsely open fingers during hand-detector loss.
        local, rotation = geometry[reference]
        trusted=[j for j in rotations if reference<=j<=i]
        if trusted:rotation=rotations[max(trusted)]
        camera=cameras[i]; target=image_shape(obs['silhouette'],camera,data['screen']['aspect'],options.get('mirror'))
        seed = np.asarray(data['screen']['frames'][reference]['hands'][side]['points'])
        hint = seed[0]-seed[[5,9,13,17]].mean(0)
        hint[0] *= camera[2]/data['screen']['aspect']*(-1 if options.get('mirror') else 1)
        wrist_hint=hint/max(1e-6,np.linalg.norm(hint)) if np.linalg.norm(hint)>.01 else None
        requested,aimed,error=fit_outline(local,rotation,before,target,camera,previous,wrist_hint)
        offset = aimed @ Vector(local.mean(0))
        center = project(requested+offset,camera)
        depth = -(camera[0]@(requested+offset)).z
        requested,limited,possible=ray_reach(center,depth,offset,arm.head,reach,camera)
        previous=(requested.copy(),aimed.copy())
        goal=before.lerp(requested,weight); q=original_rotation.slerp(aimed,weight)
        direction=(goal-arm.head).normalized()
        if np.linalg.norm(hint)<.01:
            hint=np.array([0.,.1])
        # The captured IK branch can put the elbow in front of the fist, so a
        # numerically aligned hand is completely hidden by its own forearm.
        # The visible palm-to-wrist direction supplies the bend-side evidence.
        height=data.get('video_orientation',{}).get('decoded_height_width',[0])[0]
        image_points=data['frames'][i].get('image_points',[])
        desired=np.asarray(image_points[7 if side=='L' else 8][:2],dtype=float)/height if height and len(image_points)>8 else None
        if desired is not None and np.isfinite(desired).all() and -.5<desired[0]<1.5 and -.2<desired[1]<1.8:
            desired[0]*=camera[2]/data['screen']['aspect']
            if options.get('mirror'):desired[0]=camera[2]-desired[0]
        else:
            desired=np.asarray(center)+hint*3
        bend=elbow.head-arm.head
        bend-=direction*bend.dot(direction)
        bend=bend.normalized() if bend.length>1e-6 else direction.orthogonal()
        captured_bend=elbow.head-arm.head;captured_bend-=direction*captured_bend.dot(direction)
        # The complete chain below selects one anatomical branch. Do not blend
        # opposing elbow vectors across the event's fade weights.
        goals.append((goal,q,bend))
        row.update(active=True,weight=weight,source=obs['source'],partial=any(obs['silhouette']['clipped']),
                   limited=limited,ray_possible=possible,requested_depth=float(depth),
                   silhouette_fit_error_pixels=error*options.get('height',1920),target=list(goal),
                   target_bounds=[*target.min(0),*target.max(0)],fist_reference=reference)
        metadata.append(row)
    from sam_ellen_temporal import trajectory,quaternions
    enabled=np.array([g is not None for g in goals])
    edges=np.diff(np.r_[False,enabled,False].astype(int))
    for first,end in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        local_times=times[first:end]
        positions=trajectory(local_times,[goals[i][0] for i in range(first,end)],.045)
        orientations=quaternions(local_times,[goals[i][1] for i in range(first,end)],.06)
        for j,i in enumerate(range(first,end)):
            goals[i]=(Vector(positions[j]),orientations[j],goals[i][2])
    previous_arm=None;held_evidence=None
    def chain_goal(i):
        nonlocal previous_arm,held_evidence
        a,e,w=arm.head.copy(),elbow.head.copy(),wrist.head.copy()
        normal=(e-a).cross(w-e).normalized()
        original=(a,e,w,normal)
        limit=min(30,540*(times[i]-times[i-1] if i else 1/data['fps']))
        if goals[i] is None:
            held_evidence=None
            previous_arm=None
            return None
        else:
            goal,q,bend=goals[i]
            evidence=arm_evidence[i]
            if evidence is not None:held_evidence=evidence
            elif held_evidence is not None:evidence=(held_evidence[0],.35,0.)
        goal,bend,details=refine_goal(arm,elbow,wrist,goal,bend,cameras[i],evidence,previous_arm,depth_strength=5.,
            max_step_degrees=limit,front_plane=torso_front_plane(rig),hand_axis=q@Vector((0,1,0)))
        if details.get('elbow_position') is not None:
            previous_arm=tuple(np.array(v) for v in (arm.head,details['elbow_position'],goal,details['normal']))
        metadata[i].update(details,target=list(goal))
        return goal,q,bend
    yield from correct_contacts(scratch,rig,channels,data,{'side':side,'stable_ik':True,'anatomical':True},goal_provider=chain_goal)
    for i,row in enumerate(metadata):
        if row['active']:
            pose(i)
            local=posed_hand(blocks,rig,side)
            actual=(np.c_[local,np.ones(len(local))] @ np.asarray(wrist.matrix).T)[:,:3]
            cam=(np.c_[actual,np.ones(len(actual))] @ np.asarray(cameras[i][0]).T)[:,:3]
            polygon=clip_polygon(hull(project_mesh(cam,cameras[i])),cameras[i][2])
            if len(polygon)>=3:
                row['after_bounds']=[*polygon.min(0),*polygon.max(0)]
                row['outline_error_pixels']=float(np.mean(abs(supports(polygon)-supports(image_shape(
                    observations[i]['silhouette'],cameras[i],data['screen']['aspect'],options.get('mirror'))))))*options.get('height',1920)
            row['ik_error']=(wrist.head-Vector(row['target'])).length
            row['after_depth']=-(cameras[i][0]@wrist.head).z
        report.append(row)

"""Observed head pose and bounded eye/expression retargeting for Ellen.

Face rotation is camera-relative. Eye-look scores estimate rotation in the
head frame; they are not a command to lock the eyes onto the output camera.
"""
import numpy as np
import bpy
from mathutils import Matrix, Quaternion, Vector
from bpy_extras.object_utils import world_to_camera_view


def visual_iris_bias(scene, rig, meshes):
    """Calibrate the character's eye-bone axis to its visible iris plane.

    The MMD eye bone pivots are above the rendered iris centre. Without this
    one-time correction a mathematically correct camera ray puts the iris high
    in the eyelid opening, especially in a close crop.
    """
    cam=scene.camera;slots_by_mesh=[];centres={}
    for mesh in meshes:
        slots={i for i,m in enumerate(mesh.data.materials) if m and m.name.rsplit('|',1)[-1].strip().split('.')[0]=='目'}
        transform=rig.matrix_world.inverted()@mesh.matrix_world
        for side in ['L','R']:
            group=mesh.vertex_groups.get('Eye_'+side)
            if group is None:continue
            polys=[p for p in mesh.data.polygons if p.material_index in slots and
                   all(any(g.group==group.index and g.weight>.95 for g in mesh.data.vertices[i].groups) for i in p.vertices)]
            area=sum(p.area for p in polys)
            if area>1e-9:centres[side]=transform@(sum((p.center*p.area for p in polys),Vector())/area)
    def project(point):
        q=cam.matrix_world.inverted()@(rig.matrix_world@point)
        return np.array([q.x/-q.z,q.y/-q.z])
    values=[]
    for side,prefix in [('L','eyeLf'),('R','eyeRt')]:
        if side not in centres:continue
        up=project(rig.pose.bones[prefix+'03Joint'].head);lo=project(rig.pose.bones[prefix+'07Joint'].head)
        values.append((up+lo)/2-project(centres[side]))
    if not values:return np.zeros(2)
    # target offsets are camera angular coordinates; clamp pathological mesh
    # group assignments instead of creating a large eye correction.
    return np.clip(np.mean(values,axis=0),[-.12,-.12],[.12,.12])


def load_face(path, count):
    if path is None:
        return None
    data = np.load(path)
    if not np.array_equal(data['frame_indices'], np.arange(count)):
        raise ValueError('Face observations must align with every motion frame')
    valid = data['valid'].copy() & np.isfinite(data['matrices']).all(axis=(1, 2))
    rotations = [None] * count
    for i in np.flatnonzero(valid):
        u, _, vt = np.linalg.svd(data['matrices'][i, :3, :3])
        rotations[i] = Matrix((u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt).tolist()).to_quaternion()
    # Smooth only contiguous observations. Never average rotation matrices or
    # fill a long missing interval with a fabricated look-at-camera pose.
    filtered = list(rotations)
    for i in range(1, count-1):
        if valid[i-1:i+2].all():
            a, b, c = [q.copy() for q in rotations[i-1:i+2]]
            if a.dot(b) < 0: a.negate()
            if c.dot(b) < 0: c.negate()
            filtered[i] = a.slerp(c, .5).slerp(b, .5).normalized()
    previous = None
    for i in range(count):
        if not valid[i]:
            previous = None
            continue
        filtered[i] = filtered[i].copy()
        if previous is not None and filtered[i].dot(previous)<0: filtered[i].negate()
        previous = filtered[i]
    weights = np.zeros(count)
    for i in np.flatnonzero(valid):
        left = i
        while left > 0 and valid[left-1]: left -= 1
        right = i
        while right < count-1 and valid[right+1]: right += 1
        weights[i] = min(1., (i-left+1)/4 if left else 1.,
                         (right-i+1)/4 if right < count-1 else 1.)
    shapes = data['blendshapes'].copy()
    for i in range(1, count-1):
        if valid[i-1:i+2].all():
            shapes[i] = (data['blendshapes'][i-1]+2*data['blendshapes'][i]+data['blendshapes'][i+1])/4
    return dict(rotations=filtered, valid=valid, weights=weights,
                shapes=[dict(zip(data['blendshape_names'].tolist(), row)) for row in shapes],
                landmarks=data['landmarks'].copy() if 'landmarks' in data else None)


def head_target(face, frame, camera_cv_to_world, rig_world_rotation, rest_head, rig_to_face):
    """MediaPipe uses OpenGL camera axes and canonical +Z face forward."""
    if face is None or not face['valid'][frame]:
        return None
    target = (rig_world_rotation.T @ camera_cv_to_world @ np.diag([1., -1., -1.])
              @ np.asarray(face['rotations'][frame].to_matrix()) @ rig_to_face @ rest_head)
    return target


def correct_head_roll(scene, rig, face, frame):
    """Match the projected eye-line roll to the source video for this frame."""
    if face is None or face['landmarks'] is None or not face['valid'][frame]:
        return 0.0
    head = rig.pose.bones['Head']
    left = world_to_camera_view(scene, scene.camera,
                                rig.matrix_world @ rig.pose.bones['Eye_R'].head)
    right = world_to_camera_view(scene, scene.camera,
                                 rig.matrix_world @ rig.pose.bones['Eye_L'].head)
    current = np.arctan2(right.y-left.y, right.x-left.x)
    source = face['landmarks'][frame]
    # MediaPipe landmarks use top-left image coordinates; convert to the
    # bottom-left convention used by world_to_camera_view.
    a = source[33]
    b = source[263]
    desired = np.arctan2(a[1]-b[1], b[0]-a[0])
    delta = float(np.arctan2(np.sin(desired-current), np.cos(desired-current)))
    # Blender's camera forward axis is -Z; its positive roll direction is
    # opposite to the image-space y-down angle above.
    delta = -delta
    delta = float(np.clip(delta*1.5, -np.radians(25.0), np.radians(25.0)))
    delta = float(np.clip(delta, -np.radians(25.0), np.radians(25.0)))
    if abs(delta) < np.radians(.05):
        return float(np.degrees(delta))
    axis = scene.camera.matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0))
    axis.normalize()
    world = rig.matrix_world @ head.matrix
    pivot = world.translation.copy()
    correction = Matrix.Translation(pivot) @ Matrix.Rotation(delta, 4, axis) @ Matrix.Translation(-pivot)
    pose_matrix = rig.matrix_world.inverted() @ correction @ world
    parent = head.parent
    kwargs = {}
    if parent:
        kwargs = {'parent_matrix': parent.matrix, 'parent_matrix_local': parent.bone.matrix_local}
    q = head.bone.convert_local_to_pose(
        pose_matrix, head.bone.matrix_local, invert=True, **kwargs).to_quaternion().normalized()
    return q, float(np.degrees(delta))


def apply_face(scene, rig, face, key_quat, K, image_size, reference_frame, gaze_mode='video', head_mode='body', preserve_expressions=False):
    from gaze_transfer import source_track, common_target, legacy_eye_rotation
    if face is None:
        return {'mode': 'not_requested'}
    meshes = [o for o in scene.objects if o.type == 'MESH' and o.data.shape_keys
              and any(m.type == 'ARMATURE' and m.object == rig for m in o.modifiers)]
    # The legacy gaze feature never resets the character's facial styling.
    # Only the channels explicitly supplied by this video are replaced below.
    defaults={m.name:{k.name:k.value for k in m.data.shape_keys.key_blocks} for m in meshes}
    height,width=image_size
    if gaze_mode=='video':
        if reference_frame is None:raise ValueError('Video gaze needs a labelled lens-contact reference frame')
        offsets,reliability,tracking=source_track(face,K,width,height,reference_frame-1)
    else:
        offsets=np.zeros((len(face['valid']),2));reliability=np.ones(len(offsets))
        tracking={'calibration':'legacy always-camera reference, not video gaze'}
    previous={};rotations=[];aim_errors=[];targets=[];head_angles=[]
    scene.frame_set(1);bpy.context.view_layer.update()
    visual_bias=visual_iris_bias(scene,rig,meshes)
    for t in range(len(face['valid'])):
        scene.frame_set(t+1)
        head=rig.pose.bones['Head']
        # Head orientation is solved once by calibrated_bridge. Re-solving it
        # here toward the output camera erases real nods and shakes detected by
        # the face tracker, leaving only the eyes visibly moving.
        if head_mode == 'face':
            roll_result = correct_head_roll(scene, rig, face, t)
            if isinstance(roll_result, tuple):
                q, roll_delta = roll_result
                key_quat(head, q, t+1, previous)
                head_angles.append(abs(roll_delta))
                bpy.context.view_layer.update()
        head_delta=head.matrix.to_quaternion() @ head.bone.matrix_local.to_quaternion().inverted()
        eye_centre=sum((rig.matrix_world @ rig.pose.bones[n].head for n in ['Eye_L','Eye_R']),Vector())*.5
        target=common_target(eye_centre,scene.camera.matrix_world.translation,
                             scene.camera.matrix_world.to_3x3(),offsets[t]+visual_bias)
        targets.append(target.tolist())
        target_rig=rig.matrix_world.inverted() @ Vector(target.tolist())
        for side in ['L','R']:
            bone=rig.pose.bones['Eye_'+side]
            direction=target_rig-bone.head
            desired,angle=legacy_eye_rotation(head_delta,bone.bone.matrix_local.to_quaternion(),direction)
            neutral=head_delta @ bone.bone.matrix_local.to_quaternion()
            if neutral.dot(desired)<0:desired.negate()
            desired=neutral.slerp(desired,float(reliability[t]))
            # Use Blender's bone transform conversion, as in the legacy driver,
            # so rest roll / inherited transforms are not replaced by screen axes.
            matrix=desired.to_matrix().to_4x4();matrix.translation=bone.head
            parent=bone.parent
            q=bone.bone.convert_local_to_pose(matrix,bone.bone.matrix_local,invert=True,
                parent_matrix=parent.matrix,parent_matrix_local=parent.bone.matrix_local).to_quaternion().normalized()
            key_quat(bone,q,t+1,previous)
            rotations.append(angle)
            achieved=desired @ bone.bone.matrix_local.to_quaternion().inverted() @ Vector((0,-1,0))
            aim_errors.append(float(np.degrees(achieved.angle(direction))))
        valid=face['valid'][t];shapes=face['shapes'][t] if valid else {};weight=face['weights'][t]
        for mesh in ([] if preserve_expressions else meshes):
            keys=mesh.data.shape_keys.key_blocks;base=defaults[mesh.name]
            values={}
            for name,suffix in [('ウィンク２','Left'),('ウィンク２右','Right')]:
                blink=float(np.clip((shapes.get('eyeBlink'+suffix,0)-.12)/.88,0,1))*weight
                values[name]=base.get(name,0)+(1-base.get(name,0))*blink
            values['あ']=float(np.clip(shapes.get('jawOpen',0)*1.2,0,.8))*weight
            for name,value in values.items():
                if name in keys:
                    keys[name].value=value;keys[name].keyframe_insert(data_path='value',frame=t+1)
    return {'mode':gaze_mode,
            'head_source':('camera-facing +Z head with body pose as smoothed prior'
                           if head_mode=='face' else 'FlowHMR body head rotation preserved'),
            'head_mode':head_mode,
            'eye_solver':'legacy anatomical eye rotation; one shared 3D target',
            'facial_style':'preserved input styling; observed blinks and jaw opening',
            'observed_face_frames':int(face['valid'].sum()),
            'head_camera_correction_max_deg':max(head_angles) if head_angles else 0.,
            'eye_rotation_max_deg':max(rotations),'eye_target_error_max_deg':max(aim_errors),
            'source_gaze':tracking,'visual_iris_bias_camera':visual_bias.tolist(),'target_world':targets,
            'limitations':['gaze angles are estimates, not calibrated clinical eye tracking',
                           '20 degree legacy eye limit can constrain extreme glances']}

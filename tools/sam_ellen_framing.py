"""Smooth image framing for fixed-root upper-body takes; no world-motion inference."""
import numpy as np
from mathutils import Matrix, Vector


def track_framing(scene, target_rig, scratch, rig, channels, data, options, camera, cameras, report):
    # Use the 0.35 eye-anchor / relative-face-distance policy for every take.
    # Fitting the character's jaw and shoulder proportions changed the crop and
    # optics in 0.36. Hand refinement remains independent and runs afterwards.
    from sam_ellen_camera_fit import reference_indices, framing_driver_world
    import bpy
    render_camera = bpy.data.objects.get(options.get('camera', '')) or camera
    from sam_ellen_frame_body import relative_face_depth
    from sam_ellen_retarget import gaze_bones
    from sam_ellen_screen import project, unproject
    from sam_ellen_sequence import timeline_frame, curves
    from sam_ellen_temporal import trajectory,bridge_missing
    times = np.array([r['time'] for r in data['frames']])
    faces = [r.get('face') for r in data['screen']['frames']]
    reference = reference_indices(data)
    ratios, reliable = relative_face_depth(times, faces, reference)
    eyes = [gaze_bones(rig)[label] for label in ('左目', '右目')]
    centers = []
    for i in range(len(times)):
        for name, samples in channels.items():
            rig.pose.bones[name].rotation_quaternion = samples[i][1]
        scratch.view_layers[0].update()
        centers.append((eyes[0].head+eyes[1].head)*.5)
        yield i+1, 2*len(times)
    anchor_depth = float(np.median([-(cameras[i][0]@centers[i]).z for i in reference]))
    offsets, weights, targets = [], [], []
    for i, (face, cam, center) in enumerate(zip(faces, cameras, centers)):
        relative = times[i]-times[0]
        envelope = np.clip((relative-options.get('start', 0))/.15, 0, 1)
        if options.get('end', 0)>0:
            envelope *= np.clip((options['end']-relative)/.15, 0, 1)
        strength = float(envelope*options['strength']) if face else 0.
        before = project(center, cam)
        target = Vector(face['center']) if face else before.copy()
        if face:
            target.x *= cam[2]/data['screen']['aspect']
            if options.get('mirror'): target.x = cam[2]-target.x
        depth = -(cam[0]@center).z
        desired_depth = anchor_depth/ratios[i] if reliable[i]>.1 else depth
        desired_depth = depth+(desired_depth-depth)*strength
        wanted = unproject(before.lerp(target, strength), desired_depth, cam)
        # Moving the camera by this vector is equivalent to moving the subject
        # oppositely, while keeping its roots, legs and captured proportions.
        offsets.append(np.array(center-wanted))
        weights.append(strength)
        targets.append(target)
    offsets=bridge_missing(times,offsets,np.array(weights)>0)
    offsets = trajectory(times, offsets, .055, np.maximum(.05, weights), robust=.025)
    from sam_ellen_framing_stability import stabilize_punch_framing
    offsets,punch_stable=stabilize_punch_framing(data,offsets)
    frame, subframe = scene.frame_current, scene.frame_subframe
    world_matrices = []
    try:
        for i, cam in enumerate(cameras):
            f = timeline_frame(data, data['frames'][i]['time'])
            scene.frame_set(int(f), subframe=f-int(f))
            world = target_rig.evaluated_get(__import__('bpy').context.evaluated_depsgraph_get()).matrix_world.copy()
            transform = cam[0].copy()
            relative=times[i]-times[0]
            envelope=float(np.clip((relative-options.get('start',0))/.15,0,1))
            if options.get('end',0)>0:envelope*=float(np.clip((options['end']-relative)/.15,0,1))
            offset = Vector(offsets[i])*envelope
            inverse = transform.inverted()
            inverse.translation += offset
            updated = (inverse.inverted(), *cam[1:])
            cameras[i] = updated
            world_matrices.append((f, framing_driver_world(scene, camera, render_camera, world@inverse)))
            report.append({'time':float(times[i]), 'active':bool(envelope>0),
                'observed':bool(faces[i]),'measurement_held':bool(not faces[i] and offset.length>1e-8),
                'before_center_px':float((project(centers[i], cam)-targets[i]).length*options.get('height',1920)),
                'after_center_px':float((project(centers[i], updated)-targets[i]).length*options.get('height',1920)),
                'offset_rig':list(offset), 'relative_face_scale':float(ratios[i]),
                'face_interpolated':bool(faces[i] is None and offset.length>1e-8),
                'render_camera':render_camera.name,'framing_driver':camera.name})
            yield len(times)+i+1, 2*len(times)
        # Copy actions before inserting keys: append must preserve preceding shots.
        if camera.animation_data and camera.animation_data.action:
            camera.animation_data.action = camera.animation_data.action.copy()
        camera.rotation_mode = 'QUATERNION'
        previous = None
        for f, world in world_matrices:
            camera.matrix_world = world
            if previous and camera.rotation_quaternion.dot(previous)<0: camera.rotation_quaternion.negate()
            previous = camera.rotation_quaternion.copy()
            for prop in ('location','rotation_quaternion','scale'): camera.keyframe_insert(prop,frame=f)
        for curve in curves(camera.animation_data.action,camera.animation_data.action_slot):
            for point in curve.keyframe_points: point.interpolation='LINEAR'
        camera['sam_framing_model']='face-trajectory-v1 / fixed roots / camera translation'
    finally:
        scene.frame_set(frame,subframe=subframe)

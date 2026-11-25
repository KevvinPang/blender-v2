"""Framing correction that retains evaluated source-camera motion and optics."""
import json
import math

import bpy
import numpy as np
from mathutils import Matrix, Vector


def bind_render_camera(scene, driver):
    """Use the retained native view throughout solving, not only after commit.

    Keep its local animation untouched. The calibrated parent owns generated
    framing; a failed bake restores the previous presentation binding.
    """
    import ellen_camera_distance as nav
    view = nav.adjustment_camera(scene)
    state = {'camera': scene.camera, 'view': view,
             'enabled': scene.get(nav.ENABLED), 'stored_view': scene.get(nav.VIEW)}
    if view:
        state.update(parent=view.parent, inverse=view.matrix_parent_inverse.copy(),
                     data=view.data, source=view.get(nav.SOURCE),
                     lock_parent=view.use_camera_lock_parent)
    return nav.ensure_camera(scene, driver), state


def restore_render_camera(scene, state):
    import ellen_camera_distance as nav
    view = nav.adjustment_camera(scene)
    if state['view']:
        view = state['view']
        view.parent = state['parent']
        view.matrix_parent_inverse = state['inverse']
        view.data = state['data']
        view.use_camera_lock_parent = state['lock_parent']
        if state['source'] is None:
            if nav.SOURCE in view: del view[nav.SOURCE]
        else: view[nav.SOURCE] = state['source']
    elif view:
        bpy.data.objects.remove(view, do_unlink=True)
    for key, value in ((nav.ENABLED, state['enabled']), (nav.VIEW, state['stored_view'])):
        if value is None:
            if key in scene: del scene[key]
        else: scene[key] = value
    scene.camera = state['camera']
    bpy.context.view_layer.update()


def framing_driver_world(scene, driver, render_camera, desired_world):
    """Write generated framing to the parent, preserving authored child keys."""
    if render_camera == driver:
        return desired_world
    if render_camera.parent != driver:
        raise ValueError('画面对齐与最终摄像机的父子关系已改变，请重新生成')
    graph = bpy.context.evaluated_depsgraph_get()
    evaluated = render_camera.evaluated_get(graph)
    if render_camera.constraints:
        raise ValueError('视图调整摄像机含额外约束，请先烘焙约束再进行画面对齐')
    local = evaluated.matrix_parent_inverse @ evaluated.matrix_basis
    return desired_world @ local.inverted()


def sync_camera_views(scene):
    """Refresh stored viewport cameras after a committed scene-camera change."""
    from ellen_camera_distance import sync_camera
    camera = sync_camera(scene)
    if camera is None:
        return 0
    changed = 0
    for window in bpy.context.window_manager.windows:
        if window.scene != scene:
            continue
        for area in window.screen.areas:
            # A console/text editor can retain an inactive VIEW_3D space.
            for space in area.spaces:
                if space.type == 'VIEW_3D' and not space.use_local_camera and space.camera != camera:
                    space.camera = camera
                    changed += 1
                    area.tag_redraw()
    return changed


def reference_indices(data):
    faces = data['screen']['frames']
    times = np.array([r['time'] for r in data['frames']])
    candidates = []
    for i, row in enumerate(faces):
        if times[i]-times[0] > 1.5:
            break
        indices = [j for j in range(i, len(times)) if times[i] <= times[j] <= times[i]+.45 and faces[j].get('face')]
        if len(indices) < 5 or times[indices[-1]]-times[indices[0]] < .25:
            continue
        centers = np.array([faces[j]['face']['center'] for j in indices])
        widths = np.array([faces[j]['face']['width'] for j in indices])
        if not np.isfinite(centers).all() or not np.isfinite(widths).all() or min(widths) < .005:
            continue
        score = float(np.std(centers, axis=0).sum()/np.median(widths)+np.std(np.log(widths)))
        candidates.append((score+.04*(times[i]-times[0]), indices))
    if not candidates:
        raise ValueError('视频开头没有足够稳定、可见的脸部，无法校准镜头')
    return min(candidates, key=lambda item:item[0])[1]


def camera_source(camera):
    """Resolve calibration ancestry without treating another shot as the same camera."""
    seen = set()
    while camera and camera.name not in seen:
        seen.add(camera.name)
        from ellen_camera_distance import source_camera
        unwrapped = source_camera(camera)
        if unwrapped != camera:
            camera = unwrapped
            continue
        parent = bpy.data.objects.get(camera.get('sam_original_camera', ''))
        if parent is None:
            break
        camera = parent
    return camera


def append_input_camera(camera, start):
    """Recover this clip's input, retaining corrections from earlier clips."""
    if camera is None:return camera
    from ellen_camera_distance import source_camera
    camera = source_camera(camera)
    source = bpy.data.objects.get(camera.get('sam_sequence_source_camera',''))
    if source and abs(camera.get('sam_sequence_start',-1)-start)<1e-4:
        return source
    # Migrate legacy generated sequence cameras, which retained their input
    # under this exact name but did not yet store explicit provenance.
    if camera.get('sam_framing_model') and camera.name.startswith('SAM Sequence - '):
        source=bpy.data.objects.get(camera.name[len('SAM Sequence - '):])
        if source and source.get('sam_framing_model'):
            return source
    return camera


def calibrated_camera(scene, target_rig, scratch, rig, channels, data, options):
    from sam_ellen_retarget import gaze_bones
    from sam_ellen_frame_body import rotation
    from sam_ellen_screen_fit import projected
    camera = bpy.data.objects.get(options.get('camera','')) or scene.camera
    from ellen_camera_distance import source_camera
    camera = source_camera(camera)
    if camera is None or camera.type != 'CAMERA' or camera.data.type != 'PERSP':
        raise ValueError('基础构图校准目前需要透视摄像机')
    # Appending must not change the framing of previously imported clips.
    if data.get('_sam_timeline'):
        if camera.get('sam_framing_model') and camera.parent is None and not camera.constraints:
            clone=camera.copy()
            clone.data=camera.data.copy()
            if camera.animation_data and camera.animation_data.action:
                clone.animation_data.action=camera.animation_data.action.copy()
            clone.name='SAM Sequence - '+camera.name
            clone['sam_sequence_source_camera']=camera.name
            clone['sam_sequence_start']=data['_sam_timeline']['start']
            scene.collection.objects.link(clone)
            return clone, {'model':'framing-append-v1','prefix_preserved':True,
                'camera':clone.name,'reused_for_append':False}
        return camera, {'model':'static-framing-v1','reused_for_append':True,'camera':camera.name}
    source = camera_source(camera)
    fixed=options.get('camera_mode')=='FIXED'
    if source.type != 'CAMERA' or source.data.type != 'PERSP':
        raise ValueError('原始摄像机已失效，请重新选择摄像机')
    indices = reference_indices(data)
    bones = gaze_bones(rig)
    eye_bones = [bones['左目'],bones['右目']]
    shoulders = [rig.pose.bones.get('Arm_'+side) for side in ('L','R')]
    if not all(shoulders):
        raise ValueError('人物缺少肩膀骨骼，无法校准构图')
    frame, subframe = scene.frame_current,scene.frame_subframe
    physics = getattr(scene,'wiggle_enable',None)
    ancestry, ancestor = [], source
    while ancestor is not None:
        ancestry.append(ancestor)
        ancestor = ancestor.parent
    animated = any(obj.animation_data or obj.constraints for obj in ancestry) or bool(source.data.animation_data)
    motion = []
    optical_properties = ('lens', 'shift_x', 'shift_y', 'sensor_width', 'sensor_height',
                          'clip_start', 'clip_end', 'ortho_scale')
    reference_optics = None
    try:
        if physics is not None:scene.wiggle_enable=False
        from sam_ellen_sequence import timeline_frame
        sample_frame=timeline_frame(data,data['frames'][indices[len(indices)//2]]['time'])
        scene.frame_set(int(sample_frame),subframe=sample_frame-int(sample_frame))
        graph=bpy.context.evaluated_depsgraph_get()
        evaluated = source.evaluated_get(graph)
        baseline=evaluated.matrix_world.copy()
        reference_optics = {p: getattr(evaluated.data, p) for p in optical_properties}
        reference_sensor_fit = evaluated.data.sensor_fit
        rig_world=target_rig.evaluated_get(graph).matrix_world.copy()
        render=scene.render
        aspect=render.resolution_x*render.pixel_aspect_x/(render.resolution_y*render.pixel_aspect_y)
        projection=source.evaluated_get(graph).calc_matrix_camera(graph,x=render.resolution_x,y=render.resolution_y,
            scale_x=render.pixel_aspect_x,scale_y=render.pixel_aspect_y)
        if animated:
            for row in data['frames']:
                f = timeline_frame(data, row['time'])
                scene.frame_set(int(f), subframe=f-int(f))
                evaluated = source.evaluated_get(bpy.context.evaluated_depsgraph_get())
                motion.append((f, evaluated.matrix_world.copy(),
                               {p: getattr(evaluated.data, p) for p in optical_properties}))
    finally:
        scene.frame_set(frame,subframe=subframe)
        if physics is not None:scene.wiggle_enable=physics
    from sam_ellen_image_geometry import head_geometry,posed_geometry
    geometry=head_geometry(rig) if fixed else None
    samples=[]
    for i in indices:
        for name,values in channels.items():
            rig.pose.bones[name].rotation_quaternion=values[i][1]
        scratch.view_layers[0].update()
        points_here=[bone.head for bone in [*eye_bones,*shoulders]]
        if geometry:points_here.extend(posed_geometry(rig,geometry)[1:])
        samples.append([list(rig_world@p) for p in points_here])
    points=np.median(samples,axis=0)
    observations=[data['screen']['frames'][i]['face'] for i in indices]
    target=np.median([f['center'] for f in observations],axis=0)
    target[0]*=aspect/data['screen']['aspect']
    width=float(np.median([f['width'] for f in observations]))*aspect/data['screen']['aspect']
    if options.get('mirror'):target[0]=aspect-target[0]
    source_height=data.get('video_orientation',{}).get('decoded_height_width',[0])[0]
    targets=[]
    if source_height:
        for side,index in enumerate((5,6)):
            visible=[]
            for i in indices:
                row=data['frames'][i]
                if not row.get('image_points'):continue
                p=np.array(row['image_points'][index][:2],dtype=float)/source_height
                if np.isfinite(p).all() and 0<=p[0]<=data['screen']['aspect'] and 0<=p[1]<=1:
                    p[0]*=aspect/data['screen']['aspect']
                    if options.get('mirror'):p[0]=aspect-p[0]
                    visible.append(p)
            if len(visible)>=len(indices)*.6:
                targets.append((side,np.median(visible,axis=0)))
    face_widths=[np.ptp(np.asarray(f['jaw'])[:,0]) for f in observations if f.get('jaw')]
    use_surface=bool(geometry and face_widths)
    fit_width=float(np.median(face_widths))*aspect/data['screen']['aspect'] if use_surface else width
    fit_ids=[4,5] if use_surface else [0,1]
    local=np.c_[points,np.ones(len(points))]@np.asarray(baseline.inverted()).T
    depth=-float(local[:2,2].mean())
    if depth<.05:raise ValueError('人物在摄像机后方或距离过近，请先调整基础机位')
    cam=(None,projection,aspect,'PERSP')
    def transform(value):
        delta=np.eye(4)
        delta[:3,:3]=rotation(np.array([value[3],value[4],0.]))
        delta[:3,3]=value[:3]*depth
        return np.asarray(baseline)@delta
    def image(value):
        matrix=np.linalg.inv(transform(value))
        local=np.c_[points,np.ones(len(points))]@matrix.T
        if np.any(local[:,2]>-.03):return None
        return projected(local[:,:3],cam)
    def residual(value):
        p=image(value)
        if p is None:return np.full(3+2*len(targets),100.)
        terms=[*(3*(p[:2].mean(0)-target)),2*(np.linalg.norm(p[fit_ids[0]]-p[fit_ids[1]])-fit_width)]
        for side,point in targets:terms.extend((.25 if fixed else .55)*(p[side+2]-point))
        return np.array(terms)
    before=image(np.zeros(5))
    ratio=float(np.clip(np.linalg.norm(before[fit_ids[0]]-before[fit_ids[1]])/fit_width,.08 if fixed else .35,4. if fixed else 2.))
    value=np.array([0.,0.,ratio-1.,0.,0.])
    # Solve lateral framing analytically before the bounded orientation refinement.
    ray=np.asarray(projection.inverted())@np.array([2*target[0]/aspect-1,1-2*target[1],-1.,1.])
    ray=ray[:3]/ray[3]
    desired=ray*(-depth*ratio/ray[2])
    value[:2]=(local[:2,:2].mean(0)-desired[:2])/depth
    regular=np.array([.0001,.0001,.0001,.02,.02])
    def constrain(v):
        v[:2]=np.clip(v[:2],-.6,.6)
        v[2]=np.clip(v[2],-.92 if fixed else -.65,3. if fixed else 1.)
        v[3:]=np.clip(v[3:],-math.radians(10),math.radians(10))
        return v
    value=constrain(value)
    def cost(v):return np.dot(residual(v),residual(v))+np.dot(regular*v,v)
    for _ in range(30):
        r=residual(value)
        columns=[]
        for j in range(5):
            trial=value.copy();trial[j]+=1e-5
            columns.append((residual(trial)-r)/1e-5)
        jac=np.stack(columns,axis=1)
        step=np.linalg.solve(jac.T@jac+np.diag(regular),-jac.T@r-regular*value)
        step*=min(1.,.15/max(1e-10,np.linalg.norm(step)))
        improved=False
        for factor in (1.,.5,.25,.1):
            trial=constrain(value+step*factor)
            if cost(trial)<cost(value):value,improved=trial,True;break
        if not improved or np.linalg.norm(step)<1e-6:break
    after=image(value)
    if after is None or not np.isfinite(after).all() or np.linalg.norm(residual(value))>=np.linalg.norm(residual(np.zeros(5))):
        raise ValueError('镜头校准未改善基础构图，已保留原机位')
    clone=source.copy()
    clone.data=source.data.copy()
    clone.name='SAM Calibrated - '+source.name
    clone.animation_data_clear()
    clone.data.animation_data_clear()
    for prop, optical_value in reference_optics.items():
        setattr(clone.data, prop, optical_value)
    clone.data.sensor_fit = reference_sensor_fit
    clone.parent=None
    for constraint in list(clone.constraints):clone.constraints.remove(constraint)
    scene.collection.objects.link(clone)
    clone.matrix_world=Matrix(transform(value).tolist())
    if animated and not fixed:
        correction = baseline.inverted()@clone.matrix_world
        clone.rotation_mode = 'QUATERNION'
        previous = None
        for f, matrix, optics in motion:
            clone.matrix_world = matrix@correction
            if previous is not None and clone.rotation_quaternion.dot(previous) < 0:
                clone.rotation_quaternion.negate()
            previous = clone.rotation_quaternion.copy()
            for prop in ('location', 'rotation_quaternion', 'scale'):
                clone.keyframe_insert(prop, frame=f)
            for prop, optical_value in optics.items():
                setattr(clone.data, prop, optical_value)
                clone.data.keyframe_insert(prop, frame=f)
        for owner in (clone, clone.data):
            if owner.animation_data and owner.animation_data.action:
                from sam_ellen_sequence import curves
                for curve in curves(owner.animation_data.action, owner.animation_data.action_slot):
                    for point in curve.keyframe_points:
                        point.interpolation = 'LINEAR'
        scene.frame_set(frame, subframe=subframe)
    clone['sam_original_camera']=source.name
    if fixed:clone['sam_framing_model']='fixed-camera-subject-v2'
    h=render.resolution_y
    report={'model':'evaluated-camera-offset-v2','camera':clone.name,'original_camera':source.name,
        'source_animation_preserved':bool(animated and not fixed), 'baked_camera_samples':len(motion) if not fixed else 0,
        'scale_anchor':'face-skin-width' if use_surface else 'eye-distance',
        'target_anchor_width':fit_width,'fixed_camera':fixed,
        'reference_seconds':[data['frames'][indices[0]]['time'],data['frames'][indices[-1]]['time']],
        'reference_indices':indices,'target_center':target.tolist(),'target_eye_width':width,
        'lens_mm':reference_optics['lens'],
        'focal_source':'evaluated original camera; not recovered phone intrinsics',
        'before_center_px':float(np.linalg.norm(before[:2].mean(0)-target)*h),
        'after_center_px':float(np.linalg.norm(after[:2].mean(0)-target)*h),
        'before_width_px':float(abs(np.linalg.norm(before[0]-before[1])-width)*h),
        'after_width_px':float(abs(np.linalg.norm(after[0]-after[1])-width)*h),
        'before_shoulders_px':[float(np.linalg.norm(before[j+2]-p)*h) for j,p in targets],
        'after_shoulders_px':[float(np.linalg.norm(after[j+2]-p)*h) for j,p in targets],
        'before_eye_distance_m':float(np.linalg.norm(np.array(baseline.translation)-points[:2].mean(0))),
        'after_eye_distance_m':float(np.linalg.norm(np.array(clone.matrix_world.translation)-points[:2].mean(0))),
        'original_matrix':[list(r) for r in baseline],'matrix':[list(r) for r in clone.matrix_world]}
    from sam_ellen_shot import configure
    report['shot']=configure(scene,clone,target,options)
    report['lens_mm']=clone.data.lens
    clone['sam_calibration_report']=json.dumps(report)
    return clone,report

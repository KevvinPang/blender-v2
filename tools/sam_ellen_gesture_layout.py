"""Map observed arm layout about the shoulders into a fixed close-up shot.

A shared shoulder-relative body map retains left/right and above/below
relationships. Palm shape is translated intact, never squeezed by torso proportions. Raw observations and contact classification remain untouched.
"""
import json
import numpy as np


def prepare(scene,scratch,rig,channels,data,cameras,options):
    from sam_ellen_shot import mapping,render_points
    from sam_ellen_screen import project
    from sam_ellen_arm_visibility import screen_hand,hand_landmark_mask
    from sam_ellen_temporal import trajectory
    shot=mapping(scene.camera)
    data.pop('_sam_gesture_layout',None)
    if 'sam_gesture_layout' in scene.camera.data:del scene.camera.data['sam_gesture_layout']
    if not shot or options.get('camera_mode')!='FIXED':return
    n=len(data['frames']);height=data.get('video_orientation',{}).get('decoded_height_width',[1280])[0]
    times=np.array([r['time'] for r in data['frames']]);sources=[];targets=[];scales=[];caps=[]
    aspect=cameras[0][2];view=scratch.view_layers[0]
    for i,row in enumerate(data['frames']):
        for name,values in channels.items():rig.pose.bones[name].rotation_quaternion=values[i][1]
        view.update()
        image=np.asarray(row.get('image_points',[]),float)
        if image.ndim!=2 or len(image)<7:raise ValueError('近景手势适配缺少肩部观测')
        shoulder=image[[5,6],:2]/height;shoulder[:,0]*=aspect/data['screen']['aspect']
        if options.get('mirror'):shoulder[:,0]=aspect-shoulder[:,0]
        source=shoulder.mean(0)
        posed=render_points([project(rig.pose.bones['Arm_'+s].head,cameras[i]) for s in 'LR'],shot)
        target=posed.mean(0);scale=float(np.linalg.norm(posed[1]-posed[0])/max(.05,np.linalg.norm(shoulder[1]-shoulder[0])))
        # Only genuinely visible source landmarks constrain the shot. Missing
        # fingers and hands leaving the video are not pulled back into frame.
        cap=np.array([1.,scale])
        for side in 'LR':
            hand=screen_hand(data['screen'],i,side)
            if not hand:continue
            points=np.asarray(hand['points'],float).copy();points[:,0]*=aspect/data['screen']['aspect']
            if options.get('mirror'):points[:,0]=aspect-points[:,0]
            mask=hand_landmark_mask(hand)&(points[:,0]>.012)&(points[:,0]<aspect-.012)&(points[:,1]>.012)&(points[:,1]<.988)
            if mask.sum()<12:continue
            center=points[[0,5,9,13,17]].mean(0)
            delta=center-source;shape=points[mask]-center
            for axis,high in ((0,aspect),(1,1.)):
                value=delta[axis]
                bound=(high-(.015 if axis==0 else .07)-target[axis]-shape[:,axis].max())/value if value>1e-5 else ((.015 if axis==0 else .07)-target[axis]-shape[:,axis].min())/value if value< -1e-5 else scale
                if bound>0:cap[axis]=min(cap[axis],float(bound))
        sources.append(source);targets.append(target);scales.append(scale);caps.append(cap)
    source=trajectory(times,np.array(sources),.065)
    target=trajectory(times,np.array(targets),.065)
    scale=trajectory(times,np.asarray(caps),.10)
    # An earlier/later visible hand cannot be clipped by smoothing the scale.
    scale=np.minimum(scale,caps)
    layout={'model':'shoulder-relative-body-map-intact-palms-v2','video':data['video'],
        'times':times.tolist(),'scale':scale.tolist(),'source':source.tolist(),'target':target.tolist(),
        'shot':shot,'aspect':aspect}
    data['_sam_gesture_layout']=layout
    scene.camera.data['sam_gesture_layout']=json.dumps(layout)


def points(values,data,index,aspect,mirror=False,output=False):
    p=np.asarray(values,float).copy();p[:,0]*=aspect/data['screen']['aspect']
    if mirror:p[:,0]=aspect-p[:,0]
    layout=data.get('_sam_gesture_layout')
    if not layout:return p
    if len(p)==21:
        center=p[[0,5,9,13,17]].mean(0)
        destination=(center-np.asarray(layout['source'][index]))*layout['scale'][index]+np.asarray(layout['target'][index])
        p=p-center+destination
    else:
        p=(p-np.asarray(layout['source'][index]))*layout['scale'][index]+np.asarray(layout['target'][index])
    return p if output else (p-np.asarray(layout['shot']['offset']))/layout['shot']['scale']


def restore(scene,data):
    if '_sam_gesture_layout' in data:return
    raw=scene.camera.data.get('sam_gesture_layout')
    if not raw:return
    value=json.loads(raw)
    if value['video']==data['video'] and len(value['times'])==len(data['frames']) and np.allclose(value['times'],[r['time'] for r in data['frames']]):data['_sam_gesture_layout']=value

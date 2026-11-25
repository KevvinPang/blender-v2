"""Anatomical elbow placement around a hand arriving from outside the shot."""
import math

import numpy as np
from mathutils import Matrix, Quaternion, Vector

from calibrated_bridge import key_quat


def ease(x):
    x = min(1., max(0., x))
    return x*x*(3.-2.*x)


def redirect(rotation, source_axis, direction):
    current = rotation @ source_axis
    return current.rotation_difference(direction) @ rotation


def settle_before_entry(scene, bones, torso, track, start, sign, fade, previous):
    import bpy
    gap=start
    while gap and not track['valid'][gap-1]:
        gap-=1
    if start==gap or start-gap<fade or ('elbow_visible' not in track
            or track['elbow_visible'][gap:start].any()):
        return False
    scene.frame_set(max(1,gap))
    departure=[b.rotation_quaternion.copy() for b in bones]
    arm,elbow,wrist=bones
    for t in range(gap,start):
        scene.frame_set(t+1)
        body=torso.matrix.to_quaternion()@torso.bone.matrix_local.to_quaternion().inverted()
        weight=ease((t-gap+1)/max(1,fade*.65)) if gap else 1.
        for bone,child,offset,q in ((arm,elbow,(sign*.18,-.04,-1.),departure[0]),
                                    (elbow,wrist,(sign*.12,-.32,-1.),departure[1])):
            bind=bone.bone.matrix_local.to_quaternion()
            axis=(bind.inverted()@(child.bone.head_local-bone.bone.head_local)).normalized()
            neutral=redirect(body@bind,axis,body@Vector(offset).normalized())
            inherited=bone.parent.matrix.to_quaternion()@bone.parent.bone.matrix_local.to_quaternion().inverted()@bind
            key_quat(bone,q.slerp(inherited.inverted()@neutral,weight),t+1,previous)
            bpy.context.view_layer.update()
        key_quat(wrist,departure[2].slerp(Quaternion(),weight),t+1,previous)
    return True


def release_lower_exits(scene,rig,tracks):
    """Continue from the last visible pose to a lowered forearm in torso space."""
    import bpy
    fade=max(5,round(scene.render.fps/scene.render.fps_base*.36))
    torso=rig.pose.bones['UpperBody2']
    for side,track in tracks.items():
        if 'elbow_visible' not in track:continue
        obs=track['observed']
        sign=1. if side=='L' else -1.
        arm,elbow,wrist=[rig.pose.bones[n+'_'+side] for n in ('Arm','Elbow','Wrist')]
        previous={}
        for end in np.flatnonzero(obs & ~np.r_[obs[1:],False]):
            first=end+1
            if first>=len(obs) or track['edge'][first]!=3 or track['elbow_visible'][first]:continue
            if np.isfinite(track['boundary'][first]):continue
            stop=first
            while stop<len(obs) and not obs[stop] and not track['elbow_visible'][stop] and not np.isfinite(track['boundary'][stop]):
                if track['edge'][stop] not in (-1,3):break
                stop+=1
            if stop-first<fade+3:continue
            scene.frame_set(int(end+1))
            anchor=[b.rotation_quaternion.copy() for b in (arm,elbow,wrist)]
            # Sample before editing so later gap/arrival animation can be joined.
            original=[]
            for t in range(first,stop):
                scene.frame_set(t+1)
                original.append([b.rotation_quaternion.copy() for b in (arm,elbow,wrist)])
            for k,t in enumerate(range(first,stop)):
                scene.frame_set(t+1)
                body=torso.matrix.to_quaternion()@torso.bone.matrix_local.to_quaternion().inverted()
                entering=ease((k+1)/fade)
                leaving=ease(1.-(stop-1-t)/fade) if stop<len(obs) else 0.
                for j,(bone,child,offset) in enumerate(((arm,elbow,(sign*.2,-.05,-1.)),
                        (elbow,wrist,(sign*.12,-.33,-1.)),(wrist,None,None))):
                    local=Quaternion()
                    if child is not None:
                        bind=bone.bone.matrix_local.to_quaternion()
                        axis=(bind.inverted()@(child.bone.head_local-bone.bone.head_local)).normalized()
                        neutral=redirect(body@bind,axis,body@Vector(offset).normalized())
                        inherited=bone.parent.matrix.to_quaternion()@bone.parent.bone.matrix_local.to_quaternion().inverted()@bind
                        local=inherited.inverted()@neutral
                    q=anchor[j].slerp(local,entering).slerp(original[k][j],leaving)
                    key_quat(bone,q,t+1,previous);bpy.context.view_layer.update()


def relax_entries(scene, rig, tracks):
    import bpy
    fps = scene.render.fps / scene.render.fps_base
    fade = max(4, round(.4*fps))
    previous = {}
    count = 0
    for side, track in tracks.items():
        obs = track['observed']
        valid = track['valid']
        arm, elbow, wrist = [rig.pose.bones[n+'_'+side] for n in ('Arm','Elbow','Wrist')]
        torso = rig.pose.bones['UpperBody2']
        starts = np.flatnonzero(obs & ~np.r_[False, obs[:-1]])
        for first in starts:
            start = int(first)
            while start and valid[start-1] and not obs[start-1]:
                start -= 1
            # Only a boundary-supported arrival authorizes this recovery.
            if start == first or np.any(track.get('edge',np.full(len(obs),-1))[start:first] < 0):
                continue
            left = max(0,start-fade)
            sign = 1. if side=='L' else -1.
            if settle_before_entry(scene,(arm,elbow,wrist),torso,track,start,sign,fade,previous):
                left=start
            if np.any(valid[left:start]):
                continue
            end = int(first)
            while end+1 < len(obs) and obs[end+1] and end-first < fade:
                end += 1
            rows = []
            for t in range(left,end+1):
                scene.frame_set(t+1)
                rows.append(([b.head.copy() for b in (arm,elbow,wrist)],
                             [b.matrix.to_quaternion() for b in (arm,elbow,wrist)],
                             torso.matrix.to_quaternion() @ torso.bone.matrix_local.to_quaternion().inverted()))
            near_rows=rows[first-left+1:first-left+max(3,round(fps*.2))]
            clearance=np.median([(p[2]-p[0]).dot(body@Vector((0.,-1.,0.))) for p,_,body in near_rows]) if near_rows else 0.
            for t, (positions, original, body) in zip(range(left,end+1),rows):
                scene.frame_set(t+1)
                a,e,w = positions
                upper,lower = (e-a).length,(w-e).length
                down = body @ Vector((sign*.18,-.04,-1.)).normalized()
                fore = body @ Vector((sign*.12,-.32,-1.)).normalized()
                rest_goal = a+down*upper+fore*lower
                strength = (ease((t-left)/max(1,start-left)) if t<start else
                            1.-ease((t-first)/max(1,end-first)))
                if strength < 1e-6:
                    continue
                arrival = ease((t-start)/max(1,first-start))
                visible_goal=w.copy()
                front=body@Vector((0.,-1.,0.))
                camera=rig.matrix_world.inverted()@scene.camera.matrix_world.translation
                ray=(camera-w).normalized()
                # Preserve the wrist pixel when a partial detection puts the
                # visible hand behind the shirt in a front-facing shot.
                if ray.dot(front)>.45 and clearance>0:
                    missing=min(clearance,upper*.8)-(w-a).dot(front)
                    visible_goal+=ray*max(0.,missing)*strength/ray.dot(front)
                goal = visible_goal.lerp(rest_goal,(1.-arrival)*strength)
                direction = (goal-a).normalized()
                distance = np.clip((goal-a).length,abs(upper-lower)+1e-6,upper+lower-1e-6)
                goal = a+direction*distance
                base = e-a
                pole = base-direction*base.dot(direction)
                # A slightly rearward elbow leaves room for the wrist to rise
                # in front of the chest even on an almost vertical reach.
                hint = body @ Vector((sign*.18,.26,-1.))
                natural = hint-direction*hint.dot(direction)
                if natural.length < 1e-5:
                    natural = direction.orthogonal()
                natural.normalize()
                if pole.length > 1e-5:
                    pole.normalize()
                    angle = math.atan2(direction.dot(natural.cross(pole)),natural.dot(pole))
                    natural = Quaternion(direction,angle*(1.-strength)) @ natural
                along = (upper**2-lower**2+distance**2)/(2*distance)
                solved = a+direction*along+natural*math.sqrt(max(0.,upper**2-along**2))
                for i,(bone,child,target) in enumerate(((arm,elbow,solved-a),(elbow,wrist,goal-solved))):
                    bind = bone.bone.matrix_local.to_quaternion()
                    axis = (bind.inverted()@(child.bone.head_local-bone.bone.head_local)).normalized()
                    axis_target = target.normalized()
                    neutral = redirect(body@bind,axis,axis_target)
                    desired = redirect(original[i],axis,axis_target).slerp(neutral,strength)
                    inherited = bone.parent.matrix.to_quaternion() @ bone.parent.bone.matrix_local.to_quaternion().inverted() @ bind
                    key_quat(bone,inherited.inverted()@desired,t+1,previous)
                    bpy.context.view_layer.update()
                inherited = elbow.matrix.to_quaternion() @ elbow.bone.matrix_local.to_quaternion().inverted() @ wrist.bone.matrix_local.to_quaternion()
                # A visible wrist keeps its complete orientation, including roll.
                local = (inherited.inverted()@original[2]).slerp(Quaternion(),(1.-arrival)*strength)
                key_quat(wrist,local,t+1,previous)
                count += 1
    return count

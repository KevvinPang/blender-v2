"""Rotation calibration from reference poses, independent of bone roll."""
import math,bpy
from mathutils import Vector, Matrix, Quaternion
from .mapping import FINGERS, BODY


def evaluated(obj):
    return obj.evaluated_get(bpy.context.evaluated_depsgraph_get())


def world(obj):
    return evaluated(obj).matrix_world


def rest(obj, name):
    return world(obj) @ obj.data.bones[name].matrix_local


def pose(obj, name):
    ob=evaluated(obj)
    return ob.matrix_world @ ob.pose.bones[name].matrix


def direction(obj, name):
    b = obj.data.bones[name]
    return (world(obj).to_3x3() @ (b.tail_local-b.head_local)).normalized()


def length(obj, name):
    b = obj.data.bones[name]
    return (world(obj).to_3x3() @ (b.tail_local-b.head_local)).length


def anatomical_frame(y, normal):
    y = y.normalized()
    z = normal-y*normal.dot(y)
    if z.length < 1e-8:
        raise ValueError('手掌标定点退化，请检查手腕、食指、中指和小指映射。')
    z.normalize()
    x = y.cross(z).normalized()
    return Matrix((x, y, x.cross(y).normalized())).transposed()


def palm(obj, names):
    w, index, middle, pinky = [rest(obj,n).translation for n in names]
    along = (middle-w).normalized()
    normal = (index-pinky).normalized().cross(along).normalized()
    return anatomical_frame(along, normal)


def calibration(source, rig, mapping, arm=None, deform=None):
    deform = deform or {}
    offsets = {}
    for n, sn in mapping.items():
        sr, tr = rest(source,sn).to_quaternion(), rest(rig,n).to_quaternion()
        # Torso/head use the complete rest-relative rotation. Matching raw bone
        # Y axes here would reintroduce the fixed 24/30/7-degree posture errors.
        aim = Quaternion() if n in BODY else direction(rig,n).rotation_difference(direction(source,sn))
        if n.startswith('shoulder.') and n in deform:
            aim = direction(arm,deform[n]).rotation_difference(direction(source,sn))
        offsets[n] = sr.inverted() @ aim @ tr
    for side in 'LR':
        hand = 'hand_fk.'+side
        landmarks = [hand, 'f_index.01.'+side, 'f_middle.01.'+side, 'f_pinky.01.'+side]
        if not all(n in mapping for n in landmarks):
            continue
        ps = palm(source,[mapping[n] for n in landmarks])
        pt = palm(rig,landmarks)
        offsets[hand] = rest(source,mapping[hand]).to_quaternion().inverted() @ ps.to_quaternion() @ pt.to_quaternion().inverted() @ rest(rig,hand).to_quaternion()
        for base in FINGERS:
            for j in (1,2,3):
                n = f'{base}.{j:02d}.{side}'
                if n not in mapping:
                    continue
                so = anatomical_frame(direction(source,mapping[n]), ps.col[2])
                to = anatomical_frame(direction(arm,deform[n]) if n in deform else direction(rig,n), pt.col[2])
                offsets[n] = rest(source,mapping[n]).to_quaternion().inverted() @ so.to_quaternion() @ to.to_quaternion().inverted() @ rest(rig,n).to_quaternion()
    return offsets


def solve_two_bone(origin, goal, pole, a, b, q1, q2):
    delta = goal-origin
    if delta.length < 1e-8:
        delta = q1 @ Vector((0,1e-8,0))
    axis = delta.normalized()
    eps = max(1e-7, (a+b)*2e-5)
    reach = min(max(delta.length,abs(a-b)+eps),a+b-eps)
    along = (a*a-b*b+reach*reach)/(2*reach)
    bend = pole-axis*pole.dot(axis)
    if bend.length < 1e-7:
        bend = q1 @ Vector((0,0,1)); bend -= axis*bend.dot(axis)
    if bend.length < 1e-7:
        bend = axis.orthogonal()
    elbow = origin+axis*along+bend.normalized()*math.sqrt(max(0,a*a-along*along))
    endpoint = origin+axis*reach
    first = (q1@Vector((0,1,0))).rotation_difference((elbow-origin).normalized())@q1
    second = (q2@Vector((0,1,0))).rotation_difference((endpoint-elbow).normalized())@q2
    return first, second, abs(delta.length-reach)


def paired_arms(source, rig, mapping, quats, strength=1., heading=None):
    heading = heading or Quaternion()
    smid = sum((pose(source,mapping['upper_arm_fk.'+s]).translation for s in 'LR'),Vector())*.5
    tmid = sum((pose(rig,'upper_arm_fk.'+s).translation for s in 'LR'),Vector())*.5
    scale = sum(length(rig,n+'.L') for n in ['upper_arm_fk','forearm_fk'])/sum(length(source,mapping[n+'.L']) for n in ['upper_arm_fk','forearm_fk'])
    wrists = [pose(source,mapping['hand_fk.'+s]).translation for s in 'LR']
    source_length = sum(length(source,mapping[n+'.L']) for n in ['upper_arm_fk','forearm_fk'])
    proximity = max(0,min(1,(1.35-(wrists[0]-wrists[1]).length/source_length)/.65))
    weight = proximity*proximity*(3-2*proximity)
    fk = [pose(rig,'hand_fk.'+s).translation for s in 'LR']
    paired = [tmid+heading@((w-smid)*scale) for w in wrists]
    source_axis = (pose(source,mapping['upper_arm_fk.L']).translation-pose(source,mapping['upper_arm_fk.R']).translation).normalized()
    axis = heading@source_axis
    sg = (wrists[0]-wrists[1]).dot(source_axis)
    fg, pg = [(p[0]-p[1]).dot(axis) for p in [fk,paired]]
    if sg > .025 and fg < .5*pg and pg > fg:
        weight = max(weight,(.8*pg-fg)/(pg-fg))
    weight *= strength
    for i,s in enumerate('LR'):
        origin = pose(rig,'upper_arm_fk.'+s).translation
        radius = sum(length(rig,n+'.'+s) for n in ['upper_arm_fk','forearm_fk'])-1e-5
        p,v = fk[i]-origin,paired[i]-fk[i]
        aa,bb,cc = v.dot(v),2*p.dot(v),p.dot(p)-radius*radius
        if aa > 1e-12 and (p+weight*v).length > radius:
            weight = min(weight,max(0,(-bb+math.sqrt(max(0,bb*bb-4*aa*cc)))/(2*aa)))
    result = {}
    for i,s in enumerate('LR'):
        upper,lower = 'upper_arm_fk.'+s,'forearm_fk.'+s
        goal = fk[i].lerp(paired[i],weight)
        pole = heading@(pose(source,mapping[lower]).translation-pose(source,mapping[upper]).translation)
        a,b,_ = solve_two_bone(pose(rig,upper).translation,goal,pole,length(rig,upper),length(rig,lower),quats[upper],quats[lower])
        result[upper],result[lower] = a,b
    return result

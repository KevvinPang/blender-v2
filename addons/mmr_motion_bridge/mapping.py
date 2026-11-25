import re

DEFAULT = {'spine_fk':'Hips', 'spine_fk.001':'Spine', 'spine_fk.002':'Spine1',
           'spine_fk.003':'Spine2', 'neck':'Neck', 'head':'Head'}
DEFORM = {'spine_fk':'下半身', 'spine_fk.001':'上半身', 'spine_fk.002':'上半身1',
          'spine_fk.003':'上半身2', 'neck':'首', 'head':'頭'}
FINGERS = {'thumb':('Thumb',['親指０','親指１','親指２']),
           'f_index':('Index',['人指１','人指２','人指３']),
           'f_middle':('Middle',['中指１','中指２','中指３']),
           'f_ring':('Ring',['薬指１','薬指２','薬指３']),
           'f_pinky':('Pinky',['小指１','小指２','小指３'])}
for side, word in [('L','Left'), ('R','Right')]:
    for n,s,d in [('shoulder','Shoulder','肩'), ('upper_arm_fk','Arm','腕'),
                  ('forearm_fk','ForeArm','ひじ'), ('hand_fk','Hand','手首'),
                  ('thigh_fk','UpLeg','足'), ('shin_fk','Leg','ひざ'),
                  ('foot_fk','Foot','足首'), ('toe','ToeBase','足先EX')]:
        DEFAULT[n+'.'+side] = word+s
        DEFORM[n+'.'+side] = d+'.'+side
    for base, (finger, deform) in FINGERS.items():
        for j, d in enumerate(deform, 1):
            DEFAULT[f'{base}.{j:02d}.{side}'] = f'{word}Hand{finger}{j}'
            DEFORM[f'{base}.{j:02d}.{side}'] = d+'.'+side

REQUIRED = [n for n in DEFAULT if not n.startswith(('thumb.', 'f_', 'toe.'))]
SWITCHES = [base+'.'+side for side in 'LR' for base in ['upper_arm_parent','thigh_parent']]
BODY = ['spine_fk','spine_fk.001','spine_fk.002','spine_fk.003','neck','head']


def normalized(name):
    return re.sub('[^a-z0-9]', '', name.rsplit(':', 1)[-1].lower())


def auto_map(source):
    if not source:
        return dict(DEFAULT)
    lookup = {normalized(b.name): b.name for b in source.data.bones}
    result = {}
    for target, wanted in DEFAULT.items():
        result[target] = lookup.get(normalized(wanted), '')
    return result


def deform_names(arm):
    if not arm:
        return {}
    names = {b.name: b.name for b in arm.data.bones}
    for pb in arm.pose.bones:
        mmd = getattr(pb, 'mmd_bone', None)
        if mmd and getattr(mmd, 'name_j', ''):
            names.setdefault(mmd.name_j, pb.name)
    return {n: names[d] for n,d in DEFORM.items() if d in names}


def is_mmr(obj):
    return obj and obj.type == 'ARMATURE' and all(n in obj.pose.bones for n in ['torso','spine_fk','upper_arm_fk.L','thigh_fk.L','head'])


def deform_candidates(scene, rig, exclude=()):
    """MMD armatures with usable pose-bone bindings to this exact controller."""
    if not is_mmr(rig) or rig.name not in scene.objects:
        return []
    candidates = []
    for ob in scene.objects:
        if ob == rig or ob in exclude or ob.type != 'ARMATURE':
            continue
        known = set(deform_names(ob).values())
        score = sum(any(c.type in {'COPY_TRANSFORMS','COPY_ROTATION','COPY_LOCATION','CHILD_OF'}
                        and not c.mute and c.influence > 0
                        and getattr(c,'target',None) == rig
                        and getattr(c,'subtarget','') in rig.pose.bones
                        and not c.name.startswith('MMRB_')
                        for c in pb.constraints)
                    for pb in ob.pose.bones if pb.name in known)
        if score:
            candidates.append((score, ob))
    return [ob for score,ob in sorted(candidates,key=lambda x:(-x[0],x[1].name))]


def find_deform(scene, rig, exclude=()):
    candidates = deform_candidates(scene,rig,exclude)
    # More constraints do not prove that one of two bound characters is intended.
    return candidates[0] if len(candidates)==1 else None

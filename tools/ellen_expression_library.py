"""Additional facial presets and an animatable skin-attached cheek blush."""
import math

BLUSH_CONTROL = 'Ellen_CheekBlush'
BLUSH_ATTRIBUTE = 'Ellen_CheekBlushMask'
DISGUST_CONTROL = 'Ellen_DisgustFrown'
GRIMACE_CONTROL = 'Ellen_DisgustGrimace'
REFERENCE_EYES = 'Ellen_DisgustReferenceEyes'
REFERENCE_BROWS = 'Ellen_DisgustReferenceBrows'
REFERENCE_MOUTH = 'Ellen_DisgustReferenceMouth'
REFERENCE_CONTROLS = (REFERENCE_EYES, REFERENCE_BROWS, REFERENCE_MOUTH)
PAIR_CONTROLS = ('Ellen_DisgustSmallOpen', 'Ellen_DisgustCurvedTeeth')
SIDE_GLANCE_CONTROL = 'Ellen_ReferenceSideGlance'
SOFT_SMILE_CONTROL = 'Ellen_ReferenceSoftSmile'
BLANK_BROWS_CONTROL = 'Ellen_ReferenceBlankBrows'
BELLE_REFERENCE_PRESET = 'belle_weary'
BELLE_WEARY_BROWS = 'Belle_WearyReferenceBrows'
SELECTED_REFERENCE_PRESETS = ('relaxed_listen', 'side_glance', 'playful_wink', 'blank_stare')
TEETH_PRESETS = frozenset(('laugh', 'disgust', 'disgust_curved'))
EXTRA_PRESETS = {
    'happy_closed': ('弯眼甜笑', {'笑い': 1.0, 'にこり': .78, '上': .28, '口角上げ': .68,
                               '口横狭め': .08}),
    'disgust': ('嫌弃 / 小开口', {PAIR_CONTROLS[0]: 1.0, 'Ellen_ShowTeeth': 1.0,
                                 'Ellen_ForeheadShadow': .60}),
    'shy': ('脸红害羞', {'じと目': .2, 'まばたき': .12, '困る': .48, 'にこり': .2,
                       '上': .18, '口角上げ': .2, '口横狭め': .2, BLUSH_CONTROL: .85}),
    'blush_smile': ('脸红闭眼笑', {'笑い': 1.0, 'にこり': .65, '上': .23,
                                  '口角上げ': .42, BLUSH_CONTROL: .75}),
    'embarrassed': ('害羞 / 窘迫', {'じと目': .42, '困る': .95, '上': .3, '下': .12,
                                    'ウィンク２右': .20, '口角下げ': .3, '口横狭め': .15, 'あ': .08, BLUSH_CONTROL: 1.0}),
    'smug': ('得意 / 坏笑', {'じと目': .5, 'にこり': .4, '上左': .70,
                              '上右': .1, '口角上げ': .78, '口横狭め': .08}),
    'surprised': ('惊讶 / 愣住', {'びっくり左': .7, 'びっくり右': .7, '上': .8,
                                  '困る': .12, 'お': .3}),
    'pout': ('委屈 / 小生气', {'じと目': .4, '怒り': .68, '困る': .25, '下': .42,
                                'う': .3, '口角下げ': .48, '口横狭め': .15}),
    'disgust_curved': ('嫌弃 / 弧形露牙', {PAIR_CONTROLS[1]: 1.0, 'Ellen_ShowTeeth': 1.0,
                                       'Ellen_ForeheadShadow': .60}),
    'relaxed_listen': ('参考01 / 慵懒听你说', {'じと目': .45, 'まばたき': .025,
                         'にこり': .45, '上': .12, SOFT_SMILE_CONTROL: .38, '口横狭め': .06}),
    'side_glance': ('参考02 / 半眯眼侧视', {'じと目': 1.0, 'まばたき': .48,
                       '真面目': .18, '上左': .20, '上右': .04,
                       '口角上げ': .10, SIDE_GLANCE_CONTROL: 1.0}),
    'playful_wink': ('参考09 / 调皮眨眼', {'ウィンク２右': 1.0, 'にこり': .60,
                         '上': .13, '口角上げ': .28, 'ワ': .52}),
    'blank_stare': ('新参考 / 呆萌小开口', {'じと目': .15, BLANK_BROWS_CONTROL: 1.0,
                        'お': .085, '口横狭め': .12}),
    BELLE_REFERENCE_PRESET: ('铃 / 半眯眼坏笑', {'じと目': .85, 'まばたき': .18,
                        'にこり': .20, '上': .06, '口角下げ': .16, '口角上げ': .14,
                        '口横狭め': .40, BLUSH_CONTROL: .30}),
}

EXTRA_PRESETS['blank_stare_shy'] = ('呆萌 / 微红害羞', {
    **EXTRA_PRESETS['blank_stare'][1], BLUSH_CONTROL: .65})

MOUTH_PROFILES = {
    'happy_closed': (.9, .25, .85), 'disgust': (.82, 0, .8),
    'shy': (.85, .1, .82), 'blush_smile': (.9, .25, .85),
    'embarrassed': (.85, 0, .82), 'smug': (.88, .16, .85),
    'surprised': (1.0, 0, .95), 'pout': (.85, 0, .82),
    'disgust_curved': (.82, 0, .8),
    'relaxed_listen': (.90, .08, .85), 'side_glance': (.88, 0, .85),
    'playful_wink': (.90, .20, .85), 'blank_stare': (.95, 0, .90),
    BELLE_REFERENCE_PRESET: (.90, .12, .85),
}
MOUTH_PROFILES['blank_stare_shy'] = MOUTH_PROFILES['blank_stare']

# This opt-in profile belongs to Ellen's revised scene; other actors keep their
# existing presets and saved enum numbers.
CORE_POUT = 'Ellen_CorePout'
CORE_SMIRK = 'Ellen_CoreSmirk'
CORE_DOWN_GAZE = 'Ellen_CoreDownGaze'
CORE_ANGRY_BROWS = 'Ellen_CoreAngryBrows'
CORE_SAD_BROWS = 'Ellen_CoreSadBrows'
CORE_FROWN = 'Ellen_CoreFrown'
CORE_STARTLE = 'Ellen_CoreStartleIris'
CORE_SMILE = 'Ellen_CoreSmoothSmile'
CONFIDENT_FANG = 'Ellen_ConfidentSingleFang'
SOFT_SAD_TEETH = 'Ellen_SoftSadTeethRest'
SOFT_SAD_EYE_TILT = 'Ellen_SoftSadEyeTilt'
SOFT_SAD_EYE_TAIL = 'Ellen_SoftSadEyeTailCurve'
ELLEN_CORE_PRESETS = {
    'neutral': ('自然 / 听你说', {}),
    'ellen_core_smile': ('开心 / 明显微笑', {
        '笑い': .35, 'にこり': .65, CORE_SMILE: 1, '上': .2}),
    'ellen_core_laugh': ('大笑 / 闭眼露牙', {
        '笑い': 1, 'にこり': .8, '上': .25, 'ワ': .85,
        '口角上げ': .4, 'Ellen_ShowTeeth': 1}),
    'ellen_core_surprise': ('惊讶 / 睁眼张嘴', {
        'びっくり左': 1, 'びっくり右': 1, '上': .9, 'お': .65, CORE_STARTLE: 1}),
    'ellen_core_angry': ('生气 / 皱眉质问', {
        CORE_ANGRY_BROWS: 1, '怒り目': .85, 'じと目': .22,
        CORE_FROWN: .65, 'あ': .16}),
    'ellen_core_tsundere': ('傲娇 / 生气却脸红', {
        CORE_ANGRY_BROWS: .9, '怒り目': .72, 'じと目': .28,
        CORE_POUT: 1, SIDE_GLANCE_CONTROL: .8, BLUSH_CONTROL: 1}),
    'ellen_core_shy': ('害羞 / 垂眼躲闪', {
        CORE_SAD_BROWS: .5, 'じと目': .25, 'まばたき': .15,
        CORE_DOWN_GAZE: 1, CORE_SMILE: .3, BLUSH_CONTROL: .9}),
    'ellen_core_sad': ('委屈 / 难过', {
        CORE_SAD_BROWS: 1, 'じと目': .18, 'まばたき': .08,
        CORE_FROWN: 1, CORE_DOWN_GAZE: .35, 'あ': .08}),
    'ellen_core_disgust': ('嫌弃 / 咧嘴侧视', {
        PAIR_CONTROLS[1]: 1, SIDE_GLANCE_CONTROL: 1,
        'Ellen_ShowTeeth': 1, 'Ellen_ForeheadShadow': .4}),
    'ellen_core_smug': ('得意 / 单边坏笑', {
        'じと目': .6, '上左': 1, '上右': .05,
        CORE_SMIRK: 1, SIDE_GLANCE_CONTROL: .45}),
}
# Keep requested entries from the original 24-expression catalog as
# explicit choices on the revised Ellen profile.  They use the authored
# controls, so their appearance remains the one shown in the old catalog
# instead of silently collapsing into one of the ten core expressions.
ELLEN_CORE_PRESETS.update({
    'ellen_legacy_08_brooding': ('旧08 / 阴沉·额头阴影', {
        '怒り': .95, '下': .3, 'じと目': .65, 'まばたき': .18,
        'あ': .2, '口角下げ': .15, 'Ellen_ForeheadShadow': 1.0}),
    'ellen_legacy_09_worried': ('旧09 / 担忧·委屈', {
        '困る': .95, 'じと目': .15, 'まばたき': .1, 'あ': .1,
        '口角下げ': .3}),
    'ellen_legacy_10_smile_closed': ('旧10 / 闭眼笑着说话', {
        '笑い': 1.0, 'にこり': .7, '上': .15, '口角上げ': .4}),
    'ellen_legacy_22_playful_wink': ('旧22 / 调皮眨眼', {
        'ウィンク２右': 1.0, 'にこり': .60, '上': .13,
        '口角上げ': .28, 'ワ': .52}),
    # Append to preserve numeric enum IDs already saved for the other entries.
    'ellen_legacy_15_embarrassed': ('旧15 / 害羞·窘迫',
        dict(EXTRA_PRESETS['embarrassed'][1])),
    'ellen_confident_fang': ('得意 / 半眯眼浅露齿', {
        # Exact eyes/brows from the archived corrected smile (2026-11-09).
        # The native tooth row follows vowels correctly; the old compressed
        # row was compensated for one fixed mouth pose and could sink behind
        # the lips when captured speech changed that pose.
        '笑い': .35, 'にこり': .65, '上': .20,
        'ワ': .48, '口角上げ': .32,
        CORE_SMIRK: .50, CONFIDENT_FANG: 0, 'Ellen_ShowTeeth': 1}),
    'ellen_melancholy': ('淡淡忧伤 / 舒展垂眼', {
        # Reference only the relaxed eyes and gently raised inner brows.
        # Keep the established closed mouth; captured speech controls opening.
        'じと目': .14, 'まばたき': .025, '困る': .18,
        '悲しむ': .08, '眼角下': .08,
        CORE_SAD_BROWS: .80, CORE_FROWN: .34, SOFT_SAD_EYE_TILT: .52,
        SOFT_SAD_EYE_TAIL: 1,
        SOFT_SAD_TEETH: 0, 'Ellen_ShowTeeth': 1}),
    # Append: retain the enum numbers used by existing expression takes.
    # Closed, small frown with heavy upper lids; captured speech opens it.
    'ellen_disdain_closed': ('嫌弃脸 / 半眯眼闭嘴', {
        'じと目': 1, 'まばたき': .22, '怒り目': .18,
        CORE_ANGRY_BROWS: .45, '真面目': .18,
        CORE_FROWN: .45, '口横狭め': .20, 'Ellen_ShowTeeth': 1}),
})
ELLEN_CORE_CONTROLS = (CORE_POUT, CORE_SMIRK, CORE_DOWN_GAZE,
                       CORE_ANGRY_BROWS, CORE_SAD_BROWS, CORE_FROWN, CORE_STARTLE, CORE_SMILE,
                       CONFIDENT_FANG, '悲しむ', '眼角下', SOFT_SAD_TEETH,
                       SOFT_SAD_EYE_TILT, SOFT_SAD_EYE_TAIL)
ELLEN_CORE_MOUTH_PROFILES = {
    name: (1.0, .45 if name == 'ellen_core_laugh' else
           .3 if name == 'ellen_core_smile' else 0.0, 1.0)
    for name in ELLEN_CORE_PRESETS
}
ELLEN_CORE_MOUTH_PROFILES.update({
    'ellen_legacy_08_brooding': (.72, 0.0, .80),
    'ellen_legacy_09_worried': (.78, 0.0, .80),
    'ellen_legacy_10_smile_closed': (.85, .24, .80),
    'ellen_legacy_22_playful_wink': (.90, .20, .85),
    'ellen_legacy_15_embarrassed': MOUTH_PROFILES['embarrassed'],
    'ellen_confident_fang': (1.0, .55, 1.0),
})
ELLEN_CORE_TEETH_PRESETS = frozenset(('ellen_core_laugh', 'ellen_core_disgust',
                                    'ellen_confident_fang', 'ellen_melancholy',
                                    'ellen_disdain_closed'))
ELLEN_CORE_LEGACY_MAP = {
    'neutral': 'neutral', 'talk': 'neutral', 'smile': 'ellen_core_smile',
    'unimpressed': 'ellen_core_disgust', 'serious': 'ellen_core_angry',
    'laugh': 'ellen_core_laugh', 'emphasis': 'ellen_core_angry',
    'brooding': 'ellen_legacy_08_brooding', 'worried': 'ellen_legacy_09_worried',
    'smile_closed': 'ellen_legacy_10_smile_closed', 'happy_closed': 'ellen_core_smile',
    'disgust': 'ellen_core_disgust', 'shy': 'ellen_core_shy',
    'blush_smile': 'ellen_core_shy', 'embarrassed': 'ellen_legacy_15_embarrassed',
    'smug': 'ellen_core_smug', 'surprised': 'ellen_core_surprise',
    'pout': 'ellen_core_tsundere', 'disgust_curved': 'ellen_core_disgust',
    'relaxed_listen': 'neutral', 'side_glance': 'ellen_core_disgust',
    'playful_wink': 'ellen_legacy_22_playful_wink', 'blank_stare': 'neutral',
    'blank_stare_shy': 'ellen_core_shy',
}


def ensure_ellen_core_controls(mesh):
    """Local additive sculpts, measured against Ellen's own brow/iris shells."""
    import numpy as np
    keys = mesh.data.shape_keys.key_blocks
    assert '艾莲' in mesh.name, '此核心表情仅适用于艾莲'
    ensure_blush_control(mesh)
    ensure_disgust_pair(mesh)
    ensure_reference_side_glance(mesh)
    ensure_reference_soft_smile(mesh)
    base = np.empty((len(keys[0].data), 3), dtype=np.float32)
    keys[0].data.foreach_get('co', base.ravel())
    labels = [m.get('original_material', m.name.rsplit('|', 1)[-1].strip().split('.')[0])
              if m else '' for m in mesh.data.materials]
    regions = {name: set() for name in ('眉', '目')}
    for p in mesh.data.polygons:
        if labels[p.material_index] in regions:
            regions[labels[p.material_index]].update(p.vertices)
    brows = np.array(sorted(regions['眉']), dtype=int)
    irises = np.array(sorted(regions['目']), dtype=int)
    assert len(brows) and len(irises)
    def delta(name):
        a = np.empty_like(base)
        keys[name].data.foreach_get('co', a.ravel())
        return a-base
    def put(name, values):
        if name in keys:
            return
        key = mesh.shape_key_add(name=name, from_mix=False)
        key.data.foreach_set('co', (base+values).ravel())
        key.relative_key = keys[0]
        key.value = 0.0
    put(CORE_FROWN, delta('口角下げ')*2.2)
    put(CORE_POUT, delta('口角下げ')*1.5+delta('口横狭め')*.45)
    # Bend the closed mouth and surrounding skin together with a smooth field.
    # Amplifying the authored corner key creates pointed folds at the endpoints.
    mouth_ids = {i for p in mesh.data.polygons if labels[p.material_index] == '口线'
                 for i in p.vertices}
    center_z = float(np.median(base[list(mouth_ids), 2]))
    mouth_shells = {i for p in mesh.data.polygons
                    if labels[p.material_index] in {'颜', '颜2', '口线', '口舌'}
                    for i in p.vertices}
    mask = np.zeros(len(base), dtype=np.float32)
    mask[list(mouth_shells)] = 1
    x, y, z = base.T
    weight = np.exp(-(x/.022)**4-((z-center_z)/.008)**2)
    weight *= np.clip((-.04-y)/.01, 0, 1)*mask
    smile = np.zeros_like(base)
    smile[:, 2] = (.0032*(x/.011)**2-.00035)*weight
    smile[:, 0] = x*.07*weight
    put(CORE_SMILE, smile)
    smirk = smile*np.clip(.5+base[:, 0]/.012, .05, 1)[:, None]
    put(CORE_SMIRK, smirk)
    # Keep the authored upper row, showing only a shallow strip. Weld the
    # positional duplicates in the tooth material to distinguish upper/lower
    # tooth shells without relying on vertex numbering.
    teeth = {i for p in mesh.data.polygons if labels[p.material_index] == '齿'
             for i in p.vertices}
    sad_teeth = np.zeros_like(base)
    sad_teeth[list(teeth), 1] = .05
    put(SOFT_SAD_TEETH, sad_teeth)
    position = {i: tuple(np.round(base[i], 6)) for i in teeth}
    graph = {p: set() for p in position.values()}
    for polygon in mesh.data.polygons:
        if labels[polygon.material_index] == '齿':
            points = {position[i] for i in polygon.vertices}
            for point in points: graph[point].update(points)
    upper = set()
    stack = [p for p in graph if p[2] > 1.305]
    while stack:
        point = stack.pop()
        if point in upper: continue
        upper.add(point); stack.extend(graph[point]-upper)
    upper_ids = np.array([i for i in teeth if position[i] in upper], dtype=int)
    assert len(upper_ids) > 20
    fang = np.zeros_like(base)
    fang[list(teeth), 1] = .05
    authored_open = delta('ワ')*.48+delta('口横広げ')*.30+delta('口角上げ')*.25
    fang[upper_ids, 1] = -.002-authored_open[upper_ids, 1]
    fang[upper_ids, 2] = (1.2995+(base[upper_ids, 2]-1.3015)*.18
                           -base[upper_ids, 2]-authored_open[upper_ids, 2])
    put(CONFIDENT_FANG, fang)
    if mesh.get('ellen_confident_teeth_revision', 0) < 2:
        keys[CONFIDENT_FANG].data.foreach_set('co', (base+fang).ravel())
        mesh['ellen_confident_teeth_revision'] = 2
    gaze = np.zeros_like(base)
    gaze[irises, 2] = -.0065
    put(CORE_DOWN_GAZE, gaze)
    startle = np.zeros_like(base)
    iris_shells = {i for p in mesh.data.polygons
                   if labels[p.material_index] in {'目', '目光', '目光2', '目光3'}
                   for i in p.vertices}
    for sign in (-1, 1):
        eye = np.array([i for i in iris_shells if base[i, 0]*sign > 0], dtype=int)
        center = (base[eye].min(axis=0)+base[eye].max(axis=0))*.5
        startle[eye, 0] = (center[0]-base[eye, 0])*.2
        startle[eye, 2] = (center[2]-base[eye, 2])*.2
    put(CORE_STARTLE, startle)
    inner = min(abs(base[brows, 0]));outer = max(abs(base[brows, 0]))
    t = np.clip((abs(base[brows, 0])-inner)/(outer-inner), 0, 1)
    angry = np.zeros_like(base)
    angry[brows, 2] = -.008*(1-t)+.005*t
    angry[brows, 0] = -np.sign(base[brows, 0])*.0015*(1-t)
    put(CORE_ANGRY_BROWS, angry)
    sad = np.zeros_like(base)
    sad[brows, 2] = .012*(1-t)-.0045*t
    put(CORE_SAD_BROWS, sad)
    # Rotate each complete eye opening in the melancholy pose, including lids,
    # whites, iris and nearby skin. A shared smooth field avoids seams between
    # the separately authored eye shells. Other expressions leave this at zero.
    if SOFT_SAD_EYE_TILT not in keys:
        posed = base.copy()
        for name, value in (('じと目', .42), ('まばたき', .12),
                            ('悲しむ', .18), ('眼角下', .15)):
            posed += delta(name)*value
        eye_materials = {'颜', '颜2', '白目', '目', '睫', '二重影', '二重'}
        eye_ids = {i for p in mesh.data.polygons
                   if labels[p.material_index] in eye_materials for i in p.vertices}
        tilt = np.zeros_like(base)
        theta = math.radians(8)
        def fade(v):
            a = np.clip(v, 0, 1)
            return 1-a*a*(3-2*a)
        for sign in (-1, 1):
            eye = irises[posed[irises, 0]*sign > 0]
            center = (posed[eye].min(axis=0)+posed[eye].max(axis=0))*.5
            side = np.array(sorted(i for i in eye_ids if posed[i, 0]*sign > 0), dtype=int)
            u = posed[side, 0]*sign-abs(center[0])
            v = posed[side, 2]-center[2]
            weight = fade((abs(u)-.027)/.014)*fade((abs(v)-.016)/.013)
            weight *= np.clip((-.018-posed[side, 1])/.012, 0, 1)
            tilt[side, 0] = sign*((math.cos(theta)-1)*u+math.sin(theta)*v)*weight
            tilt[side, 2] = (-math.sin(theta)*u+(math.cos(theta)-1)*v)*weight
        put(SOFT_SAD_EYE_TILT, tilt)
    # Curve only the outer lid/lash ends downward. Whole-eye rotation leaves
    # Ellen's authored outer wings upturned, so it cannot create this contour.
    if SOFT_SAD_EYE_TAIL not in keys:
        posed = base.copy()
        for name, value in (('じと目', .14), ('まばたき', .025), ('悲しむ', .08),
                            ('眼角下', .08), (SOFT_SAD_EYE_TILT, .52)):
            posed += delta(name)*value
        eye_materials = {'颜', '颜2', '白目', '目', '睫', '二重影', '二重'}
        ids = np.array(sorted({i for p in mesh.data.polygons
                               if labels[p.material_index] in eye_materials
                               for i in p.vertices}), dtype=int)
        u = np.clip((np.abs(posed[ids, 0])-.043)/.024, 0, 1)
        t = np.clip((np.abs(posed[ids, 2]-1.347)-.014)/.012, 0, 1)
        weight = u*u*(1-t*t*(3-2*t))
        weight *= np.clip((-.018-posed[ids, 1])/.012, 0, 1)
        curve = np.zeros_like(base)
        curve[ids, 2] = -.008*weight
        put(SOFT_SAD_EYE_TAIL, curve)
    if mesh.get('EllenExpressionTransferAlignment') and '校服' in mesh.name:
        ensure_ellen_school_confident_teeth(mesh)


def ensure_ellen_school_confident_teeth(mesh):
    """Keep the school's shallow tooth row inside the lip opening."""
    import json
    import numpy as np
    if mesh.get('ellen_school_confident_teeth_revision', 0) >= 1:
        return
    alignment = json.loads(mesh['EllenExpressionTransferAlignment'])
    scale = alignment['scale']
    keys = mesh.data.shape_keys.key_blocks
    def coords(key):
        a = np.empty((len(key.data), 3), np.float32)
        key.data.foreach_get('co', a.ravel())
        return a
    base = coords(keys[0])
    old = coords(keys[CONFIDENT_FANG])
    labels = [m.get('original_material', m.name.rsplit('|', 1)[-1].strip().split('.')[0])
              if m else '' for m in mesh.data.materials]
    teeth = np.array(sorted({v for p in mesh.data.polygons
                            if labels[p.material_index] == '齿' for v in p.vertices}), dtype=int)
    upper = teeth[(old[teeth, 1]-base[teeth, 1]) < .02*scale]
    assert len(upper) > 20
    recessed = old.copy()
    recessed[upper, 1] += .004*scale
    recessed[upper, 0] *= .88
    natural = base.copy()
    natural[teeth, 1] += .05*scale
    natural[upper] = base[upper]
    natural[upper, 1] += .005*scale
    natural[upper, 0] *= .86
    natural[upper, 2] += .0008*scale
    fixed = recessed*.65+natural*.35
    # Keep the original non-tooth coordinates byte-identical.
    old[teeth] = fixed[teeth]
    keys[CONFIDENT_FANG].data.foreach_set('co', old.ravel())
    mesh['ellen_school_confident_teeth_revision'] = 1


def ensure_belle_weary_brows(mesh):
    """Flatten Belle's brow arch and lift the inner ends for the weary reference."""
    keys = mesh.data.shape_keys
    if BELLE_WEARY_BROWS in keys.key_blocks:
        return
    if 'Belle_MouthWidth_Adjustment' not in keys.key_blocks:
        raise ValueError('此参考眉形仅为铃模型配置')
    labels = [m.get('original_material', m.name.rsplit('|', 1)[-1].strip().split('.')[0])
              if m else '' for m in mesh.data.materials]
    ids = {v for p in mesh.data.polygons if labels[p.material_index] == '眉' for v in p.vertices}
    if not ids or '困る' not in keys.key_blocks:
        raise ValueError('缺少铃的眉毛网格或原始眉形')
    rest = keys.key_blocks[0].data
    inner = min(abs(rest[i].co.x) for i in ids)
    outer = max(abs(rest[i].co.x) for i in ids)
    key = mesh.shape_key_add(name=BELLE_WEARY_BROWS, from_mix=False)
    for i in ids:
        t = (abs(rest[i].co.x) - inner) / (outer - inner)
        key.data[i].co = rest[i].co + (keys.key_blocks['困る'].data[i].co - rest[i].co) * .55
        key.data[i].co.z += .003 * (1 - t) - .001 * t
        # Keep the moved brow surface in front of the forehead skin.
        key.data[i].co.y -= .0015
    key.value = 0


def ensure_reference_side_glance(mesh):
    """A small iris-only offset, independent of captured eyelids and eye bones."""
    keys = mesh.data.shape_keys
    if SIDE_GLANCE_CONTROL in keys.key_blocks:
        return
    slots = {i for i, material in enumerate(mesh.data.materials)
             if material and material.name.rsplit('|', 1)[-1].strip() == '目'}
    iris_ids = {v for polygon in mesh.data.polygons if polygon.material_index in slots
                for v in polygon.vertices}
    if not iris_ids:
        raise ValueError('未找到艾莲独立的虹膜材质，无法创建侧视表情')
    rest = keys.key_blocks[0].data
    key = mesh.shape_key_add(name=SIDE_GLANCE_CONTROL, from_mix=False)
    for index in iris_ids:
        key.data[index].co = rest[index].co
        key.data[index].co.x += .003
        key.data[index].co.y -= .00035
    key.value = 0


def ensure_reference_blank_brows(mesh):
    """Raise the inner brow ends and lower the outer ends on both sides."""
    keys = mesh.data.shape_keys
    if BLANK_BROWS_CONTROL in keys.key_blocks:
        return
    slots = {i for i, material in enumerate(mesh.data.materials)
             if material and material.name.rsplit('|', 1)[-1].strip() == '眉'}
    brow_ids = {v for polygon in mesh.data.polygons if polygon.material_index in slots
                for v in polygon.vertices}
    if not brow_ids:
        raise ValueError('未找到艾莲独立的眉毛材质，无法创建参考眉形')
    rest = keys.key_blocks[0].data
    key = mesh.shape_key_add(name=BLANK_BROWS_CONTROL, from_mix=False)
    for index in brow_ids:
        t = max(0, min(1, (abs(rest[index].co.x) - .015) / .045))
        key.data[index].co.z += .010 * (1 - t) - .005 * t
    key.value = 0


def ensure_reference_soft_smile(mesh):
    """Make the subtle authored mouth-corner lift readable in a closeup."""
    keys = mesh.data.shape_keys
    if SOFT_SMILE_CONTROL in keys.key_blocks:
        return
    rest = keys.key_blocks[0].data
    up = keys.key_blocks.get('口角上げ')
    if up is None:
        raise ValueError('缺少原始嘴角上扬形态，无法创建轻笑表情')
    key = mesh.shape_key_add(name=SOFT_SMILE_CONTROL, from_mix=False)
    for i, point in enumerate(key.data):
        basis = rest[i].co
        point.co = basis + (up.data[i].co - basis) * 3.0
    key.value = 0


def ensure_disgust_pair(mesh):
    """Separate the two references instead of blending incompatible mouth shapes."""
    keys = mesh.data.shape_keys
    if all(name in keys.key_blocks for name in PAIR_CONTROLS) and mesh.get('ellen_disgust_pair_revision', 0) >= 2:
        return
    rest = keys.key_blocks[0].data
    required = ('じと目', 'まばたき', '□', 'え３', 'い２')
    if any(name not in keys.key_blocks for name in required):
        raise ValueError('当前角色缺少艾莲原始嘴形和眼形')
    ids = {}
    for label in ('眉', '齿'):
        slots = {i for i, m in enumerate(mesh.data.materials)
                 if m and m.name.rsplit('|', 1)[-1].strip() == label}
        ids[label] = {v for p in mesh.data.polygons if p.material_index in slots for v in p.vertices}
    for index, name in enumerate(PAIR_CONTROLS):
        key = keys.key_blocks.get(name)
        old_value = key.value if key is not None else 0.0
        if key is None:
            key = mesh.shape_key_add(name=name, from_mix=False)
        blink, square, e3, bend = (.28, .22, .10, .001) if index == 0 else (.32, .035, 0, .008)
        for i, point in enumerate(key.data):
            co = rest[i].co.copy()
            x, y, z = co
            if z > 1.325 and i not in ids['眉']:
                for source, weight in (('じと目', .9), ('まばたき', blink)):
                    co += (keys.key_blocks[source].data[i].co-rest[i].co)*weight
            if i in ids['眉']:
                t = max(0, min(1, (abs(x)-.015)/.045))
                co.z += .008*(1-t)**1.8-.002*t+.0008*math.sin(math.pi*t)
                co.x -= math.copysign(.0006*(1-t), x)
            if z < 1.332:
                fade = math.exp(-(x/.029)**4-((z-1.301)/.018)**2)
                fade *= max(0, min(1, (-.01-y)/.02))
                co += ((keys.key_blocks['□'].data[i].co-rest[i].co)*square
                       +(keys.key_blocks['え３'].data[i].co-rest[i].co)*e3
                       +(keys.key_blocks['い２'].data[i].co-rest[i].co)*(.5 if index else 0))*fade
                co.z += (bend*.3-bend*(x/.018)**2+.0015*x/.02)*fade
                if index == 0:
                    co.x -= co.x*.12*fade
                elif i in ids['齿']:
                    co.z = 1.301+(co.z-1.301)*.8
                if index == 1:
                    co.x += co.x*.08*fade
            point.co = co
        key.value = old_value
    mesh['ellen_disgust_pair_revision'] = 2


def ensure_reference_disgust(mesh):
    """Three localized additive sculpts; authored keys and Basis stay unchanged."""
    keys = mesh.data.shape_keys
    if all(name in keys.key_blocks for name in REFERENCE_CONTROLS) and mesh.get('ellen_disgust_mouth_revision', 0) >= 2:
        return
    required = ('い２', 'え３', 'じと目', '怒り目２', '眼角下', 'まばたき')
    if any(name not in keys.key_blocks for name in required):
        raise ValueError('当前角色缺少艾莲面部形态，无法创建参考嫌弃脸')
    slots = {label: {i for i, m in enumerate(mesh.data.materials)
                    if m and m.name.rsplit('|', 1)[-1].strip() == label}
             for label in ('眉', '齿')}
    ids = {label: {v for p in mesh.data.polygons if p.material_index in indices for v in p.vertices}
           for label, indices in slots.items()}
    if not ids['眉'] or not ids['齿']:
        raise ValueError('未找到独立眉毛或牙齿材质')
    rest = keys.key_blocks[0].data
    for name in REFERENCE_CONTROLS:
        if name in keys.key_blocks and name != REFERENCE_MOUTH:
            continue
        key = keys.key_blocks.get(name)
        if key is None:
            key = mesh.shape_key_add(name=name, from_mix=False)
        for i, point in enumerate(key.data):
            co = rest[i].co.copy()
            x, y, z = co
            if name == REFERENCE_EYES and z > 1.325 and i not in ids['眉']:
                side_weight = .55 if x < 0 else 1.0
                for source, weight in (('じと目', .7), ('怒り目２', .1),
                                       ('眼角下', .2), ('まばたき', .18)):
                    co += (keys.key_blocks[source].data[i].co-rest[i].co)*weight*side_weight
            elif name == REFERENCE_BROWS and i in ids['眉']:
                t = max(0, min(1, (abs(x)-.015)/.045))
                co.z += .014*(1-t)**2.2-.003*t-.002*math.sin(math.pi*t)
                co.x -= math.copysign(.0015*(1-t), x)
            elif name == REFERENCE_MOUTH and z < 1.332:
                fade = math.exp(-(x/.029)**4-((z-1.301)/.018)**2)
                fade *= max(0, min(1, (-.01-y)/.02))
                co += ((keys.key_blocks['い２'].data[i].co-rest[i].co)*.6
                       +(keys.key_blocks['え３'].data[i].co-rest[i].co)*.10)*fade
                co.x += .0015*fade
                co.z += (1.301-co.z)*.25*fade
                co.z += .003*(x/.022)*fade
            point.co = co
        key.value = 0
    mesh['ellen_disgust_mouth_revision'] = 2


def ensure_disgust_grimace(mesh):
    """Narrow, downturned tooth-bearing mouth, separate from captured vowels."""
    keys = mesh.data.shape_keys
    if GRIMACE_CONTROL in keys.key_blocks:
        return
    rest = keys.key_blocks[0]
    down = keys.key_blocks.get('口角下げ')
    clenched = keys.key_blocks.get('い２')
    if down is None or clenched is None:
        raise ValueError('当前角色缺少咧嘴形态，无法创建露牙嫌弃嘴形')
    key = mesh.shape_key_add(name=GRIMACE_CONTROL, from_mix=False)
    for index, point in enumerate(key.data):
        basis = rest.data[index].co
        point.co = basis + (down.data[index].co-basis)*2.8 + (clenched.data[index].co-basis)*.7
    key.value = 0


def ensure_disgust_control(mesh):
    """An additive, closed-lip frown; keep all authored facial keys untouched."""
    keys = mesh.data.shape_keys
    if DISGUST_CONTROL in keys.key_blocks:
        return
    rest = keys.key_blocks[0]
    down = keys.key_blocks.get('口角下げ')
    wide = keys.key_blocks.get('口横広げ')
    if down is None or wide is None:
        raise ValueError('当前角色缺少嘴角形态，无法创建嫌弃嘴形')
    key = mesh.shape_key_add(name=DISGUST_CONTROL, from_mix=False)
    for index, point in enumerate(key.data):
        basis = rest.data[index].co
        point.co = basis + (down.data[index].co-basis)*2.5 + (wide.data[index].co-basis)*.2
    key.value = 0


def ensure_blush_control(mesh):
    """Only tint facial skin; a zero-value control leaves original shading intact."""
    keys = mesh.data.shape_keys
    if BLUSH_CONTROL not in keys.key_blocks:
        mesh.shape_key_add(name=BLUSH_CONTROL, from_mix=False).value = 0
    labels = [m.get('original_material', m.name.rsplit('|', 1)[-1].strip().split('.')[0])
              if m else '' for m in mesh.data.materials]
    skin = {i for i, label in enumerate(labels) if label in {'颜', '颜2'}}
    brows = {i for i, label in enumerate(labels) if label == '眉'}
    brow_ids = {v for p in mesh.data.polygons if p.material_index in brows for v in p.vertices}
    if not skin or not brow_ids:
        raise ValueError('当前角色没有可定位的脸部皮肤或眉毛材质，无法添加脸红')
    rest = keys.key_blocks[0].data
    brow_height = sum(rest[v].co.z for v in brow_ids)/len(brow_ids)
    brow_width = max(abs(rest[v].co.x) for v in brow_ids)
    scale = brow_width/.0596
    center_z = brow_height-.046*scale
    front = min(rest[v].co.y for p in mesh.data.polygons if p.material_index in skin for v in p.vertices)
    skin_ids = {v for p in mesh.data.polygons if p.material_index in skin for v in p.vertices}
    attribute = mesh.data.attributes.get(BLUSH_ATTRIBUTE)
    if attribute is None:
        attribute = mesh.data.attributes.new(BLUSH_ATTRIBUTE, 'FLOAT', 'POINT')
    values = []
    for i, point in enumerate(rest):
        x, y, z = point.co
        distance = ((abs(x)-.042*scale)/(.025*scale))**2+((z-center_z)/(.014*scale))**2
        front_fade = max(0, min(1, (front+.045*scale-y)/(.02*scale)))
        values.append(math.exp(-distance*1.5)*front_fade if i in skin_ids else 0)
    attribute.data.foreach_set('value', values)
    for index in skin:
        material = mesh.data.materials[index]
        if material.get('ellen_blush_key') == keys.name:
            continue
        if material.node_tree.nodes.get('Ellen Cheek Blush Tint'):
            raise ValueError('脸红材质属于另一个面部控制，请先建立独立材质副本')
        material = material.copy()
        material.name = 'Ellen Blush | '+mesh.data.materials[index].name.rsplit('|', 1)[-1].strip()
        target = next((n for n in material.node_tree.nodes if n.type == 'GROUP'
                       and n.inputs.get('DData') and n.inputs['DData'].is_linked), None)
        if not target:
            raise ValueError('脸部材质缺少 DData 输入，未改动材质')
        material['ellen_blush_key'] = keys.name
        mesh.data.materials[index] = material
        nodes, links = material.node_tree.nodes, material.node_tree.links
        source = target.inputs['DData'].links[0].from_socket
        mask = nodes.new('ShaderNodeAttribute')
        mask.name, mask.attribute_name = 'Ellen Cheek Blush Mask', BLUSH_ATTRIBUTE
        value = nodes.new('ShaderNodeValue')
        value.name, value.label = 'Ellen Cheek Blush Strength', '脸红强度'
        driver = value.outputs[0].driver_add('default_value').driver
        variable = driver.variables.new()
        variable.name, variable.type = 'blush', 'SINGLE_PROP'
        variable.targets[0].id_type, variable.targets[0].id = 'KEY', keys
        variable.targets[0].data_path = keys.key_blocks[BLUSH_CONTROL].path_from_id('value')
        driver.expression = 'blush * 0.64'
        amount = nodes.new('ShaderNodeMath')
        amount.operation, amount.name = 'MULTIPLY', 'Ellen Cheek Blush Amount'
        tint = nodes.new('ShaderNodeMixRGB')
        tint.name, tint.label = 'Ellen Cheek Blush Tint', '双颊红晕'
        tint.blend_type = 'MIX'
        tint.inputs[2].default_value = (.72, .105, .14, 1)
        links.new(mask.outputs['Fac'], amount.inputs[0])
        links.new(value.outputs[0], amount.inputs[1])
        links.new(amount.outputs[0], tint.inputs[0])
        links.new(source, tint.inputs[1])
        links.new(tint.outputs[0], target.inputs['DData'])
        mask.location, value.location = (-800, -700), (-800, -900)
        amount.location, tint.location = (-550, -700), (-200, -100)
    mesh['ellen_blush_revision'] = 1

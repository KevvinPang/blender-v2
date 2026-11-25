"""Calibrate visual speech against the character's actual lip opening."""
import json
import math
import subprocess
from pathlib import Path

VERSION = 1
VOWELS = ('あ', 'い', 'う', 'え', 'お', 'ワ')


def clamp(value, low=0.0, high=1.0):
    return min(high, max(low, value))


def ramp(value, low, high):
    t = clamp((value-low)/(high-low))
    return t*t*(3-2*t)


def calibration(mesh, rig):
    """Use central lip vertices and eye spacing in the authored rest pose."""
    fallback = {'baseline': .0135, 'deltas': {'あ': .183, 'う': .067, 'お': .209},
                'measured': False}
    if any(n not in rig.data.bones for n in ('lip05Joint','lip14Joint','Eye_L','Eye_R')):
        return fallback
    keys = mesh.data.shape_keys.key_blocks
    if any(n not in keys for n in ('あ','う','お')):
        return fallback
    transform = rig.matrix_world.inverted() @ mesh.matrix_world
    coords = [transform @ p.co for p in keys[0].data]
    ids = [min(range(len(coords)), key=lambda i: (coords[i]-rig.data.bones[n].head_local).length_squared)
           for n in ('lip05Joint','lip14Joint')]
    eye_span = (rig.data.bones['Eye_L'].head_local-rig.data.bones['Eye_R'].head_local).length
    if eye_span < 1e-5:
        return fallback
    def opening(key):
        return ((transform @ key.data[ids[0]].co).z-(transform @ key.data[ids[1]].co).z)/eye_span
    baseline = opening(keys[0])
    deltas = {n: opening(keys[n])-baseline for n in ('あ','う','お')}
    if not .02 < deltas['あ'] < .6 or any(v <= .005 for v in deltas.values()):
        return fallback
    return {'baseline': baseline, 'deltas': deltas, 'measured': True, 'lip_vertices': ids}


def mouth_weights(row, gain=1.0, model=None):
    """Keep lip closure; use image separation only with independent support."""
    shapes = row['blendshapes']
    jaw = clamp(shapes.get('jawOpen', 0))
    # A detector score is not a fraction of a character's full mouth opening.
    # This bounded response makes ordinary small syllables visible without
    # imposing a nonzero minimum opening on pauses.
    response = clamp((jaw-.008)/.52)
    opening = response*(2-response)
    model = model or {'baseline': .0135, 'deltas': {'あ': .183, 'う': .067, 'お': .209}}
    observation = row.get('mouth')
    if (isinstance(observation, dict) and observation.get('in_frame')
            and observation.get('mouth_pixels', 0) >= 12
            and .20 <= observation.get('width_to_eyes', 0) <= 1.15
            and all(math.isfinite(observation.get(k, float('nan')))
                    for k in ('aperture', 'width_to_eyes', 'dark_gap'))):
        aperture = observation['aperture']
        dark = observation['dark_gap']
        measured = max(0, (aperture*observation['width_to_eyes']-model['baseline'])/model['deltas']['あ'])
        support = ramp(dark, .012, .05)*ramp(jaw, .018, .07)
        opening = max(opening, measured*support)
        # A hallucinated lip gap during pursing must not open the character.
        if dark < .012 and jaw < .06:
            opening *= ramp(jaw, .02, .06)
        opening *= ramp(aperture, .012, .035)
    opening = clamp(opening*max(0,gain))
    pucker = clamp(shapes.get('mouthPucker',0))
    funnel = clamp(shapes.get('mouthFunnel',0))
    shares = {'う': .20*pucker, 'お': .60*funnel}
    shares['あ'] = 1-sum(shares.values())
    # Rounding changes the mouth shape, not whether it is open. Compensate
    # each vowel's different physical opening before sharing the unit budget.
    values = {n: opening*model['deltas']['あ']*share/model['deltas'][n]
              for n, share in shares.items()}
    total = sum(values.values())
    if total > 1:
        values = {n: value/total for n,value in values.items()}
    return {n: values.get(n, 0.0) for n in VOWELS}


def ensure_evidence(face, cache_path=''):
    """Upgrade legacy face observations while leaving SAM/WiLoR caches intact."""
    if face.get('mouth_features_version',0) >= VERSION:
        return face
    video = Path(face.get('video',''))
    python = Path.home()/'Desktop/小小布朗尼/tools/motion_venv/Scripts/python.exe'
    worker = Path.home()/'Desktop/sam-3d-body-main/local_app/face_worker.py'
    if not video.is_file() or not python.is_file() or not worker.is_file():
        return face
    parent = Path(cache_path).parent if cache_path else video.parent
    output = parent/'face_mouth_v1.json'
    stamp = [str(video.resolve()), video.stat().st_mtime_ns, video.stat().st_size,
             face['frames'][0]['source_frame'],face['frames'][-1]['source_frame'],face.get('sample_fps',30)]
    if output.is_file():
        cached = json.loads(output.read_text(encoding='utf-8'))
        if cached.get('_mouth_input') == stamp and cached.get('mouth_features_version',0) >= VERSION:
            return cached
    start = face['frames'][0]['time']
    end = face['frames'][-1]['time']+1/face['fps']
    process = subprocess.run([str(python),str(worker),'--video',str(video),'--output',str(output),
        '--rate',str(face.get('sample_fps',30)),'--start',str(start),'--end',str(end)],
        capture_output=True, timeout=180, creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if process.returncode:
        raise RuntimeError('补充嘴部图像证据失败；原口型未更改：'+process.stderr.decode('utf8',errors='replace')[-700:])
    cached = json.loads(output.read_text(encoding='utf-8'))
    if cached.get('mouth_features_version',0) < VERSION:
        raise RuntimeError('嘴部图像识别程序版本过旧，请更新后重试')
    cached['_mouth_input'] = stamp
    output.write_text(json.dumps(cached,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    return cached

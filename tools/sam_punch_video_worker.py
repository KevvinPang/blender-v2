"""Track a fist locally with SAM 2; prompts come from the existing video evidence."""
import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
SERVICE = Path.home()/'Desktop/sam-3d-body-main/local_app'
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'vtuber_tools/SAM2'), str(SERVICE)]
from sam2.build_sam import build_sam2_video_predictor
from sam_punch_evidence import detect, mask_signature, MASK_VERSION,approach_candidates
from video_input import open_video, SequentialFrames


def silhouette(mask, probability, agreement):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype('uint8'))
    if count < 2:
        return None
    component = 1+np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == component).astype('uint8')
    y, x = np.nonzero(mask)
    h, w = mask.shape
    if len(x) < 35 or probability < .7 or agreement is not None and agreement < .3:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    # Keep a bounded outline, not an entire image or unverified pseudo-landmarks.
    indices = np.unique(np.r_[np.linspace(0,len(contour)-1,min(64,len(contour))).astype(int),
                             contour[:,0].argmin(),contour[:,0].argmax(),
                             contour[:,1].argmin(),contour[:,1].argmax()])
    outline = contour[indices]/h
    bounds = [float(x.min()/h), float(y.min()/h), float((x.max()+1)/h), float((y.max()+1)/h)]
    return {'bounds': bounds, 'area': float(len(x)/(h*h)),
            'outline': outline.tolist(), 'clipped': [bool(x.min() <= 1), bool(y.min() <= 1),
                       bool(x.max() >= w-2), bool(y.max() >= h-2)],
            'mask_probability': probability, 'bidirectional_iou': agreement,
            'confidence_is_pose_accuracy': False}


def prompt(predictor, state, index, hand, face, height, obj=1):
    points = np.asarray(hand['points'], dtype=float)*height
    anchors = points[[5, 6, 9, 10, 13, 14, 17, 18]]
    # The fist is the whole hand, not just one visible knuckle patch. Spread
    # positive clicks across labelled fingers, thumb and the palm interior.
    clicks = [anchors.mean(0), points[5],points[9],points[13],points[17],
              points[4],points[8],points[12],points[16],points[20],
              (points[0]+points[[5,9,13,17]].mean(0))*.5]
    labels = [1]*len(clicks)
    if face:
        clicks.append(np.asarray(face['center'])*height); labels.append(0)
    # A foreshortened wrist is often localized inside the fist. A negative
    # click extrapolated from it can erase the very hand we need to track.
    margin=np.maximum(8.,np.ptp(points,axis=0)*.25)
    box = np.r_[points.min(0)-margin, points.max(0)+margin]
    predictor.add_new_points_or_box(state, frame_idx=index, obj_id=obj,
        points=np.asarray(clicks), labels=np.asarray(labels, np.int32), box=box)


def extract(body, side, diagnostics=None):
    screen = copy.deepcopy(body['screen'])
    seeds, observations = detect(screen, side)
    candidates=approach_candidates(screen,side)
    for candidate in candidates:
        if not any(s['start']<=candidate['start']<=s['end'] for s in seeds):seeds.append(candidate)
    tracking = screen.get('punch_tracking', {})
    if not tracking or len(tracking.get('frames', [])) != len(body['frames']):
        tracking = {'video': body['video'], 'version': 1,
                    'frames': [{'time': r['time'], 'hands': {}} for r in body['frames']]}
    tracking = copy.deepcopy(tracking)
    if tracking.get('input_signature')!=mask_signature(body):
        tracking['completed_sides']=[]
        tracking['approach_candidates']={};tracking['verified_approach_references']={}
        for row in tracking.get('frames',[]):
            for observation in row.get('hands',{}).values():observation.pop('silhouette',None)
    tracking.update(version=2, mask_version=MASK_VERSION, mask_side=side, mask_model='SAM 2.1 Hiera Tiny',
                    automatic_prompts=True, completed_sides=sorted(set(tracking.get('completed_sides', [])) | {side}),
                    input_signature=mask_signature(body))
    screen['punch_tracking'] = tracking
    tracking.setdefault('approach_candidates',{})[side]=candidates
    if not seeds:
        tracking['warning'] = '没有连续闭合拳头的前伸证据，未套用出拳姿势'
        return screen
    checkpoint = ROOT/'vtuber_tools/SAM2Models/sam2.1_hiera_tiny.pt'
    if not checkpoint.is_file():
        raise ValueError('缺少本地 SAM 2 拳头轮廓模型')
    if diagnostics:
        diagnostics = Path(diagnostics); diagnostics.mkdir(parents=True, exist_ok=True)
    capture = open_video(body['video']); reader = SequentialFrames(capture)
    images = []
    try:
        for row in body['frames']:
            ok, bgr = reader.read(row['source_frame'])
            if not ok:
                raise ValueError('Cannot decode punch mask frame')
            ratio = min(1., 640/max(bgr.shape[:2]))
            images.append(cv2.resize(bgr, None, fx=ratio, fy=ratio))
    finally:
        capture.release()
    times = np.asarray([r['time'] for r in body['frames']])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    context = torch.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda' else __import__('contextlib').nullcontext()
    with torch.inference_mode(), context:
        predictor = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml', str(checkpoint),
                                               device=device, apply_postprocessing=False)
        done = set()
        for seed in seeds:
            reference = seed['reference_index']
            print('PUNCH_CANDIDATE_VERIFY',side,reference,float(times[reference]),flush=True)
            begin = max(0, int(np.searchsorted(times, times[reference]-.45)))
            end = min(len(times), int(np.searchsorted(times, times[reference]+3.0, side='right')))
            # Per-event clips bound CPU/GPU memory even for long source videos.
            with tempfile.TemporaryDirectory(prefix='sam_fist_') as temporary:
                for i in range(begin, end):
                    Image.fromarray(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)).save(Path(temporary)/f'{i-begin:05d}.jpg')
                state = predictor.init_state(temporary, offload_video_to_cpu=True, offload_state_to_cpu=True)
                prompt(predictor, state, reference-begin, screen['frames'][reference]['hands'][side],
                       screen['frames'][reference].get('face'), images[reference].shape[0])
                forward, probabilities = {}, {}
                for index, _, logits in predictor.propagate_in_video(state, start_frame_idx=reference-begin):
                    values = logits[0, 0]
                    mask = (values > 0).cpu().numpy()
                    forward[index+begin] = mask
                    probabilities[index+begin] = float(torch.sigmoid(values[values > 0]).mean().item()) if mask.any() else 0.
                for index, _, logits in predictor.propagate_in_video(state, start_frame_idx=reference-begin, reverse=True):
                    values = logits[0, 0]
                    mask = (values > 0).cpu().numpy()
                    forward[index+begin] = mask
                    probabilities[index+begin] = float(torch.sigmoid(values[values > 0]).mean().item()) if mask.any() else 0.
                # Independent reverse seed after the observation gap checks that
                # a persistent mask still belongs to the same hand.
                return_indices = [i for i in range(reference+1, end) if screen['frames'][i].get('hands', {}).get(side)
                                  and observations[i] is not None and i > seed['peak_index']]
                gap_indices = [i for i in range(reference+1, end) if observations[i] is None]
                returned = next((i for i in return_indices if gap_indices and i > gap_indices[0]+3), None)
                reverse = {}
                # A fast fist may never regain landmarks on its return. A
                # second directly measured pre-approach pose can independently
                # prompt the same object; compare both tracks at the peak too.
                alternate=returned
                if alternate is None:
                    preceding=[i for i in range(reference+1,end)
                        if times[i]-times[reference]<=.30 and
                        screen['frames'][i].get('hands',{}).get(side) and
                        observations[i] is not None and observations[i].get('source')=='landmarks']
                    alternate=preceding[-1] if preceding else None
                if alternate is not None:
                    predictor.reset_state(state)
                    prompt(predictor, state, alternate-begin, screen['frames'][alternate]['hands'][side],
                           screen['frames'][alternate].get('face'), images[alternate].shape[0])
                    for index, _, logits in predictor.propagate_in_video(state, start_frame_idx=alternate-begin, reverse=True):
                        reverse[index+begin] = (logits[0, 0] > 0).cpu().numpy()
                    if returned is None:
                        for index, _, logits in predictor.propagate_in_video(state, start_frame_idx=alternate-begin):
                            reverse[index+begin] = (logits[0, 0] > 0).cpu().numpy()
                if diagnostics:
                    for i,m in reverse.items():Image.fromarray(m.astype('uint8')*255).save(diagnostics/f'reverse_{i:03d}.png')
                # A reverse tracker can follow the forearm from a bad returning
                # hand detection. It is an independent check only if it recovers
                # the original, unambiguous fist seed as well.
                reverse_valid = False
                if reference in reverse and reference in forward:
                    a, b = forward[reference], reverse[reference]
                    reverse_valid = np.logical_and(a, b).sum()/max(1, np.logical_or(a, b).sum()) >= .5
                peak_verified=False
                if not reverse_valid:
                    agrees=[]
                    for i in sorted(set(reverse)&set(forward)):
                        if i<reference or times[i]-times[reference]>1.2:continue
                        a,b=forward[i],reverse[i]
                        if np.logical_and(a,b).sum()/max(1,np.logical_or(a,b).sum())>=.65:agrees.append(i)
                    peak_verified=any(b==a+1 and c==b+1 for a,b,c in zip(agrees,agrees[1:],agrees[2:]))
                tracking['reverse_check'] = 'verified' if reverse_valid else 'approach-verified' if peak_verified else 'unavailable'
                if reverse_valid or peak_verified:tracking.setdefault('verified_approach_references',{}).setdefault(side,[]).append(reference)
                tracking.setdefault('seed_checks',{}).setdefault(side,[]).append({
                    'reference':reference,'alternate':alternate,'alternate_is_return':returned is not None,
                    'reference_verified':bool(reverse_valid),'approach_verified':bool(peak_verified)})
                for i, mask in forward.items():
                    other = reverse.get(i)
                    agreement = float(np.logical_and(mask, other).sum()/max(1, np.logical_or(mask, other).sum())) if reverse_valid and other is not None else None
                    shape = silhouette(mask, probabilities[i], agreement)
                    if shape is None:
                        continue
                    direct = observations[i]
                    obs = dict(direct) if direct else {'source': 'video-mask', 'confidence': .8, 'closed': 1.}
                    obs['silhouette'] = shape
                    # Used only to associate tracks/detect events. The solver uses
                    # the outline and does not treat this centroid as a 3-D joint.
                    x0,y0,x1,y1 = shape['bounds']
                    obs['center'] = [(x0+x1)*.5, (y0+y1)*.5]
                    obs['radius'] = float(np.sqrt(shape['area']))
                    tracking['frames'][i]['hands'][side] = obs
                    done.add(i)
                    if diagnostics:
                        Image.fromarray(mask.astype('uint8')*255).save(diagnostics/f'mask_{i:03d}.png')
                del state
        tracking['mask_samples'] = len(done)
    return screen


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--body', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--side', required=True, choices=('L', 'R')); parser.add_argument('--diagnostics')
    args = parser.parse_args()
    body = json.loads(Path(args.body).read_text(encoding='utf8'))
    result = extract(body, args.side, args.diagnostics)
    Path(args.output).write_text(json.dumps(result, allow_nan=False), encoding='utf8')
    print('PUNCH_VIDEO_EVIDENCE_SAVED', result['punch_tracking'].get('mask_samples', 0), flush=True)

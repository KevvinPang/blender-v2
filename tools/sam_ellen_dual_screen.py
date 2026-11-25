"""Two independent arm chains in one calibrated camera/body coordinate frame.

Keep anatomical L/R identities from the detector, even when the hands cross.
Never move the camera/body a second time to satisfy the other hand, or force
contact/depth from a 2D overlap. Each hand retains the existing dropout handling.
"""
import numpy as np


def validate_dual(data, space='FRAME', coordinated=False):
    from sam_ellen_screen import validate_screen, MissingScreenObservations
    if space != 'FRAME' or data.get('screen', {}).get('features_version', 1) < 2:
        raise ValueError('双手画面对齐需要原视频画面关键点，请选择原视频画面位置并重新识别')
    available = False
    for side in ('L', 'R'):
        try:
            validate_screen(data, side, space, coordinated)
        except MissingScreenObservations:
            continue
        available = True
    if not available:
        raise MissingScreenObservations('没有足够的双手或脸部画面关键点，保留现有动作')
    return data['screen']


def correct_dual(scratch, rig, channels, data, options, cameras, report):
    from sam_ellen_screen_fit import correct_landmarks, landmark_observations
    validate_dual(data, options.get('space', 'FRAME'), options.get('coordinated_body', False))
    if not options.get('landmarks'):
        raise ValueError('双手画面对齐需要开启指尖 / 掌心联合对齐')
    reports = {}
    for index, side in enumerate(('L', 'R')):
        rows = []
        # No usable image evidence: do not smooth or solve this arm at all.
        observations = landmark_observations(data['screen'], side, space='FRAME')
        if any(o is not None for o in observations):
            for done, total in correct_landmarks(scratch, rig, channels, data,
                    dict(options, side=side, dual=True), cameras, rows):
                yield index+done/total, 2
        else:
            from sam_ellen_arm_visibility import apply_visibility
            apply_visibility(scratch,rig,channels,data,side)
            rows = [{'time': r['time'], 'active': False, 'reason': 'no-hand-observations'}
                    for r in data['frames']]
            yield index+1, 2
        reports[side] = rows
    for i, row in enumerate(data['frames']):
        hands = {side: reports[side][i] for side in ('L', 'R')}
        report.append({'time': row['time'], 'active': any(h['active'] for h in hands.values()),
                       'model': 'dual-hand-shared-frame-v1', 'hands': hands})


def measure_dual(scratch, rig, channels, data, options, cameras, report, contact=None, punch=None):
    from sam_ellen_screen_fit import measure_final
    # Measure both sides after both chains and all optional overrides are solved.
    for index, side in enumerate(('L', 'R')):
        rows = [r['hands'][side] for r in report]
        for done, total in measure_final(scratch, rig, channels, data,
                dict(options, side=side, dual=True), cameras, rows, contact, punch):
            yield index+done/total, 2
    for row in report:
        left, right = (row['hands'][s] for s in ('L', 'R'))
        if not all(h.get('target_center_xy') is not None for h in (left, right)):
            continue
        source = np.asarray(left['target_center_xy'])-right['target_center_xy']
        before = np.asarray(left['before_center_xy'])-right['before_center_xy']
        after = np.asarray(left['final_center_xy'])-right['final_center_xy']
        height = options.get('height', 1920)
        row['pair'] = {
            'target_distance_pixels': float(np.linalg.norm(source)*height),
            'final_distance_pixels': float(np.linalg.norm(after)*height),
            'before_relative_error_pixels': float(np.linalg.norm(before-source)*height),
            'after_relative_error_pixels': float(np.linalg.norm(after-source)*height),
        }

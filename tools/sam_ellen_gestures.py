"""Explicit, timed hand-pose assistance; original capture data stays immutable."""
import copy
import math
from mathutils import Quaternion, Vector


def interval_weight(time, start, end, fade=.2):
    if not all(math.isfinite(v) for v in (time, start, end, fade)) or start < 0 or end <= start or fade < 0:
        raise ValueError('Invalid gesture interval')
    if time <= start or time >= end:
        return 0.
    if not fade:
        return 1.
    value = min(1., (time-start)/fade, (end-time)/fade)
    return value*value*(3-2*value)


def apply_gesture(data, options):
    if not options or options['mode'] == 'OFF':
        return data
    if options['mode'] != 'MIDDLE_ONLY' or options['side'] not in ('L', 'R'):
        raise ValueError('Unsupported gesture')
    if not math.isfinite(options['strength']) or not 0 <= options['strength'] <= 1:
        raise ValueError('Invalid gesture strength')
    interval_weight(0, options['start'], options['end'])
    result = copy.deepcopy(data)
    offset = 36 if options['side'] == 'L' else 0
    for row in result['frames']:
        t = row['time']-result['frames'][0]['time']
        weight = interval_weight(t, options['start'], options['end'])*options['strength']
        if not weight:
            continue
        points = [Vector(p) for p in row['positions']]
        forward = (points[52+offset]-points[42+offset]).normalized()
        # Use the captured ring's fold direction to determine the palm's inside.
        across = (points[56+offset]-points[44+offset]).normalized()
        normal = across.cross(forward).normalized()
        fold = points[51+offset]-points[48+offset]
        if normal.dot(fold) < 0:
            normal.negate()
        # Only correct the unwanted index extension. Keep the captured middle,
        # thumb and remaining fingers, including their real pointing direction.
        for first, angles in ((56, (45, 125, 165)),):
            first += offset
            lengths = [(points[first+i+1]-points[first+i]).length for i in range(3)]
            original = [(points[first+i+1]-points[first+i]).normalized() for i in range(3)]
            pos = points[first].copy()
            for i, angle in enumerate(angles):
                radians = math.radians(angle)
                desired = forward*math.cos(radians)+normal*math.sin(radians)
                # Quaternion interpolation avoids zero vectors during a full fold.
                direction = original[i].rotation_difference(desired)
                direction = Quaternion().slerp(direction, weight) @ original[i]
                pos += direction.normalized()*lengths[i]
                row['positions'][first+i+1] = list(pos)
    result['gesture_assistance'] = dict(options)
    return result

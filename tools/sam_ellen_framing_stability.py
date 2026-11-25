"""Keep an approaching fist from driving an impact-like camera excursion."""
import numpy as np


def stabilize_punch_framing(data, values):
    """Bridge short punch intervals with the surrounding smooth camera path.

    Hand geometry owns fist position and scale. Face estimates during the
    approach, occlusion, and reacquisition must not create a camera impulse.
    Authored camera motion is not an input here: values are generated framing
    corrections only. Unrelated portions of the correction stay identical.
    """
    from sam_punch_evidence import detect
    from sam_ellen_temporal import trajectory
    values=np.asarray(values,dtype=float)
    times=np.array([r['time'] for r in data['frames']],dtype=float)
    if len(times)<4:return values.copy(),np.zeros(len(times))
    relative=times-times[0];mask=np.zeros(len(times))
    for side in ('L','R'):
        segments,_=detect(data.get('screen',{}),side)
        for segment in segments:
            # detect uses screen-relative times; no source or timeline offset.
            start,end=segment['start'],segment['end']
            left=np.clip((relative-(start-.18))/.18,0,1)
            right=np.clip(((end+.18)-relative)/.18,0,1)
            blend=np.minimum(left,right);blend=blend*blend*(3-2*blend)
            mask=np.maximum(mask,blend)
    if not mask.any():return values.copy(),mask
    # Zero weight in the event joins reliable samples on both sides, rather
    # than smoothing a spurious face reacquisition displacement into the shot.
    weight=1-mask
    if np.count_nonzero(weight>.99)<3:return values.copy(),np.zeros(len(times))
    stable=trajectory(times,values,.18,weight)
    return values+(stable-values)*mask[:,None],mask

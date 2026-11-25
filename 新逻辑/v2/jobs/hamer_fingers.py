"""Estimate MANO finger geometry for detected hands in the canonical video."""
from pathlib import Path
import inspect
import sys
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[2] / 'source'
HAWOR = SOURCE / 'HaWoR'
sys.path[:0] = [str(SOURCE / 'FlowHMR-Windows'), str(HAWOR), str(HAWOR / 'runtime_deps')]
from ultralytics import YOLO
from flowhmr.utils.runtime_hamer_worker import run
from lib.models.mano_wrapper import MANO


def estimate(video, output):
    # Legacy MANO pickle and Ultralytics checkpoint use the installed WiLoR environment.
    inspect.getargspec = inspect.getfullargspec
    for key, value in dict(int=int, bool=bool, float=float, complex=complex, object=object, str=str, unicode=str).items():
        if key not in np.__dict__:
            setattr(np, key, value)
    detector = YOLO(str(HAWOR / 'weights/external/detector.pt'))
    capture = cv2.VideoCapture(str(video))
    boxes = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            detected = detector.track(frame, conf=.2, persist=True, device=0, verbose=False)[0].boxes
            row = np.full((2, 4), np.nan, np.float32)
            for side in (0, 1):
                matches = detected[detected.cls == side]
                if len(matches):
                    row[side] = matches.xyxy[matches.conf.argmax()].cpu().numpy()
            boxes.append(row)
            if len(boxes) % 60 == 0:
                print(f'V2_PROGRESS HaMeR 手框 {len(boxes)} 帧', flush=True)
    finally:
        capture.release()
    del detector
    if not boxes:
        raise ValueError(f'Cannot decode {video}')
    with TemporaryDirectory(dir=output.parent) as temp:
        directory = Path(temp)
        np.save(directory/'boxes.npy', boxes)
        run(video, directory/'boxes.npy', 0, 0, directory/'rotations.npz', 1)
        with np.load(directory/'rotations.npz') as result:
            rotations, valid = result['rotations'], result['valid']
    points = np.full((len(boxes), 2, 21, 3), np.nan, np.float32)
    with torch.inference_mode():
        for side, model_path in enumerate(('_DATA/data_left/mano_left', '_DATA/data/mano')):
            ids = np.flatnonzero(valid[:, side])
            if not len(ids):
                continue
            mano = MANO(model_path=str(HAWOR/model_path), is_rhand=bool(side), use_pca=False)
            for start in range(0, len(ids), 128):
                frames = ids[start:start+128]
                pose = torch.from_numpy(rotations[frames, side])
                points[frames, side] = mano(hand_pose=pose, global_orient=torch.eye(3).expand(len(frames),1,3,3),
                                            pose2rot=False).joints.numpy()
    np.savez_compressed(output, points=points, valid=valid)
    print('V2_PROGRESS HaMeR 左右手有效帧', valid.sum(axis=0).tolist(), flush=True)


if __name__ == '__main__':
    import os
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    estimate(Path(sys.argv[1]), Path(sys.argv[2]))

"""Retain SAM image observations for independent retarget/projection checks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import configure_imports
configure_imports()
import argparse
import numpy as np
import cv2
import torch
from flowhmr.utils.runtime_sam_features import build_sam3d_extractor

def main():
    p=argparse.ArgumentParser()
    p.add_argument('job'); p.add_argument('--stride', type=int, default=3)
    a=p.parse_args(); job=Path(a.job)
    camera_path=next(job.glob('*static_camera.npz'),None)
    if camera_path is None:
        camera_path=next(job.glob('*vggt_camera.npz'))
    camera=np.load(camera_path)
    boxes=np.load(next(job.glob('*bbox.npz')))['bbox']
    cap=cv2.VideoCapture(str(next(job.glob('*30fps.mp4'))))
    extractor=build_sam3d_extractor(torch.device('cuda'),frame_batch_size=1,
                                   use_gpu_preprocess=False,use_compile=False)
    if len(boxes)<2:raise ValueError('At least two video frames are required')
    ids=sorted(set(range(0,len(boxes),max(1,a.stride)))|{len(boxes)-1})
    uv=[]; xyz=[]; ts=[]
    for i in ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read()
        if not ok: raise RuntimeError(f'Cannot read frame {i}')
        with torch.no_grad():
            _,out=extractor.extract_frame(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),bboxes=boxes[i:i+1],
                  cam_int=torch.from_numpy(camera['K'][i:i+1]).float())
        if len(out)!=1: raise RuntimeError(f'Expected one person at {i}')
        o=out[0]
        uv.append(np.asarray(o['pred_keypoints_2d']))
        xyz.append(np.asarray(o['pred_keypoints_3d']))
        ts.append(np.asarray(o['pred_cam_t']))
        print(f'OBS {i+1}/{len(boxes)}',flush=True)
    cap.release()
    np.savez_compressed(job/'sam_observations.npz',frame_indices=ids,
                        keypoints_2d=uv,keypoints_3d=xyz,cam_t=ts)
    print('SAVED',job/'sam_observations.npz',flush=True)
if __name__=='__main__': main()

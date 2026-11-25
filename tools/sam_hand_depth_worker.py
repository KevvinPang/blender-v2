"""Supplement existing image observations with measured hand depth only."""
import json,sys,hashlib
from pathlib import Path


def signature(body):
    video=Path(body['video']);st=video.stat()
    value=[str(video.resolve()),st.st_size,st.st_mtime_ns,
           [(r['source_frame'],r['time']) for r in body['frames']],
           [r.get('hands',{}) for r in body['screen']['frames']]]
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def supplement(body):
    import cv2,mediapipe as mp,numpy as np
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker,HandLandmarkerOptions,RunningMode
    worker=Path.home()/'Desktop/sam-3d-body-main/local_app';sys.path.insert(0,str(worker))
    from video_input import open_video,SequentialFrames
    options=HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(worker/'models/hand_landmarker.task')),
        running_mode=RunningMode.VIDEO,num_hands=2,min_hand_detection_confidence=.65,
        min_hand_presence_confidence=.65,min_tracking_confidence=.65)
    cap=open_video(body['video']);reader=SequentialFrames(cap);rows=[]
    try:
        with HandLandmarker.create_from_options(options) as detector:
            for i,(sample,observed) in enumerate(zip(body['frames'],body['screen']['frames'])):
                ok,bgr=reader.read(sample['source_frame'])
                if not ok:raise ValueError('Cannot decode hand orientation evidence')
                h,w=bgr.shape[:2];scale=min(1.,1280/max(h,w));bgr=cv2.resize(bgr,None,fx=scale,fy=scale)
                result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)),round(sample['time']*1000))
                found={};used=set()
                for side,hand in observed.get('hands',{}).items():
                    expected=np.array(hand['points']);pairs=[]
                    for j,points in enumerate(result.hand_landmarks):
                        if j in used:continue
                        p=np.array([[v.x*w/h,v.y] for v in points])
                        distance=float(np.median(np.linalg.norm(p-expected,axis=1)))
                        if distance<.055:pairs.append((distance,j))
                    if pairs:
                        distance,j=min(pairs);used.add(j)
                        p=np.array([[v.x,v.y,v.z] for v in result.hand_world_landmarks[j]])
                        if np.isfinite(p).all():found[side]={'points':p.tolist(),'image_match_error':distance}
                rows.append(found)
                if (i+1)%30==0:print('HAND_DEPTH',i+1,len(body['frames']),flush=True)
    finally:cap.release()
    screen=dict(body['screen']);screen['hand_pose_depth']={'version':1,'video':body['video'],
        'input_signature':signature(body),'coordinates':'MediaPipe camera: x right, y down, z away',
        'frames':rows}
    return screen


if __name__=='__main__':
    body=json.loads(Path(sys.argv[1]).read_text(encoding='utf8'))
    Path(sys.argv[2]).write_text(json.dumps(supplement(body),allow_nan=False),encoding='utf8')

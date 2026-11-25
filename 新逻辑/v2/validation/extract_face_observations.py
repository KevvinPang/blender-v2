"""Video-aligned face rotation, iris landmarks and expression observations."""
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[3]
extensions = root/'vtuber_tools/apps/Blender52/blender-5.2.2-windows-x64/portable/extensions'
sys.path.insert(0, str(next((extensions/'.local/lib').glob('python*/site-packages'))))
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

job = Path(sys.argv[-1])
count = len(np.load(next(job.glob('*_seed0_origin.npz')))['poses'])
model = extensions/'user_default/media_to_pose/assets/models/face_landmarker.task'
options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_buffer=model.read_bytes()),
    running_mode=vision.RunningMode.VIDEO, num_faces=1,
    output_face_blendshapes=True, output_facial_transformation_matrixes=True)
cap = cv2.VideoCapture(str(next(job.glob('*30fps.mp4'))))
landmarks = np.full((count, 478, 3), np.nan)
matrices = np.full((count, 4, 4), np.nan)
valid = np.zeros(count, bool)
shapes = []
with vision.FaceLandmarker.create_from_options(options) as detector:
    for i in range(count):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f'Missing video frame {i}')
        result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), round(i*1000/cap.get(cv2.CAP_PROP_FPS)))
        row = {}
        if result.face_landmarks and result.facial_transformation_matrixes:
            landmarks[i] = [[p.x, p.y, p.z] for p in result.face_landmarks[0]]
            matrices[i] = result.facial_transformation_matrixes[0]
            row = {c.category_name: c.score for c in result.face_blendshapes[0]}
            valid[i] = True
        shapes.append(row)
cap.release()
names = sorted({k for row in shapes for k in row})
np.savez_compressed(job/'face_observations.npz', frame_indices=np.arange(count),
    landmarks=landmarks, matrices=matrices, valid=valid,
    blendshape_names=names, blendshapes=[[r.get(n, np.nan) for n in names] for r in shapes])
print(f'FACE_OBSERVATIONS_SAVED {valid.sum()}/{count}', flush=True)

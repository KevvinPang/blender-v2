import sys,argparse
from pathlib import Path
root=Path(__file__).resolve().parents[3]
extensions=root/'vtuber_tools/apps/Blender52/blender-5.2.2-windows-x64/portable/extensions'
site=next((extensions/'.local/lib').glob('python*/site-packages'))
sys.path.insert(0,str(site))
import cv2,numpy as np,mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
parser=argparse.ArgumentParser()
parser.add_argument('job');parser.add_argument('output',nargs='?')
parser.add_argument('--mode',choices=['image','video'],default='image')
parser.add_argument('--single-pass',action='store_true',help='Diagnostic: disable crop/mirror consensus')
args=parser.parse_args()
job=Path(args.job);video=next(job.glob('*30fps.mp4'))
model=extensions/'user_default/media_to_pose/assets/models/hand_landmarker.task'
options=vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_buffer=model.read_bytes()),
    # VIDEO mode can retain an incorrect finger configuration during fast
    # re-entry. Independent detections plus offline track fusion are default.
    running_mode=vision.RunningMode.IMAGE if args.mode=='image' else vision.RunningMode.VIDEO,
    num_hands=2,min_hand_detection_confidence=.25,
    min_hand_presence_confidence=.25,min_tracking_confidence=.25)
cap=cv2.VideoCapture(str(video));rows=[];world=[];labels=[];scores=[]
fps=cap.get(cv2.CAP_PROP_FPS) or 30.
frame_count=len(np.load(next(job.glob('*bbox.npz')))['bbox'])
with vision.HandLandmarker.create_from_options(options) as det:
    for i in range(frame_count):
        ok,im=cap.read()
        if not ok:raise RuntimeError(i)
        rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        timestamp_ms=round(i*1000./fps)
        image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
        out=det.detect(image) if args.mode=='image' else det.detect_for_video(image,timestamp_ms)
        candidates=[]
        def collect(result,x0,y0,cw,ch,flip=False):
            for h in range(len(result.hand_landmarks)):
                p=np.array([[v.x,v.y,v.z] for v in result.hand_landmarks[h]])
                q=np.array([[v.x,v.y,v.z] for v in result.hand_world_landmarks[h]])
                label=result.handedness[h][0].category_name
                if flip:
                    p[:,0]=1-p[:,0];q[:,0]*=-1
                    label='Left' if label=='Right' else 'Right'
                p[:,0]=(p[:,0]*cw+x0)/rgb.shape[1];p[:,1]=(p[:,1]*ch+y0)/rgb.shape[0];p[:,2]*=cw/rgb.shape[1]
                candidates.append((p,q,result.handedness[h][0].score,label))
        H,W=rgb.shape[:2];collect(out,0,0,W,H)
        if args.mode=='image' and not args.single_pass:
            mirrored=det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb[:,::-1])))
            collect(mirrored,0,0,W,H,True)
            # Independent ROI observations help with fingers mislabelled in
            # the full image. Margins include extended fingers above the palm.
            boxes=[]
            # A portrait close-up can hide an extended fingertip above an
            # incorrectly detected fist. Test each visible half with overlap,
            # independently of the erroneous landmark bounding box.
            centers=[c[0][[0,5,9,13,17],:2].mean(0) for c in candidates]
            for right in (False,True):
                if not any((c[0]>=.5)==right for c in centers):continue
                x0=int(W*.39) if right else 0;x1=W if right else int(W*.61)
                y0=int(H*.35);crop=rgb[y0:H,x0:x1]
                for flip in (False,True):
                    pixels=np.ascontiguousarray(crop[:,::-1] if flip else crop)
                    result=det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=pixels))
                    collect(result,x0,y0,x1-x0,H-y0,flip)
            for candidate in candidates[:]:
                xy=candidate[0][:,:2]*[W,H];center=(xy.min(0)+xy.max(0))*.5
                extent=np.maximum(np.ptp(xy,axis=0)*1.6,[W*.55,H*.65])
                lo=np.maximum(0,center-extent*.5).astype(int);hi=np.minimum([W,H],center+extent*.5).astype(int)
                if any(np.linalg.norm(center-c)<W*.3 for c in boxes):continue
                boxes.append(center);x0,y0=lo;x1,y1=hi;crop=rgb[y0:y1,x0:x1]
                for flip in (False,True):
                    pixels=np.ascontiguousarray(crop[:,::-1] if flip else crop)
                    result=det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=pixels))
                    collect(result,x0,y0,x1-x0,y1-y0,flip)
            # Group spatially before consensus, so the two hands never vote
            # against one another. The handedness score is not joint accuracy.
            groups=[]
            for candidate in candidates:
                center=candidate[0][[0,5,9,13,17],:2].mean(0)
                group=next((g for g in groups if np.linalg.norm(center-g[0][0][[0,5,9,13,17],:2].mean(0))<.22),None)
                if group is None:groups.append([candidate])
                else:group.append(candidate)
            chosen=[]
            for group in groups:
                costs=[]
                for p,q,score,label in group:
                    costs.append(sum(np.mean(np.linalg.norm((p[:,:2]-other[0][:,:2])*[W,H],axis=1)) for other in group))
                chosen.append(group[int(np.argmin(costs))])
            candidates=sorted(chosen,key=lambda c:c[2],reverse=True)[:2]
        row=np.full((2,21,3),np.nan);w=row.copy();ss=np.zeros(2);ll=['','']
        for h,(p,q,score,label) in enumerate(candidates[:2]):
            row[h]=p;w[h]=q;ss[h]=score;ll[h]=label
        rows.append(row);world.append(w);scores.append(ss);labels.append(ll)
        if i%60==0:print(f'V2_PROGRESS 手部多视图核对 {i+1}/{frame_count}',flush=True)
cap.release()
output=Path(args.output) if args.output else job/'hand_observations.npz'
from hand_image_flow import measure,edge_witness
print('V2_PROGRESS 校验前后帧手部运动方向',flush=True)
np.savez_compressed(output,normalized_landmarks=rows,world_landmarks=world,scores=scores,labels=labels,
                    **measure(video,frame_count),**edge_witness(video,np.asarray(rows),scores))
print('HAND_OBSERVATIONS_SAVED',sum(np.any(s>0) for s in scores))

"""Behavioral checks: camera shake vs actor motion, timing and weak tracking."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'camera'))
from joint_motion import smooth,stabilize_relative,transform_points
from prepare_joint_camera import regularize_camera
from scipy.spatial.transform import Rotation

fps=30.;N=240;t=np.arange(N)/fps
H=np.repeat(np.eye(3)[None],N,axis=0)
H[:,0,2]=8*np.sin(t*21)+t*12
H[:,1,2]=4*np.sin(t*29)
still=np.repeat([[360.,650.,1.]],N,axis=0)
shaken=still.copy();shaken[:,:2]=transform_points(H,still[:,:2])
out,_=stabilize_relative(shaken,fps,H,np.ones(N))
assert np.allclose(out,shaken,atol=1e-8),'Camera shake was converted into actor compensation'
noisy=shaken.copy();noisy[:,0]+=3*np.sin(np.arange(N)*2.5)
out,_=stabilize_relative(noisy,fps,H,np.ones(N))
assert np.std((out-shaken)[15:-15,0])<.2,'Static actor jitter survived'
linear=np.c_[t*80+200,t*-10+500,t*.01+1]
assert np.allclose(smooth(linear,fps,.11),linear,atol=1e-9)
pulse=np.exp(-.5*((t-3)/.16)**2)*180
fast=still.copy();fast[:,0]+=pulse;fast[:,:2]=transform_points(H,fast[:,:2])
out,_=stabilize_relative(fast,fps,H,np.ones(N))
res=transform_points(np.linalg.inv(H),out[:,:2])[:,0]-360
assert abs(np.argmax(res)-np.argmax(pulse))<=1,'Fast motion gained phase lag'
assert res.max()>.85*pulse.max(),'Fast motion peak was flattened'
weak=np.zeros(N);weak[60:160]=1
out,_=stabilize_relative(noisy,fps,H,weak)
assert np.isfinite(out).all()
K=np.repeat(np.array([[1100.,0,360],[0,1100.,640],[0,0,1]])[None],N,axis=0)
rv=np.zeros((N,3));rv[:,1]=.03*t+.003*np.sin(np.arange(N)*2)
C=np.repeat(np.eye(4)[None],N,axis=0);C[:,:3,:3]=Rotation.from_rotvec(rv).as_matrix()
rt,k,report=regularize_camera(np.linalg.inv(C),K,None,None,fps)
got=Rotation.from_matrix(np.linalg.inv(rt)[:,:3,:3]).as_rotvec()
assert np.std((got[:,1]-.03*t)[15:-15])<.0002
assert np.max(abs(k-K))<1e-8
assert np.max(abs(np.linalg.det(np.linalg.inv(rt)[:,:3,:3])-1))<1e-8
shake=.004*np.sin(t*18)
C[:,:3,:3]=Rotation.from_rotvec(np.c_[shake,np.zeros((N,2))]).as_matrix()
RT=np.linalg.inv(C)
H=K@RT[:,:3,:3]@np.linalg.inv(K)
rt,k,_=regularize_camera(RT,K,H,np.ones(N),fps)
error=(Rotation.from_matrix(np.linalg.inv(rt)[:,:3,:3]).inv()*Rotation.from_matrix(C[:,:3,:3])).magnitude()
assert error.max()<1e-5,'Supported real camera shake was smoothed away or reversed'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'blender_bridge'))
from actor_observations import dense_anchors
import tempfile
with tempfile.TemporaryDirectory() as tmp:
    landmarks=np.zeros((N,478,3));landmarks[:,[362,263],:2]=[.6,.4];landmarks[:,[33,133],:2]=[.4,.4];landmarks[:,1,:2]=[.5,.5]
    body=np.zeros((N,70,2));body[:,1]=[432,512];body[:,2]=[288,512];body[:,0]=[360,640]
    body[100:106,1,0]-=35  # isolated coarse body detector eye-span collapse
    landmarks[120:,468:,0]=.9  # gaze/iris changes must never translate the torso
    path=Path(tmp)/'face.npz';np.savez(path,landmarks=landmarks,valid=np.ones(N,bool))
    dense,_=dense_anchors(path,body,720,1280)
    assert np.allclose(np.linalg.norm(dense[:,1]-dense[:,2],axis=1),144)
    assert np.allclose(dense[:,[1,2]].mean(1),[360,512])
print('PASS: camera shake separation, static actor jitter, fast motion timing/amplitude, missing background, camera SO(3) and focal invariance')
print('PASS: coarse eye-span failure rejected; iris motion does not drive body')

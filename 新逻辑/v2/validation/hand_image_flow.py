"""Compact bidirectional pixel displacements, in normalized image units."""
import cv2
import numpy as np


def measure(video, count):
    cap = cv2.VideoCapture(str(video))
    tracker = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    previous = None
    forward, backward = [], []
    try:
        for frame in range(count):
            ok, image = cap.read()
            if not ok:
                raise ValueError(f'Missing hand tracking video frame {frame+1}')
            height, width = image.shape[:2]
            scale = min(1., 480/max(height, width))
            gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None,
                              fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            h, w = gray.shape
            if previous is not None:
                for first, second, rows in ((previous, gray, forward), (gray, previous, backward)):
                    flow = tracker.calc(first, second, None) / [w, h]
                    rows.append(cv2.resize(flow, (max(4,w//4), max(4,h//4))).astype(np.float16))
            previous = gray
    finally:
        cap.release()
    return dict(flow_forward=np.asarray(forward), flow_backward=np.asarray(backward))


def edge_witness(video, landmarks, scores):
    """Find a shrinking contour against the revealed background of a single-hand exit.

    For each short exposure store the image edge and the innermost visible
    contour coordinate. Reject regions without a coherent retreat across frames.
    """
    T=len(scores)
    edge=np.full(T,-1,np.int8)
    boundary=np.full(T,np.nan)
    hands=(np.asarray(scores)>=.55)
    ends=np.flatnonzero((hands.sum(1)==1)&~np.r_[hands[1:].any(1),True])
    cap=cv2.VideoCapture(str(video))
    try:
        for end in ends:
            if end<5 or end+6>=T or hands[end+1:end+7].any():continue
            start=end-4
            samples=[]
            for t in range(start,end+1):
                choices=landmarks[t,hands[t],:,:2]
                if not len(choices):break
                if len(choices)>1:
                    centres=np.median(choices,axis=1)
                    if not samples or np.linalg.norm(centres[1]-centres[0])>.23:break
                    distance=np.median(np.linalg.norm(choices-samples[-1],axis=2),axis=1)
                    samples.append(choices[np.argmin(distance)])
                else:samples.append(choices[0])
            if len(samples)!=end-start+1:continue
            hand=np.asarray(samples)
            if not np.isfinite(hand).all():continue
            center=np.median(hand[-2:,:],axis=(0,1))
            movement=np.median(hand[-2]-hand[0],axis=0)
            distance=np.array([center[0],1-center[0],center[1],1-center[1]])
            extreme=np.array([hand[-1,:,0].min(),1-hand[-1,:,0].max(),hand[-1,:,1].min(),1-hand[-1,:,1].max()])
            travel=np.array([-movement[0],movement[0],-movement[1],movement[1]])
            votes=np.where(((distance<.24)|(extreme<.07))&(travel>.02),travel/np.maximum(.04,distance),-1)
            if votes.max()<0:continue
            side=int(np.argmax(votes));axis=side//2;sign=(-1,1)[side%2]
            cap.set(cv2.CAP_PROP_POS_FRAMES,start);frames=[]
            for _ in range(start,end+7):
                ok,im=cap.read()
                if not ok:break
                frames.append(cv2.cvtColor(cv2.resize(im,(max(16,round(im.shape[1]*480/im.shape[0])),480)),cv2.COLOR_BGR2LAB).astype(np.float32))
            if len(frames)!=end+7-start:continue
            h,w=frames[0].shape[:2];size=np.array([w,h])
            lo=np.maximum(0,np.min(hand,axis=(0,1))-.075);hi=np.minimum(1,np.max(hand,axis=(0,1))+.075)
            if sign>0:hi[axis]=1
            else:lo[axis]=0
            x0,y0=(lo*size).astype(int);x1,y1=(hi*size).astype(int)
            if x1-x0<5 or y1-y0<5:continue
            bg=np.median(frames[-2:],axis=0)[y0:y1,x0:x1]
            values=[];areas=[]
            for im in frames[:-2]:
                delta=np.linalg.norm((im[y0:y1,x0:x1]-bg)*[.55,1,1],axis=2)
                mask=cv2.morphologyEx((delta>16).astype('uint8'),cv2.MORPH_OPEN,np.ones((3,3),'uint8'))
                n,_,stats,_=cv2.connectedComponentsWithStats(mask)
                candidates=[]
                for x,y,bw,bh,area in stats[1:]:
                    a,b=((x+x0,x+x0+bw) if axis==0 else (y+y0,y+y0+bh))
                    if (b>=size[axis]-.022*size[axis] if sign>0 else a<=.022*size[axis]) and area>h*w*.00025:
                        candidates.append((area,(a if sign>0 else b)/size[axis]))
                area,pos=max(candidates,default=(0,float(side%2)))
                values.append(pos);areas.append(area)
            delta=sign*np.diff(values)
            if (areas[0]<h*w*.009 or max(areas[-2:])>areas[0]*.1
                    or sum(delta>.012)<2 or np.min(delta)<-.07):continue
            for k,(area,pos) in enumerate(zip(areas,values)):
                edge[start+k]=side;boundary[start+k]=pos
    finally:
        cap.release()
    return dict(contour_edge=edge,contour_boundary=boundary)

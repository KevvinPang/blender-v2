"""Recover only short, verified partial-hand entry tracks from source pixels."""
import json,sys
from pathlib import Path
import cv2,numpy as np
from sam_ellen_entry import VERSION,signature,merge_wilor,long_entries,hand_at,incoming_edge

def recover(data):
    cap=cv2.VideoCapture(data['video']);cache={};colors={};height=640;aspect=data['screen']['aspect'];width=round(height*aspect)
    def image(i):
        if i not in cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES,data['frames'][i]['source_frame']);ok,im=cap.read()
            if not ok:raise ValueError('Cannot decode entry source frame '+str(i))
            colors[i]=cv2.resize(im,(width,height));cache[i]=cv2.cvtColor(colors[i],cv2.COLOR_BGR2GRAY)
        return cache[i]
    record={'version':VERSION,'input_signature':signature(data),'sides':{}}
    try:
        for side in ('L','R'):
            episodes=[]
            for start in long_entries(data,side):
                if not start:continue
                hand=hand_at(data['screen'],start,side);points=np.asarray(hand['points'])*height
                mask=np.zeros((height,width),np.uint8)
                cv2.fillConvexPoly(mask,cv2.convexHull(np.round(points).astype(np.int32)),255)
                mask=cv2.erode(mask,np.ones((5,5),np.uint8))
                features=cv2.goodFeaturesToTrack(image(start),maxCorners=100,qualityLevel=.005,minDistance=3,mask=mask)
                tracked={};reason='no-texture'
                if features is not None:
                    features=features.reshape(-1,2);reason='tracking-ended'
                    for i in range(start-1,-1,-1):
                        if data['frames'][start]['time']-data['frames'][i]['time']>.24 or hand_at(data['screen'],i,side):break
                        a,b=image(i+1),image(i);p=features.astype(np.float32).reshape(-1,1,2)
                        q,status,error=cv2.calcOpticalFlowPyrLK(a,b,p,None,winSize=(35,35),maxLevel=4,criteria=(3,40,.001))
                        if q is None or status is None:reason='flow-unavailable';break
                        back,st2,_=cv2.calcOpticalFlowPyrLK(b,a,q,None,winSize=(35,35),maxLevel=4,criteria=(3,40,.001))
                        if back is None or st2 is None:reason='reverse-flow-unavailable';break
                        q=q.reshape(-1,2);fb=np.linalg.norm(back.reshape(-1,2)-features,axis=1)
                        valid=status.ravel().astype(bool)&st2.ravel().astype(bool)&(fb<2.)
                        if valid.sum()<6:reason='insufficient-forward-backward-support';break
                        mat,inliers=cv2.estimateAffinePartial2D(features[valid],q[valid],method=cv2.RANSAC,ransacReprojThreshold=3.)
                        if mat is None:reason='incoherent-flow';break
                        support=inliers.ravel().astype(bool);before=features[valid][support];after=q[valid][support]
                        scale=np.linalg.norm(mat[0,:2]);turn=abs(np.arctan2(mat[1,0],mat[0,0]));motion=np.median(after-before,axis=0)
                        if len(before)<6 or support.mean()<.20 or not .65<scale<1.5 or turn>.55:reason='weak-affine-support';break
                        if np.linalg.norm(motion)<2.5:reason='stationary-background-risk';break
                        predicted=np.c_[points,np.ones(21)]@mat.T
                        inside=(predicted[:,0]>=0)&(predicted[:,0]<width)&(predicted[:,1]>=0)&(predicted[:,1]<height)
                        if inside.sum()<4:reason='hand-outside';break
                        points=predicted;features=after
                        tracked[str(i)]={'points':(points/height).tolist(),'in_frame':inside.tolist(),
                            'feature_support':len(before),'fb_error_px':float(np.median(fb[valid][support])),
                            'confidence':.45,'source':'verified-backward-video-flow'}
                edge,edge_source=incoming_edge(data,side,start,tracked)
                # A short blurred patch touching the SAME entry boundary can
                # establish partial visibility even when texture flow fails.
                # It supplies a boundary extent, never invented finger joints.
                regions={};image(start);reference=max(0,start-7);image(reference)
                seed=np.asarray(hand['points'])*height
                if edge==3:
                    xlo=max(0,int(seed[:,0].min()-40));xhi=min(width,int(seed[:,0].max()+40))
                    for i in range(max(reference+1,start-3),start):
                        image(i);yc=cv2.cvtColor(colors[i],cv2.COLOR_BGR2YCrCb)
                        changed=np.max(cv2.absdiff(colors[i],colors[reference]),axis=2)>22
                        skin=(yc[:,:,1]>141)&(yc[:,:,1]<180)&(yc[:,:,2]>80)&(yc[:,:,2]<125)
                        region=np.zeros((height,width),np.uint8);region[int(height*.72):,xlo:xhi]=(skin&changed)[int(height*.72):,xlo:xhi]*255
                        region=cv2.morphologyEx(region,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
                        _,_,stats,_=cv2.connectedComponentsWithStats(region)
                        candidates=[b for b in stats[1:] if b[4]>500 and b[2]>height*.04 and b[1]+b[3]>=height-2 and height*.85<b[1]<height*.98]
                        if candidates:
                            box=max(candidates,key=lambda b:b[4]);regions[str(i)]={'bounds':[float(box[0]/height),float(box[1]/height),float((box[0]+box[2])/height),1.],
                                'source':'moving-color-patch-at-entry-edge','confidence':.25}
                elif edge in (0,1):
                    # A blurred fist can precede a reliable landmark detector.
                    # Restrict motion evidence to the incoming border, away
                    # from the face; retain only a connected visible extent.
                    ylo=max(0,int(seed[:,1].min()-height*.15));yhi=min(height,int(seed[:,1].max()+height*.15))
                    xlo,xhi=(0,int(width*.20)) if edge==0 else (int(width*.80),width)
                    for i in range(max(reference+1,start-4),start):
                        image(i);yc=cv2.cvtColor(colors[i],cv2.COLOR_BGR2YCrCb)
                        changed=np.max(cv2.absdiff(colors[i],colors[reference]),axis=2)>28
                        skin=(yc[:,:,1]>141)&(yc[:,:,1]<180)&(yc[:,:,2]>80)&(yc[:,:,2]<125)
                        region=np.zeros((height,width),np.uint8);region[ylo:yhi,xlo:xhi]=(skin&changed)[ylo:yhi,xlo:xhi]*255
                        region=cv2.morphologyEx(region,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
                        _,_,stats,_=cv2.connectedComponentsWithStats(region)
                        candidates=[b for b in stats[1:] if b[4]>450 and b[3]>height*.04 and
                            (b[0]<=2 if edge==0 else b[0]+b[2]>=width-2) and b[2]<width*.20]
                        if candidates:
                            box=max(candidates,key=lambda b:b[4]);regions[str(i)]={'bounds':[float(box[0]/height),float(box[1]/height),float((box[0]+box[2])/height),float((box[1]+box[3])/height)],
                                'source':'moving-color-patch-at-entry-edge','confidence':.25}
                # Flow already establishes visibility; weak color alone may
                # not extend an otherwise verified track farther backwards.
                if tracked:regions={k:v for k,v in regions.items() if int(k)>=min(map(int,tracked))}
                first=min([start]+list(map(int,tracked))+list(map(int,regions)))
                episodes.append({'start':start,'first_visible':first,
                    'edge':edge,'edge_source':edge_source,'observations':tracked,'regions':regions,'stop_reason':reason})
                print('ENTRY_VIDEO',side,start+1,'first',episodes[-1]['first_visible']+1,'edge',edge,'tracked',len(tracked),'stop',reason,flush=True)
            record['sides'][side]=episodes
    finally:cap.release()
    return record

if __name__=='__main__':
    source=Path(sys.argv[1]);destination=Path(sys.argv[2]);data=json.loads(source.read_text(encoding='utf8'))
    if '--no-wilor' not in sys.argv:merge_wilor(data,source)
    result=recover(data);temporary=destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False),encoding='utf8');temporary.replace(destination)
    print('ENTRY_VIDEO_COMPLETE',str(destination),flush=True)

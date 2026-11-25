"""Conservative gesture context for ambiguous monocular hand depth.

This supplies a soft hypothesis preference, never a measured 3-D normal.
Clear palm backs, raised hands and body contacts do not use this preference.
"""
import numpy as np


def spread_palms_evidence(data):
    from sam_ellen_arm_visibility import screen_hand,hand_landmark_mask
    n=len(data['frames']);times=np.array([f['time'] for f in data['frames']])
    height=data.get('video_orientation',{}).get('decoded_height_width',[1280])[0]
    raw=np.zeros(n,bool);eligible=np.zeros(n,bool)
    for i,frame in enumerate(data['frames']):
        image=np.asarray(frame.get('image_points',[]),float)
        if image.ndim!=2 or len(image)<7:continue
        shoulders=image[[5,6],:2]/height;center=shoulders.mean(0)
        span=max(.08,float(np.linalg.norm(shoulders[1]-shoulders[0])))
        palms=[]
        for side in ('L','R'):
            h=screen_hand(data['screen'],i,side)
            if not h:break
            mask=hand_landmark_mask(h)
            if mask[[5,9,13,17]].sum()<3 or mask.sum()<12:break
            p=np.asarray(h['points']);along=p[[5,9,13,17]].mean(0)-p[0];across=p[5]-p[17]
            winding=float(across[0]*along[1]-across[1]*along[0])/max(1e-8,np.linalg.norm(across)*np.linalg.norm(along))
            facing=winding*(1 if side=='L' else -1)
            extension=np.median([np.linalg.norm(p[j+3]-p[0])/max(.005,np.linalg.norm(p[j]-p[0])) for j in (5,9,13,17)])
            palms.append((p[[0,5,9,13,17]].mean(0),facing,extension))
        if len(palms)!=2:continue
        left,right=palms
        low=all(p[0][1]>center[1]+span*.35 for p in palms)
        separate=left[0][0]>center[0]+span*.18 and right[0][0]<center[0]-span*.18
        level=abs(left[0][1]-right[0][1])<span*.6
        # One palm may be nearly edge-on; a clearly visible back is a veto.
        front=min(p[1] for p in palms)>-.35 and max(p[1] for p in palms)>.4
        contact=any(data.get('_sam_semantic',{}).get(side,[None]*n)[i] for side in ('L','R'))
        eligible[i]=low and separate and level and front and not contact
        raw[i]=eligible[i] and min(p[2] for p in palms)>1.3
    # Require a sustained open gesture; tolerate only brief ambiguous shape.
    result=np.zeros(n)
    for i in range(n):
        if not eligible[i]:continue
        neighbors=np.flatnonzero((abs(times-times[i])<=.16)&raw)
        if len(neighbors)>=3 and times[neighbors[-1]]-times[neighbors[0]]>=.065:
            result[i]=1.
    return result

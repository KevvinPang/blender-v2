"""Persistent hand observation provenance; never promote inferred joints to visible."""
import copy
import numpy as np


def prepare(screen):
    from sam_ellen_arm_visibility import hand_localization_usable
    frames=screen.get('frames',[]);fallback=screen.get('wilor_hands',[])
    result={s:[] for s in 'LR'}
    for side in 'LR':
        last=None;last_source=None;bias=np.zeros((21,2));last_time=-100.
        for i,row in enumerate(frames):
            primary=row.get('hands',{}).get(side)
            second=fallback[i].get(side) if len(fallback)==len(frames) else None
            primary=primary if hand_localization_usable(primary) else None
            second=second if hand_localization_usable(second) else None
            chosen=primary or second
            time=float(row.get('time',i/30))
            if not chosen:
                result[side].append(None);continue
            h=copy.deepcopy(chosen);source='primary' if primary else 'wilor'
            p=np.asarray(h['points'],float)
            confidence=float(h.get('confidence',.7))
            mask=np.asarray(h.get('in_frame',[True]*21),dtype=bool)
            visible=float(mask.mean())
            edge=float(np.min(np.r_[p[mask,0],screen.get('aspect',1.)-p[mask,0],
                                    p[mask,1],1.-p[mask,1]])) if mask.any() else 0.
            edge_weight=float(np.clip((edge+.025)/.10,.2,1.))
            reliability=float(np.clip((.18+.62*confidence)*np.sqrt(visible)*edge_weight,.08,.8))
            if not primary:reliability=min(reliability,float(h.get('tracking_reliability',.48)))
            if primary and second:
                disagreement=float(np.median(np.linalg.norm(p-np.asarray(second['points']),axis=1)))
                reliability*=float(np.clip(1-disagreement/.09,.35,1.))
            if last is not None and time-last_time<.10:
                if source!=last_source:
                    # Use simultaneous source discrepancy where available, not
                    # a frozen old wrist: true fast translation must survive.
                    if primary and second and last_source=='wilor':bias=np.asarray(second['points'])-p
                    else:
                        bias=last-p
                        magnitude=np.linalg.norm(bias,axis=1,keepdims=True)
                        bias*=np.minimum(1.,.025/np.maximum(1e-8,magnitude))
                    reliability*=.65
                bias*=np.exp(-(time-last_time)/.055)
            else:bias[:]=0
            raw=p.copy();p=p+bias
            h.update(points=p.tolist(),raw_points=raw.tolist(),tracking_source=source,
                     tracking_reliability=reliability,
                     inferred=bool(h.get('inferred',False)),
                     orientation_observed=bool(not h.get('inferred',False) and mask[0] and visible>=.7))
            h['wrist']=p[0].tolist();h['palm']=p[[0,5,9,13,17]].mean(0).tolist()
            result[side].append(h);last=p;last_time=time;last_source=source
        # A one-frame excursion with agreeing neighbours is weak evidence.
        raw=copy.deepcopy(result[side])
        for i in range(1,len(raw)-1):
            if any(raw[j] is None for j in (i-1,i,i+1)):continue
            a,b,c=[np.asarray(raw[j]['points']) for j in (i-1,i,i+1)]
            back=np.linalg.norm(a-c,axis=1);jump=np.minimum(np.linalg.norm(b-a,axis=1),np.linalg.norm(b-c,axis=1))
            bad=(jump>.045)&(back<.018)
            if bad.sum()>=3:
                p=b.copy();p[bad]=(a[bad]+c[bad])*.5
                result[side][i]['points']=p.tolist();result[side][i]['tracking_reliability']=.2
                result[side][i]['repaired_joints']=np.flatnonzero(bad).tolist()
    screen['_tracked_hands']=result
    return result

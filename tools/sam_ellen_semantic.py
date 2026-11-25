"""Character-relative near-body goals, with explicit observation limits.

Sustained image proximity establishes a soft body-region relationship, never
proof of physical contact. Goal points follow the posed character surface;
the common trajectory retains fixed bones, safe links and natural wrists.
"""
import numpy as np
from mathutils import Vector
from sam_ellen_contact_geometry import coverage_ratio


def activation(descriptor):
    """Attachment progress is distinct from confidence in the detected intent."""
    if not descriptor:return 0.
    return float(np.clip(descriptor.get('activation',
        descriptor.get('weight',0.)/max(1e-8,descriptor.get('confidence',.85))),0.,1.))


def hand_anchor(points,descriptor):
    """The same hand-surface attachment in source, solver and output space."""
    points=np.asarray(points,float)
    weights=descriptor.get('hand_weights') if descriptor else None
    if weights is not None:return np.einsum('...jc,j->...c',points,np.asarray(weights))
    joint=descriptor.get('joint',-1) if descriptor else -1
    return points[...,joint,:] if joint>=0 else points[...,[0,5,9,13,17],:].mean(axis=-2)


def surface_attachment(points,target):
    """Stable barycentric attachment from 2D landmarks, not physical depth.

    Use once on the segment's normalized median hand. A fixed combination of
    native hand landmarks avoids switching fingers when detections jitter.
    """
    from itertools import combinations
    p=np.asarray(points,float)-np.asarray(target,float)
    best=None
    for ids in combinations(range(len(p)),3):
        tri=p[list(ids)];matrix=np.vstack((tri.T,np.ones(3)))
        if abs(np.linalg.det(matrix))<1e-8:continue
        weights=np.linalg.solve(matrix,[0.,0.,1.])
        if weights.min() < -1e-7:continue
        score=float(weights@np.sum(tri*tri,axis=1))
        if best is None or score<best[0]:best=(score,ids,np.maximum(weights,0.))
    result=np.zeros(len(p))
    if best is None:result[int(np.argmin(np.linalg.norm(p,axis=1)))]=1.
    else:result[list(best[1])]=best[2]/best[2].sum()
    return result.tolist()


def contact_image_target(points,descriptor,anchor):
    """Translate the intact observed hand to its character-relative anchor."""
    result=np.asarray(points,float).copy()
    if not descriptor or anchor is None:return result
    center=hand_anchor(result,descriptor)
    return result+(np.asarray(anchor)-center)*activation(descriptor)


def prepare(data,options):
    from sam_ellen_arm_visibility import screen_hand,observed_frames
    n=len(data['frames']);result={s:[None]*n for s in 'LR'}
    data['_sam_semantic']=result;data['_sam_semantic_report']={}
    if not options.get('semantic_contacts'):return
    h=data.get('video_orientation',{}).get('decoded_height_width',[1280])[0]
    times=np.array([r['time'] for r in data['frames']])
    for side in 'LR':
        visible=observed_frames(data,side)
        for i,row in enumerate(data['screen']['frames']):
            face=row.get('face');hand=screen_hand(data['screen'],i,side)
            if not visible[i] or not face or not hand:continue
            p=np.asarray(hand['points']);center=np.asarray(face['center']);width=face['width']
            if width<.01:continue
            candidates=[]
            # A temple/hair-touch often uses the fingertips, not the palm.
            for sign,label in ((-1,'temple_right'),(1,'temple_left')):
                anchor=center+np.array([sign*.95,-.12])*width
                ids=[4,8,12];j=ids[int(np.argmin(np.linalg.norm(p[ids]-anchor,axis=1)))]
                dist=float(np.linalg.norm(p[j]-anchor)/width)
                if dist<.70:candidates.append((dist,label,j,(p[j]-anchor)/width))
            if face.get('mouth'):
                anchor=np.array(face['mouth']);mask=np.asarray(hand.get('in_frame',[True]*21),dtype=bool)
                confidence=float(hand.get('confidence',0.))
                fingers=[]
                for start in (5,9,13,17):
                    chain=p[start:start+4]
                    fingers.append(float(np.linalg.norm(chain[-1]-chain[0])
                        /max(1e-8,np.linalg.norm(np.diff(chain,axis=0),axis=1).sum())))
                open_hand=sum(v>.72 for v in fingers)>=3
                cover=coverage_ratio(p[mask],anchor,width)
                if confidence>=.65 and mask.sum()>=17 and open_hand and cover>=.78:
                    candidates.append((-cover,'mouth',-1,np.zeros(2)))
            image=data['frames'][i].get('image_points',[])
            if len(image)>6:
                shoulders=np.asarray(image)[[5,6],:2]/h
                span=float(np.linalg.norm(shoulders[0]-shoulders[1]))
                anchor=shoulders.mean(0)+np.array([0.,span*.28])
                palm=p[[0,5,9,13,17]].mean(0);delta=(palm-anchor)/max(span,1e-8)
                # Central chest band and a spread palm support this region;
                # lateral waves and palms held below the torso do not.
                # Enter with a tighter gate; once established, small detector
                # drift must not repeatedly detach/re-attach a held gesture.
                held=i>0 and result[side][i-1] and result[side][i-1]['region']=='chest'
                y_limit=.27 if held else .19
                # A single extended finger above a curled hand is pointing,
                # not evidence of chest contact merely because pixels overlap.
                straight=[]
                for start in (5,9,13,17):
                    chain=p[start:start+4];straight.append(float(np.linalg.norm(chain[-1]-chain[0])/max(1e-8,np.linalg.norm(np.diff(chain,axis=0),axis=1).sum())))
                pointing=straight[0]>.72 and sum(v<.68 for v in straight[1:])>=2
                if not pointing and .10<span<1.2 and abs(delta[0])<(.34 if held else .28) and abs(delta[1])<y_limit:
                    candidates.append((float(np.linalg.norm(delta))*2.5,'chest',-1,delta))
            if candidates:
                dist,label,j,delta=min(candidates,key=lambda x:x[0])
                if options.get('mirror'):
                    delta=delta.copy();delta[0]*=-1
                    if label.startswith('temple'):label='temple_right' if label=='temple_left' else 'temple_left'
                evidence=('sustained-hand-hull-mouth-coverage; depth contact unverified'
                          if label=='mouth' else 'sustained-2d-proximity; physical contact unverified')
                result[side][i]={'region':label,'joint':j,'relative':delta.tolist(),'weight':.85,'confidence':.85,
                    'coverage':float(cover) if label=='mouth' else None,'evidence':evidence}
        # Require sustained identity; ramp attachment/release to avoid snaps.
        # Bridge only short holes inside the same supported head region.
        raw=list(result[side]);valid=[k for k,r in enumerate(raw) if r]
        for a,b in zip(valid,valid[1:]):
            if b>a+1 and times[b]-times[a]<=.12 and raw[a]['region']==raw[b]['region'] and raw[a]['region'].startswith('temple'):
                for k in range(a+1,b):
                    t=(times[k]-times[a])/(times[b]-times[a])
                    raw[k]=dict(raw[a],relative=(np.array(raw[a]['relative'])*(1-t)+np.array(raw[b]['relative'])*t).tolist())
                    result[side][k]=raw[k]
        # Discard transient identities before building approach/release spans.
        # Otherwise a rejected 1-3 frame temple/chest guess still blocks the
        # envelope of a confirmed neighboring mouth contact.
        cursor=0
        while cursor<n:
            if raw[cursor] is None:cursor+=1;continue
            first=cursor;label=raw[cursor]['region']
            while cursor+1<n and raw[cursor+1] and raw[cursor+1]['region']==label:cursor+=1
            if times[cursor]-times[first]<.14:
                for k in range(first,cursor+1):raw[k]=None;result[side][k]=None
            cursor+=1
        i=0;segments=[]
        while i<n:
            if raw[i] is None:i+=1;continue
            start=i;region=raw[i]['region']
            while i+1<n and raw[i+1] and raw[i+1]['region']==region:i+=1
            end=i;i+=1
            if times[end]-times[start]<.14:
                for k in range(start,end+1):result[side][k]=None
                continue
            if region.startswith('temple'):
                from sam_ellen_temporal import trajectory
                ids=[raw[k]['joint'] for k in range(start,end+1)]
                joint=max(set(ids),key=ids.count);rel=[]
                sign=1 if region=='temple_left' else -1
                for k in range(start,end+1):
                    face=data['screen']['frames'][k].get('face');hand=screen_hand(data['screen'],k,side)
                    if face and hand:
                        # Descriptor labels are mirrored, source pixels are not.
                        source_sign=-sign if options.get('mirror') else sign
                        d=(np.array(hand['points'][joint])-np.array(face['center']))/face['width']-np.array([source_sign*.95,-.12])
                        if options.get('mirror'):d[0]*=-1
                    else:d=np.array(raw[k]['relative'])
                    rel.append(d)
                rel=trajectory(times[start:end+1],rel,.075,np.full(len(rel),8.))
                for offset,k in enumerate(range(start,end+1)):
                    result[side][k]=dict(result[side][k],joint=joint,relative=rel[offset].tolist())
            if region=='mouth':
                normalized=[]
                for k in range(start,end+1):
                    face=data['screen']['frames'][k].get('face');hand=screen_hand(data['screen'],k,side)
                    if face and face.get('mouth') and hand:
                        normalized.append((np.asarray(hand['points'])-np.asarray(face['mouth']))/face['width'])
                if normalized:
                    attachment=surface_attachment(np.median(normalized,axis=0),[0.,0.])
                    for k in range(start,end+1):result[side][k]['hand_weights']=attachment
            # Attachment starts while the visible hand approaches, not after
            # the source already covers the mouth. Keep recognition and the
            # transition envelope separate; never bridge a competing contact
            # or manufacture a hand observation during a long loss.
            enter,leave=start,end
            for k in range(start-1,-1,-1):
                if times[start]-times[k]>.20 or raw[k] is not None:break
                if screen_hand(data['screen'],k,side,include_inferred=True) is None:break
                enter=k
            for k in range(end+1,n):
                if times[k]-times[end]>.20 or raw[k] is not None:break
                if screen_hand(data['screen'],k,side,include_inferred=True) is None:break
                leave=k
            for k in range(enter,leave+1):
                if k<start:
                    result[side][k]=dict(result[side][start]);phase='approach'
                    fade=(times[k]-times[enter])/max(1e-8,times[start]-times[enter])
                elif k>end:
                    result[side][k]=dict(result[side][end]);phase='release'
                    fade=(times[leave]-times[k])/max(1e-8,times[leave]-times[end])
                else:
                    phase='hold';fade=1.
                    if enter==start:fade=min(fade,(times[k]-times[start]+1/data['fps'])/.12)
                    if leave==end:fade=min(fade,(times[end]-times[k]+1/data['fps'])/.12)
                fade=float(np.clip(fade,0.,1.))
                result[side][k]['phase']=phase
                result[side][k]['activation']=fade*fade*(3-2*fade)
                result[side][k]['weight']=result[side][k]['confidence']*result[side][k]['activation']
                if region=='mouth' and phase!='hold':
                    face=data['screen']['frames'][k].get('face')
                    hand=screen_hand(data['screen'],k,side,include_inferred=True)
                    result[side][k]['coverage']=coverage_ratio(hand['points'],face['mouth'],face['width']) if face and face.get('mouth') else 0.
            segment={'region':region,'frames':[start+1,end+1],'physical_contact_verified':False}
            segment['transition_frames']=[enter+1,leave+1]
            if region=='mouth':
                segment['source_coverage_median']=float(np.median([result[side][k]['coverage'] for k in range(start,end+1)]))
            segments.append(segment)
        data['_sam_semantic_report'][side]=segments


def goal(rig,side,descriptor,points,geometry):
    """Return a wrist goal and a surface-relative landmark target."""
    from sam_ellen_image_geometry import posed_geometry
    label=descriptor['region'];delta=np.asarray(descriptor['relative'])
    eye=(rig.pose.bones['Eye_L'].head+rig.pose.bones['Eye_R'].head)*.5
    if label=='chest':
        chest=rig.pose.bones.get('UpperBody2') or rig.pose.bones['UpperBody1']
        body=chest.matrix.to_quaternion()@chest.bone.matrix_local.to_quaternion().inverted()
        shoulder=(rig.pose.bones['Arm_L'].head+rig.pose.bones['Arm_R'].head)*.5
        span=(rig.pose.bones['Arm_L'].head-rig.pose.bones['Arm_R'].head).length
        # Local front depth is derived from torso-weighted mesh vertices.
        rest=[]
        from sam_ellen_body_clearance import torso_geometry
        blocks=geometry.get('torso',[])
        for vertices,_ in blocks:rest.extend(vertices[:,:3])
        # Human/anime torso proportions differ: retain the chest region,
        # rather than reproducing an absolute image offset down to the belly.
        center=shoulder+body@Vector((float(delta[0])*span,0.,-span*(.24+float(np.clip(delta[1],-.08,.08)))))
        local=(chest.matrix.inverted()@center)
        if rest:
            cloud=np.array(rest);z=chest.bone.matrix_local@local
            band=cloud[(abs(cloud[:,2]-z.z)<span*.12)&(abs(cloud[:,0]-z.x)<span*.18)]
            front=float(np.percentile(band[:,1],5)) if len(band) else float(np.percentile(cloud[:,1],5))
            center=chest.matrix@chest.bone.matrix_local.inverted()@Vector((z.x,front-.018,z.z))
        anchor=sum((points[j] for j in (0,5,9,13,17)),Vector())/5
    else:
        head=rig.pose.bones['Head'];basis=head.matrix.to_quaternion()@head.bone.matrix_local.to_quaternion().inverted()
        width=(rig.pose.bones['Eye_L'].head-rig.pose.bones['Eye_R'].head).length
        if label=='mouth':
            from sam_ellen_retarget import mouth_target
            # The solver and final-scene QA must refer to the same anatomical
            # mouth anchor. An eye/chin interpolation is not a lip surface.
            center=mouth_target(rig,{})
        elif geometry.get('head'):
            chin,left,right=posed_geometry(rig,geometry['head'])
            sign=1 if label=='temple_left' else -1
            center=eye+basis@Vector((sign*width*.98,-.005,width*.20))
        else:center=eye.copy()
        # Preserve the small observed offset rather than snapping every touch
        # to the exact center of the mouth/temple.
        center+=basis@Vector((float(delta[0])*width,0.,-float(delta[1])*width))
        anchor=(Vector(hand_anchor(np.asarray(points),descriptor))
                if label=='mouth' else points[descriptor['joint']])
    return points[0]+center-anchor,center

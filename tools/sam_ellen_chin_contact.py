"""Image-supported chin/hand relationship; no preset pose or finger reshaping."""
import math
import numpy as np
from mathutils import Vector,Quaternion


def chin_anchor(rig):
    """Find the lower central head skin, or use an explicit rig-local override."""
    import bpy
    head=next((b for b in rig.pose.bones if b.mmd_bone.name_j=='頭'),None)
    eyes=[rig.data.bones.get('Eye_'+s) for s in ('L','R')]
    if head is None or not all(eyes):return None
    override=rig.get('sam_chin_anchor')
    if override is not None and len(override)==3 and np.isfinite(override).all():
        return dict(bone=head.name,local=list(override),source='explicit-head-local-anchor')
    from sam_ellen_image_geometry import head_geometry
    geometry=head_geometry(rig)
    if geometry:
        return dict(bone=head.name,local=geometry['chin'],source=geometry['source'])
    eye=np.mean([e.head_local[:] for e in eyes],axis=0)
    width=float((eyes[0].head_local-eyes[1].head_local).length)
    candidates=[]
    if width<1e-4:return None
    for obj in bpy.data.objects:
        if obj.type!='MESH' or not any(m.type=='ARMATURE' and m.object==rig for m in obj.modifiers):continue
        group=obj.vertex_groups.get(head.name)
        if group is None:continue
        transform=rig.matrix_world.inverted()@obj.matrix_world
        for v in obj.data.vertices:
            if not any(g.group==group.index and g.weight>.8 for g in v.groups):continue
            p=np.array(transform@v.co)
            if (abs(p[0]-eye[0])<width*.22 and p[1]<eye[1]-width*.35
                    and eye[2]-width*2.5<p[2]<eye[2]-width*.7):
                candidates.append(p)
    if not candidates:return None
    points=np.array(candidates)
    lowest=points[:,2].min()
    point=np.median(points[points[:,2]<lowest+width*.025],axis=0)
    local=head.bone.matrix_local.inverted()@Vector(point)
    return dict(bone=head.name,local=list(local),source='central-lower-head-skin')


def segment_distance(point,a,b):
    v=b-a
    return float(np.linalg.norm(point-(a+np.clip(np.dot(point-a,v)/max(1e-10,np.dot(v,v)),0,1)*v)))


def observations(screen,side):
    """Require sustained chin proximity to thumb and a straight index finger."""
    from sam_ellen_temporal import trajectory
    rows=screen['frames'];times=np.array([r['time'] for r in rows])
    result=[None]*len(rows)
    for i,row in enumerate(rows):
        hand=row.get('hands',{}).get(side);face=row.get('face')
        if not hand or hand.get('confidence',0)<.65 or not face or 'chin' not in face:continue
        p=np.asarray(hand.get('points',[]),dtype=float);chin=np.asarray(face['chin'],dtype=float)
        mask=np.asarray(hand.get('in_frame',[True]*21),dtype=bool)
        if p.shape!=(21,2) or chin.shape!=(2,) or not np.isfinite(p).all() or not np.isfinite(chin).all():continue
        if mask.shape!=(21,) or not mask[[2,3,4,5,6,7,8,17]].all():continue
        span=float(np.linalg.norm(p[5]-p[17]))
        if span<.025:continue
        index=p[5:9]
        straight=np.linalg.norm(index[-1]-index[0])/max(1e-8,np.linalg.norm(np.diff(index,axis=0),axis=1).sum())
        thumb=min(segment_distance(chin,p[j],p[j+1]) for j in (1,2,3))
        finger=min(segment_distance(chin,p[j],p[j+1]) for j in (5,6,7))
        if (thumb>.85*span or finger>.65*span or p[0,1]-chin[1]<.4*span or straight<.88
                or np.linalg.norm(p[5]-chin)>.85*span):continue
        result[i]=dict(offset=(p[5]-chin)/span,span=span,weight=1.)
    valid=np.array([r is not None for r in result])
    # Bridge only brief detector flicker inside an otherwise supported gesture.
    indices=np.flatnonzero(valid)
    for a,b in zip(indices,indices[1:]):
        if b-a>1 and times[b]-times[a]<=.13:
            for i in range(a+1,b):
                w=(times[i]-times[a])/(times[b]-times[a])
                result[i]=dict(offset=result[a]['offset']*(1-w)+result[b]['offset']*w,
                    span=result[a]['span']*(1-w)+result[b]['span']*w,weight=1.)
                valid[i]=True
    edges=np.diff(np.r_[False,valid,False].astype(int))
    for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        if times[b-1]-times[a]<.18:
            result[a:b]=[None]*(b-a);continue
        offsets=trajectory(times[a:b],[r['offset'] for r in result[a:b]],.09,robust=.08)
        for j,i in enumerate(range(a,b)):
            w=np.clip(min(times[i]-times[a],times[b-1]-times[i])/.28,0,1)
            result[i].update(offset=offsets[j],weight=float(w*w*(3-2*w)))
    return result


def goal(position,orientation,captured,context,camera,observation,strength):
    """Keep the image-supported gap at the character's chin, including depth.

    The index MCP is a stable local anchor at the bottom of the thumb/index gap.
    It follows the solved wrist orientation, without changing finger articulation.
    """
    from sam_ellen_screen import project,unproject
    chin,offset,across=context
    relative=Quaternion(orientation)@captured[1].inverted()
    anchor=Vector(position)+relative@Vector(offset)
    # Hand skin is allowed a small clearance in front of the chin skin.
    depth=max(.05,-(camera[0]@chin).z-.008)
    plane=unproject(project(chin,camera),depth,camera)
    span=(project(plane+relative@Vector(across),camera)-project(plane,camera)).length
    target=Vector(project(chin,camera))+Vector(observation['offset'])*span
    desired=unproject(target,depth,camera)
    weight=observation['weight']*strength
    return Vector(position)+(desired-anchor)*weight,target,anchor,weight

"""Head surface anchors shared by image framing and hand clearance."""
import numpy as np
from mathutils import Vector


def head_geometry(rig):
    import bpy
    head=rig.pose.bones.get('Head')
    if not head:return None
    eye=np.mean([rig.data.bones['Eye_'+s].head_local[:] for s in ('L','R')],axis=0)
    width=float((rig.data.bones['Eye_L'].head_local-rig.data.bones['Eye_R'].head_local).length)
    skin=[];outline=[]
    for obj in bpy.data.objects:
        if obj.type!='MESH' or not any(m.type=='ARMATURE' and m.object and m.object.data==rig.data for m in obj.modifiers):continue
        group=obj.vertex_groups.get(head.name)
        if group is None:continue
        owner=next(m.object for m in obj.modifiers if m.type=='ARMATURE' and m.object and m.object.data==rig.data)
        transform=owner.matrix_world.inverted()@obj.matrix_world
        face_ids=set()
        for poly in obj.data.polygons:
            if poly.material_index>=len(obj.data.materials):continue
            name=obj.data.materials[poly.material_index].name.lower()
            if any(token in name for token in ('颜','顔','face')):face_ids.update(poly.vertices)
        for v in obj.data.vertices:
            if not any(g.group==group.index and g.weight>.8 for g in v.groups):continue
            p=np.array(transform@v.co)
            if abs(p[0]-eye[0])>width*2.5 or not eye[2]-width*1.7<p[2]<eye[2]+width*3.2:continue
            local=list(head.bone.matrix_local.inverted()@Vector(p))
            outline.append(local)
            if v.index in face_ids:skin.append((p,local))
    if len(skin)<20:return None
    points=np.array([p for p,_ in skin]);center=abs(points[:,0]-eye[0])<width*.2
    lowest=float(points[center,2].min())
    chin=np.mean([v for (p,v) in skin if abs(p[0]-eye[0])<width*.2 and p[2]<lowest+width*.035],axis=0)
    # Cheek anchors come from actual face skin, excluding rear skull and hair.
    cheek=[]
    for sign in (-1,1):
        ids=np.flatnonzero((points[:,2]<eye[2]+width*.05)&(points[:,2]>eye[2]-width*.7)&(points[:,1]<eye[1]+width*.2))
        edge=ids[np.argmax(sign*(points[ids,0]-eye[0]))]
        cheek.append(skin[edge][1])
    return {'bone':head.name,'chin':chin.tolist(),'cheeks':cheek,
            'outline':outline[::max(1,len(outline)//500)],'source':'weighted-face-skin-v2'}


def posed_geometry(rig,geometry):
    matrix=rig.pose.bones[geometry['bone']].matrix
    return [matrix@Vector(p) for p in [geometry['chin'],*geometry['cheeks']]]

"""Reduce upper-arm drag on Ellen's weighted shoulder ornaments."""


def stabilize(rig):
    import bpy
    count = 0
    for obj in rig.children_recursive:
        if obj.type != 'MESH' or obj.find_armature() != rig or obj.data.get('ellen_frill_weights_v1'):
            continue
        for side in ('L','R'):
            arm = obj.vertex_groups.get('Arm_'+side)
            ornaments = [obj.vertex_groups.get(f'Shoulder饰{i}_{side}') for i in (1,2,3)]
            if arm is None or not all(ornaments):
                continue
            indices = {g.index for g in ornaments}
            rows = []
            for v in obj.data.vertices:
                weights = {g.group:g.weight for g in v.groups}
                cloth = sum(weights.get(i,0.) for i in indices)
                old = weights.get(arm.index,0.)
                if cloth>.001 and old>.001:
                    rows.append((v.index, abs(v.co.x), old, cloth, weights))
            if not rows:
                continue
            if obj.data.users>1:
                obj.data = obj.data.copy()
            inner = min(row[1] for row in rows)
            span = max(1e-6,max(row[1] for row in rows)-inner)
            for i,x,old,cloth,weights in rows:
                cap = .035+.105*((x-inner)/span)
                transfer = max(0.,old-cap)*min(1.,cloth/.12)
                if transfer<1e-6:
                    continue
                arm.add([i],old-transfer,'REPLACE')
                for g in ornaments:
                    w = weights.get(g.index,0.)
                    if w:
                        g.add([i],w+transfer*w/cloth,'REPLACE')
                count += 1
            obj.data['ellen_frill_weights_v1']=True
    if count:
        bpy.context.view_layer.update()
    return count

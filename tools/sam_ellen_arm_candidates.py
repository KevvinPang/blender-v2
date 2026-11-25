"""Carry feasible neighboring poses into the trajectory's candidate graph."""
import math
import numpy as np
from mathutils import Quaternion


def transport_neighbors(sets,rows,upper,lower):
    from sam_ellen_arm_trajectory import _rotate
    from sam_ellen_body_clearance import chains_safe
    added=0
    for direction in (1,-1):
        order=range(1,len(rows)) if direction==1 else range(len(rows)-2,-1,-1)
        for i in order:
            if rows[i].get('locked'):continue
            j=i-direction;source=sets[j];target=sets[i]
            # Include good neighbors and the last transported corridor nodes.
            ids=np.unique(np.r_[np.argsort(source['cost'])[:8],
                                np.arange(max(0,len(source['q'])-8),len(source['q']))])
            transform=rows[i]['body']@rows[j]['body'].inverted()
            q=np.array([tuple(transform@Quaternion(x)) for x in source['q'][ids]])
            f=source['flex'][ids]
            u=_rotate(q,[0,1,0]);e=rows[i]['a']+u*upper
            w=e+(u*np.cos(f)[:,None]+_rotate(q,[0,0,1])*np.sin(f)[:,None])*lower
            safe=chains_safe(rows[i]['a'],e,w,rows[i]['env'],upper,lower,margin=.003)
            safe&=(f>=math.radians(15)-1e-5)&(f<=math.radians(180-rows[i]['minimum_elbow'])+1e-5)
            duplicate=(abs(q@target['q'].T)>.99995)&(abs(f[:,None]-target['flex'][None,:])<.005)
            safe&=~duplicate.any(1)
            if not safe.any():continue
            q,f,e,w=q[safe],f[safe],e[safe],w[safe]
            cost=rows[i]['pose_objective'](e,w)
            for key,value in dict(q=q,flex=f,e=e,w=w,cost=cost).items():
                target[key]=np.concatenate((target[key],value))
            added+=len(q)
    return added

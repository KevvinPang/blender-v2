"""Fuse detector samples and local image motion before anatomical fitting."""
import numpy as np

PALM = [0, 5, 9, 13, 17]


def inside(points):
    return np.all((points >= 0) & (points <= 1), axis=-1)


def sample(field, points):
    h,w = field.shape[:2]
    xy = np.clip(points*[w,h]-.5, [0,0], [w-1,h-1])
    x,y = xy.astype(int).T
    u,v = (xy-xy.astype(int)).T
    right, bottom = np.minimum(x+1,w-1), np.minimum(y+1,h-1)
    return ((1-u)[:,None]*((1-v)[:,None]*field[y,x]+v[:,None]*field[bottom,x])
            + u[:,None]*((1-v)[:,None]*field[y,right]+v[:,None]*field[bottom,right]))


def transport(points, field, reverse, prior):
    visible = inside(points)
    flow = sample(field,points)
    check = np.linalg.norm(flow+sample(reverse,points+flow),axis=1)
    good = visible & (check < .018)
    if good.sum() < 3:
        return points+prior, False
    motion = np.median(flow[good],axis=0)
    # A small exposed fingertip can disappear into stationary background.
    # Continue its previous outward motion when there is too little texture.
    if visible.sum()<9 and np.linalg.norm(motion)<.35*np.linalg.norm(prior):
        return points+prior,False
    local = flow-motion
    local[~good]=0
    return points+motion+local*.65,True


def stabilize_cropped_observations(world,uv,observed,forward,backward):
    """Use nearby views of a cropped wrist in the current frame's coordinates."""
    if forward is None or backward is None:return
    source=uv.copy();poses=world.copy()
    for t in np.flatnonzero(observed):
        if np.min(np.minimum(source[t,0],1-source[t,0]))>.10:continue
        candidates=[source[t]];shapes=[poses[t]];weights=[1.]
        for step in (-1,1):
            for distance in range(1,4):
                j=t+step*distance
                if j<0 or j>=len(uv) or not observed[j]:break
                points=source[j].copy()
                fields,reverse=(forward,backward) if step<0 else (backward,forward)
                for frame in range(j,t,-step):
                    k=min(frame,frame-step)
                    flow=sample(fields[k],points)
                    supported=inside(points)&(np.linalg.norm(flow+sample(reverse[k],points+flow),axis=1)<.015)
                    if supported.sum()<12:break
                    if np.percentile(np.linalg.norm(flow[supported],axis=1),85)>.035:break
                    points+=np.where(supported[:,None],flow,np.median(flow[supported],axis=0))
                else:
                    if np.median(np.linalg.norm(points-source[t],axis=1))<.12:
                        candidates.append(points);shapes.append(poses[j]);weights.append(1.-.12*distance)
        if len(candidates)<3:continue
        uv[t]=.75*np.median(candidates,axis=0)+.25*np.average(candidates,axis=0,weights=weights)
        world[t]=np.average(shapes,axis=0,weights=weights)


def exit_edge(points,velocity):
    center=np.median(points[PALM],axis=0)
    distances=np.array([center[0],1-center[0],center[1],1-center[1]])
    speeds=np.array([-velocity[0],velocity[0],-velocity[1],velocity[1]])
    eligible=(distances<.24)&(speeds>.006)
    if not eligible.any():return -1
    return int(np.argmax(np.where(eligible,speeds/np.maximum(.025,distances),-1)))


def fuse(track,uv,valid,forward=None,backward=None):
    """Use clear samples on both sides of central gaps, and one side on exits."""
    observed=valid.copy()
    confidence=valid.astype(float)
    edge=np.full(len(valid),-1,int)
    held=np.zeros(len(valid),bool)
    corrected=np.zeros(len(valid),bool)
    speeds=np.zeros((len(valid),2))
    flows=forward is not None and backward is not None
    for t in range(1,len(valid)):
        if not observed[t] or not observed[t-1]:continue
        prev=uv[t-1]
        prior=np.median(speeds[max(1,t-3):t],axis=0) if t>1 else np.zeros(2)
        pred,reliable=(transport(prev,forward[t-1],backward[t-1],prior)
                       if flows else (prev+prior,False))
        motion=np.median(pred-prev,axis=0)
        # Only substitute a detector jump near an edge when it disagrees
        # with previous motion AND independent image evidence.
        side=exit_edge(prev,motion)
        if side<0:side=exit_edge(prev,prior)
        error=np.median(np.linalg.norm(uv[t]-pred,axis=1))
        if side>=0 and reliable and error>max(.055,np.linalg.norm(motion)*1.5):
            uv[t]=pred;track[t]=track[t-1];held[t]=corrected[t]=True
        speeds[t]=np.median(uv[t]-prev,axis=0)
    ids=np.flatnonzero(observed)
    for i,a in enumerate(ids):
        b=ids[i+1] if i+1<len(ids) else len(valid)
        if b<=a+1:continue
        start=a
        while start>a-4 and start>0 and observed[start-1]:start-=1
        recent=speeds[start+1:a+1]
        vel=np.median(recent,axis=0) if len(recent) else np.zeros(2)
        if flows and a<len(forward):
            p,ok=transport(uv[a],forward[a],backward[a],vel)
            if ok:vel=.4*vel+.6*np.median(p-uv[a],axis=0)
        side=exit_edge(uv[a],vel)
        if side>=0:
            axis=side//2;sign=(-1,1)[side%2]
            last=uv[a].copy()
            for t in range(a+1,min(b,a+10)):
                p,ok=(transport(last,forward[t-1],backward[t-1],vel) if flows else (last+vel,False))
                movement=np.median(p-last,axis=0)
                movement[axis]=sign*max(sign*movement[axis],sign*vel[axis])
                # A common palm displacement avoids shearing the extrapolated hand.
                uv[t]=last+movement;last=uv[t];track[t]=track[a]
                edge[t]=side;valid[t]=held[t]=True;confidence[t]=1.
            continue
        if b<len(valid) and b-a<=9:
            left,right=uv[a].copy(),uv[b].copy()
            ahead={};behind={}
            for t in range(a+1,b):
                if flows:left,_=transport(left,forward[t-1],backward[t-1],vel)
                ahead[t]=left
            for t in range(b-1,a,-1):
                if flows:right,_=transport(right,backward[t],forward[t],-vel)
                behind[t]=right
            for t in range(a+1,b):
                u=(t-a)/(b-a);uv[t]=(1-u)*ahead[t]+u*behind[t]
                track[t]=(1-u)*track[a]+u*track[b];valid[t]=True;confidence[t]=1.
        else:
            for t in range(a+1,min(b,a+9)):
                dt=t-a
                uv[t]=uv[a]+vel*sum(.7**j for j in range(dt))
                track[t]=track[a];valid[t]=held[t]=True;confidence[t]=1-dt/9
    return observed,confidence,edge,held,corrected


def apply_contour(world,uv,valid,observed,confidence,edge,held,witness,boundary,raw_uv=None):
    target=np.full(len(valid),np.nan)
    starts=np.flatnonzero((witness>=0)&np.r_[True,witness[1:]!=witness[:-1]])
    for start in starts:
        if not observed[start]:continue
        side=int(witness[start]);axis=side//2;sign=(-1,1)[side%2]
        # The last clear pose supplies articulation. Contour position remains
        # independently observable even after a fingertip is all that is left.
        pose=world[start].copy();shape=uv[start].copy()
        inner=np.min if sign>0 else np.max
        end=start
        while end<len(valid) and witness[end]==side:end+=1
        for t in range(start,end):
            goal=boundary[t]
            if raw_uv is not None and observed[t] and np.isfinite(raw_uv[t]).all():
                points=raw_uv[t]
                if inside(points).sum()>=8:
                    tip_extent=inner(points[:,axis])
                    # Skin elsewhere in the image can hold the segmented extent
                    # stationary as the detected hand leaves. Limit its vote.
                    goal=tip_extent+np.clip(goal-tip_extent,-.025,.025)
            if abs(goal-side%2)<1e-6:goal+=sign*.025
            if observed[t] and not corrected_jump(uv[t],shape,axis) and inside(uv[t]).mean()>.55:
                # Preserve still-visible finger motion. Once a digit leaves,
                # retain its last complete segment directions at its new base.
                old_uv,old_pose=shape,pose
                shape=uv[t].copy();pose=world[t].copy()
                for j in (1,5,9,13,17):
                    if not inside(shape[j:j+4]).all():
                        pose[j:j+4]=pose[j]+old_pose[j:j+4]-old_pose[j]
                        shape[j:j+4]=shape[j]+old_uv[j:j+4]-old_uv[j]
            # Lock only the retreating axis. Keep the last reliable transverse
            # coordinate rather than introducing a sideways detector flip.
            uv[t]=shape.copy();uv[t,:,axis]+=goal-inner(shape[:,axis])
            world[t]=pose;held[t]=True;valid[t]=True;edge[t]=side
            confidence[t]=1.;target[t]=goal
        for t in range(end,min(len(valid),end+6)):
            if observed[t]:break
            target[t]=side%2+sign*.04;uv[t]=uv[end-1].copy()
            world[t]=pose;held[t]=True;valid[t]=True;edge[t]=side;confidence[t]=1.
    return target


def corrected_jump(current,previous,axis):
    return abs(np.median(current[:,1-axis]-previous[:,1-axis]))>.065


def edge_transitions(track, fps):
    """Carry a partially visible hand across a border in either time direction."""
    uv, world = track['uv'], track['world']
    observed, valid = track['observed'], track['valid']
    contour = np.isfinite(track.get('boundary', np.full(len(valid), np.nan)))
    starts = np.flatnonzero(observed & ~np.r_[False, observed[:-1]])
    ends = np.flatnonzero(observed & ~np.r_[observed[1:], False])
    for first, last in zip(starts, ends):
        if last-first < 3:
            continue
        for anchor, step in ((first, -1), (last, 1)):
            after = anchor+step
            if not 0 <= after < len(valid) or contour[anchor] or contour[after]:
                continue
            nearby = anchor+step*np.arange(1,10)
            if observed[nearby[(nearby>=0) & (nearby<len(valid))]].any():
                continue
            window = anchor-step*np.arange(min(5, last-first+1))
            center = np.median(uv[window][:,PALM], axis=1)
            motion = np.median(-np.diff(center, axis=0), axis=0)
            extreme = np.array([uv[anchor,:,0].min(), 1-uv[anchor,:,0].max(),
                                uv[anchor,:,1].min(), 1-uv[anchor,:,1].max()])
            speeds = np.array([-motion[0], motion[0], -motion[1], motion[1]])
            score = np.where((extreme < .14) & (speeds > .007), speeds / np.maximum(.04, extreme), -1)
            if np.max(score) <= 0:
                continue
            side = int(np.argmax(score))
            axis, sign = side//2, (-1,1)[side%2]
            # Estimate the retreat of the last visible landmark. Palm centres
            # alone stop descending when the detector invents an offscreen wrist.
            inner = np.min(sign*uv[window,:,axis], axis=1)
            speed = np.clip(max(speeds[side], np.median(-np.diff(inner))), .012, .16)
            template = uv[anchor].copy()
            pose = world[anchor].copy()
            if step > 0:
                # At the border trust the last complete shape. Preserve the
                # current visible extent even when the detected wrist folds back.
                stable = next((i for i in window if inside(uv[i]).all()), anchor)
                if abs(stable-anchor) > 0:
                    template = uv[stable].copy()
                    template += np.median(uv[anchor]-template, axis=0)
                    template[:,axis] += sign*(inner[0]-np.min(sign*template[:,axis]))
                    pose = world[stable].copy()
            for dt in range(1, max(3, round(fps*.45))+1):
                i = anchor+step*dt
                if not 0 <= i < len(valid) or observed[i] or contour[i]:
                    break
                # Stop once the innermost digit clears the edge; never
                # extrapolate an accelerating arm arbitrarily far off camera.
                travel = min(speed*dt, max(0., sign*(side%2)-inner[0])+.045)
                uv[i] = template + motion*min(dt,3)
                uv[i,:,axis] = template[:,axis]+sign*travel
                world[i] = pose
                valid[i] = track['held'][i] = True
                track['confidence'][i] = 1.
                track['edge'][i] = side
                if inner[0]+travel >= sign*(side%2)+.04:
                    # Old forward predictions may continue beyond this point.
                    j = i+step
                    while 0 <= j < len(valid) and valid[j] and not observed[j] and not contour[j]:
                        valid[j] = False
                        j += step
                    break


def approach_depth(uv,world,flex,observed):
    """Short perspective-scale cues, gated by stable palm angle and arm extension."""
    result=np.ones(len(uv))
    for t in range(3,len(uv)):
        a=t-3
        if not observed[a:t+1].all() or not np.isfinite(flex[[a,t]]).all():continue
        if not inside(uv[[a,t]][:,PALM]).all():continue
        def normal(p):
            n=np.cross(p[5]-p[0],p[17]-p[0]);return n/max(np.linalg.norm(n),1e-9)
        if normal(world[a])@normal(world[t])<.96:continue
        lengths=[np.linalg.norm(uv[[a,t],j]-uv[[a,t],k],axis=1) for j,k in ((5,17),(0,9),(0,13))]
        ratio=np.array([v[1]/max(v[0],1e-7) for v in lengths])
        gain=np.median(ratio)
        if np.ptp(ratio)>.16 or abs(np.log(max(gain,.01)))<.06:continue
        if (gain-1)*(flex[a]-flex[t])<=0 or abs(flex[a]-flex[t])<.035:continue
        result[t]=np.clip(1/gain,.92,1.08)
    return result

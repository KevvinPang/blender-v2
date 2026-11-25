"""Coupled camera-ray depth and elbow-circle fit to independent image direction."""
import math
import numpy as np
from mathutils import Vector


def fit_direction(a,e,w,upper,lower,camera,evidence,plane,minimum=35.,previous=None,lock_depth=False,depth_preference=.35):
    """Keep wrist pixels and link lengths; select a front-safe image elbow.

    The image direction owns the elbow plane. Depth remains as close as possible
    to the hand-size fit. All choices are on complete two-link chains.
    """
    from sam_ellen_screen import project,unproject
    from sam_ellen_screen_fit import projected
    a,e,w=map(lambda p:np.asarray(p,dtype=float),(a,e,w))
    target=np.asarray(evidence[0],float);target/=max(1e-8,np.linalg.norm(target))
    xy=project(Vector(w),camera);depth=-(camera[0]@Vector(w)).z;reach=upper+lower
    shoulder=-(camera[0]@Vector(a)).z
    depths=np.unique(np.r_[depth,np.linspace(max(.04,shoulder-reach),shoulder+reach,49)])
    if lock_depth:depths=np.array([depth])
    origin=np.asarray(unproject(xy,0.,camera));ray=np.asarray(unproject(xy,1.,camera))-origin
    lo=math.sqrt(upper**2+lower**2-2*upper*lower*math.cos(math.radians(minimum)))+1e-6
    hi=math.sqrt(upper**2+lower**2-2*upper*lower*math.cos(math.radians(165.)))-1e-6
    forward=np.asarray(plane[1],float);forward/=np.linalg.norm(forward)
    transform=np.asarray(camera[0])

    def evaluate(z,theta):
        goals=origin+z[:,None]*ray
        delta=goals-a;distance=np.linalg.norm(delta,axis=1);axis=delta/np.maximum(1e-8,distance[:,None])
        seed=np.tile([0.,1.,0.],(len(z),1));seed[abs(axis[:,1])>.9]=[1.,0.,0.]
        x=np.cross(axis,seed);x/=np.maximum(1e-8,np.linalg.norm(x,axis=1))[:,None];y=np.cross(axis,x)
        along=(distance**2+upper**2-lower**2)/(2*np.maximum(1e-8,distance))
        radius=np.sqrt(np.maximum(0,upper**2-along**2))
        bend=x*np.cos(theta[:,None])+y*np.sin(theta[:,None])
        elbows=a+axis*along[:,None]+bend*radius[:,None]
        cp=np.c_[elbows,np.ones(len(elbows))]@transform.T
        image=projected(cp[:,:3],camera)
        v=image-np.asarray(xy);v/=np.maximum(1e-8,np.linalg.norm(v,axis=1))[:,None]
        dot=np.clip(v@target,-1,1)
        angle=np.arccos(dot)
        valid=(z>=.04)&(distance>=lo)&(distance<=hi)&((goals-a)@forward>=-1e-6)&((elbows-a)@forward>=-1e-6)&(cp[:,2]<-.025)
        # Smooth terms choose among image-compatible branches. The source
        # elbow cannot outweigh a measured 20-40 degree direction error.
        cost=40*angle**2+depth_preference*((z-depth)/reach)**2+.025*np.sum(((elbows-e)/reach)**2,axis=1)
        if previous is not None:
            olde,oldw=map(np.asarray,previous)
            cost+=.5*np.sum(((elbows-olde)/reach)**2,axis=1)
        cost[~valid]=np.inf
        return cost,elbows,goals,bend,angle

    zz,tt=np.meshgrid(depths,np.arange(180)*math.tau/180,indexing='ij')
    z,t=zz.ravel(),tt.ravel();values=evaluate(z,t);k=int(np.argmin(values[0]))
    if not np.isfinite(values[0][k]):return Vector(e),Vector(w),None,{'feasible':False}
    bestz,bestt=z[k],t[k]
    # Local multiscale refinement removes the visible grid stepping.
    for step in range(9):
        dz=0. if lock_depth else reach/48/(2**step);dt=math.tau/180/(2**step)
        z,t=np.meshgrid(bestz+np.arange(-2,3)*dz,bestt+np.arange(-2,3)*dt,indexing='ij')
        z,t=z.ravel(),t.ravel();values=evaluate(z,t);k=int(np.argmin(values[0]));bestz,bestt=z[k],t[k]
    _,es,ws,bs,angles=values
    return Vector(es[k]),Vector(ws[k]),Vector(bs[k]),{'feasible':True,
        'direction_error_degrees':math.degrees(float(angles[k])),
        'depth_change':float(bestz-depth)}

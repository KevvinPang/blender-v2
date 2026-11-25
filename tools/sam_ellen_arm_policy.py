"""Shared evidence, interval and hinge rules for automatic arm processing."""
import math
import numpy as np
from mathutils import Quaternion, Vector

VERSION = 2
PREFIXES = ('Arm_', 'Elbow_', 'Wrist_', 'ArmTwist_', 'HandTwist_')


def supported_elbow_minimum(data,side,index):
    """One evidence floor for graph candidates, smoothing and final validation."""
    policy=data.get('_sam_arm_policy',[])
    row=policy[index].get(side,{}) if len(policy)==len(data['frames']) else {}
    return float(np.clip(row.get('active_minimum_elbow',row.get('minimum_elbow',35.)),35.,150.))


def wrist_limit(elbow_degrees):
    """Avoid stacking a deep elbow fold with a second extreme wrist bend."""
    return float(np.clip(45.+.75*(elbow_degrees-35.),45.,60.))


def excluded(options, side):
    return side == options.get('exclude_side') or side in options.get('exclude_sides', ())


def activation(times, options):
    t = np.asarray(times, float)-times[0]
    start, end = options.get('start', 0.), options.get('end', 0.)
    w = np.ones(len(t))
    if start > 0: w *= np.clip((t-start)/.15, 0., 1.)
    if end > 0: w *= np.clip((end-t)/.15, 0., 1.)
    return w*w*(3-2*w)*options.get('strength', 1.)


def elbow_floor(a, e, w, evidence=None):
    """A hand-only fit cannot invent much more flexion than the capture prior.

    This is relative to the moving source pose, not a fixed idle or a ban on
    real tight raises. Forearm color axes are still not observed 3-D elbows.
    """
    angle = math.degrees((Vector(a)-Vector(e)).angle(Vector(w)-Vector(e)))
    return float(np.clip(angle-12., 35., 150.))


def transport_bend(bend, old_axis, axis):
    axis = Vector(axis).normalized()
    b = Vector(bend)
    if old_axis is not None:
        b = Vector(old_axis).normalized().rotation_difference(axis) @ b
    b -= axis*b.dot(axis)
    return b.normalized() if b.length > 1e-7 else axis.orthogonal().normalized()


def camera_reachable_goal(a, goal, upper, lower, minimum, camera, plane=None):
    """Intersect the observed camera ray with the arm's anatomical shell.

    Extending a too-short arm radially from its shoulder changes its image
    position. Moving along the camera ray preserves that observed position.
    """
    a,goal=Vector(a),Vector(goal);cp=camera[0]@goal;inverse=camera[0].inverted()
    if camera[3]=='ORTHO':
        origin=inverse@Vector((cp.x,cp.y,0));direction=inverse.to_3x3()@Vector((0,0,-1))
    else:
        origin=inverse@Vector();direction=goal-origin
    direction.normalize();t0=(goal-origin).dot(direction)
    center=(a-origin).dot(direction);perp=(a-origin-direction*center).length_squared
    low=math.sqrt(upper**2+lower**2-2*upper*lower*math.cos(math.radians(minimum)))+1e-6
    high=math.sqrt(upper**2+lower**2-2*upper*lower*math.cos(math.radians(165)))-1e-6
    if perp>high**2:return goal
    root=math.sqrt(max(0.,high**2-perp));lo=max(.025,center-root);hi=center+root
    if plane is not None:
        forward=Vector(plane[1]).normalized();base=(origin-a).dot(forward);slope=direction.dot(forward)
        if abs(slope)>1e-8:
            boundary=-base/slope
            if slope>0:lo=max(lo,boundary)
            else:hi=min(hi,boundary)
        elif base<0:return goal
    if hi<lo:return goal
    intervals=[(lo,hi)]
    if perp<low**2:
        gap=math.sqrt(low**2-perp);left,right=center-gap,center+gap
        intervals=[(lo,min(hi,left)),(max(lo,right),hi)]
    candidates=[float(np.clip(t0,x,y)) for x,y in intervals if y>=x]
    if not candidates:return goal
    return origin+direction*min(candidates,key=lambda t:abs(t-t0))


def front_circle(a, goal, upper, lower, bend, plane, minimum=35., previous=None):
    """Choose a front-safe elbow on ONE wrist sphere, never project links apart.

    Preserve the wrist goal whenever its elbow circle has a feasible front
    branch. Camera-ray reach is selected upstream; this solve keeps fixed
    link lengths and never deepens the elbow to satisfy a palm orientation.
    """
    a, goal, bend = map(Vector, (a, goal, bend))
    lo = math.sqrt(upper**2+lower**2-2*upper*lower*math.cos(math.radians(minimum)))+1e-6
    hi = math.sqrt(upper**2+lower**2-2*upper*lower*math.cos(math.radians(165.)))-1e-6
    delta = goal-a
    axis = delta.normalized() if delta.length > 1e-7 else Vector((0., -.2, -1.)).normalized()
    distance = float(np.clip(delta.length, lo, hi))
    w = a+axis*distance
    origin, forward = map(Vector, plane)
    forward.normalize()
    # The wrist may not move behind its shoulder; keep its lateral ray as far
    # as possible, then rebuild the same elbow circle at the reachable depth.
    depth = (w-a).dot(forward)
    if depth < 0:
        lateral = w-a-forward*depth
        axis = (lateral+forward*1e-6).normalized()
        w = a+axis*distance
    along = (distance**2+upper**2-lower**2)/(2*distance)
    radius = math.sqrt(max(0., upper**2-along**2))
    b = transport_bend(bend, None, axis)
    # Front condition is cos(theta) >= c on the elbow circle. Project onto
    # its nearest boundary analytically, with a stable previous-side tie.
    across = forward-axis*forward.dot(axis)
    if radius > 1e-7 and across.length > 1e-7:
        center = a+axis*along
        required = -(center-a).dot(forward)/(radius*across.length)
        if required > -1:
            target = across.normalized()
            limit = math.acos(float(np.clip(required, -1., 1.)))
            angle = math.atan2(axis.dot(target.cross(b)), target.dot(b))
            if abs(angle) > limit:
                if abs(abs(angle)-math.pi) < 1e-4 and previous is not None:
                    old = transport_bend(previous, None, axis)
                    sign = 1. if axis.dot(target.cross(old)) >= 0 else -1.
                else: sign = 1. if angle >= 0 else -1.
                b = Quaternion(axis, sign*limit) @ target
    e = a+axis*along+b*radius
    return e, w, b

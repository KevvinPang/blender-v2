"""Deterministic additive idle motion over an immutable captured action."""
import math
import numpy as np
from mathutils import Quaternion,Vector


def drift(time,seed):
    # Same continuous, scrub-safe drift as the old body/gaze assistance.
    from sam_ellen_retarget import gaze_drift
    return gaze_drift(time,.12,seed,.8)


def intermittent_sway(time,seed):
    """One gentle excursion with flat pauses; C2 at start and end."""
    def random(cell,salt):
        n=(int(seed)*2654435761+cell*2246822519+salt)&0xffffffff
        n=((n^(n>>16))*3266489917)&0xffffffff
        return (n^(n>>15))/0xffffffff
    cycle=math.floor(time/7.5);within=time-cycle*7.5
    start=1.2+random(cycle,101)*1.0;duration=3.8+random(cycle,503)*.8
    u=(within-start)/duration
    if not 0.<u<1.:return 0.
    direction=1. if random(cycle,907)>.5 else -1.
    return direction*(.65+.35*random(cycle,131))*math.sin(math.pi*u)**4


def curve_map(action,slot):
    return {(c.data_path,c.array_index):c for layer in action.layers for strip in layer.strips
            for bag in strip.channelbags if slot is None or bag.slot_handle==slot.handle for c in bag.fcurves}


def write(action,owner,path,index,values):
    identifier=owner.animation_data.action_slot.identifier if owner.animation_data.action_slot else None
    slot=next((s for s in action.slots if s.identifier==identifier),None)
    if slot is None:slot=action.slots.new(owner.id_type,owner.name)
    layer=action.layers[0] if action.layers else action.layers.new('V2')
    strip=layer.strips[0] if layer.strips else layer.strips.new(type='KEYFRAME')
    bag=strip.channelbag(slot,ensure=True)
    curve=bag.fcurves.find(path,index=index) or bag.fcurves.new(path,index=index)
    curve.keyframe_points.clear();curve.keyframe_points.add(len(values))
    curve.keyframe_points.foreach_set('co',np.column_stack((np.arange(1,len(values)+1),values)).reshape(-1).tolist())
    for p in curve.keyframe_points:p.interpolation='LINEAR'
    curve.update()


def apply(scene,rig,settings,visibility=None):
    from sam_ellen_gaze import contact_weight,merged_ranges
    intervals=[(r.start_frame,r.end_frame) for r in settings.away_ranges if r.enabled]
    if any(b<a for a,b in intervals):raise ValueError('不看镜头区间的结束帧早于开始帧')
    intervals=merged_ranges(intervals)
    if not rig.animation_data or not rig.animation_data.action:raise ValueError('请先生成或载入 V2 动作')
    previous=rig.animation_data.action
    previous_slot=rig.animation_data.action_slot
    base=rig.get('mocap_v2_base_action')
    if base is None or previous!=rig.get('mocap_v2_assisted_action'):
        base=previous.copy();base.name='V2 原始动作 · 辅助前';base.use_fake_user=True
        rig['mocap_v2_base_action']=base
    base_slot=next((s for s in base.slots if previous_slot and s.identifier==previous_slot.identifier),None)
    curves=curve_map(base,base_slot)
    T=int(rig.get('mocap_v2_frame_count',scene.frame_end));fps=scene.render.fps/scene.render.fps_base
    # Unknown visibility is conservative: do not perturb the arms.
    if visibility is None:visibility=np.ones((T,2),bool)
    visibility=np.asarray(visibility)
    if visibility.shape!=(T,2):raise ValueError('手部可见性数据长度与动作不一致')
    action=base.copy();action.name='V2 动捕 · 自然辅助'
    names=['UpperBody','UpperBody1','UpperBody2','Arm_L','Arm_R','Elbow_L','Elbow_R','Eye_L','Eye_R']
    for name in names:
        bone=rig.pose.bones.get(name)
        if bone is None:continue
        path=bone.path_from_id('rotation_quaternion')
        raw=np.array([[curves[path,k].evaluate(t+1) if (path,k) in curves else bone.rotation_quaternion[k]
                       for k in range(4)] for t in range(T)])
        raw/=np.maximum(np.linalg.norm(raw,axis=1,keepdims=True),1e-9)
        from actor_stability import smooth_trajectory
        speeds=[]
        for t in range(T):
            a=max(0,t-5);b=min(T-1,t+5)
            speeds.append(2*math.acos(min(1.,abs(float(raw[a]@raw[b]))))/max((b-a)/fps,.001))
        # Avoid modulating a slow sway with a frame-wise noisy speed gate.
        stills=smooth_trajectory((1/(1+(np.array(speeds)/math.radians(8))**2))[:,None],fps,.35)[:,0]
        stills=np.clip(stills,0,1)
        output=[];last=None
        for t in range(T):
            q=Quaternion(raw[t].tolist());a=max(0,t-5);b=min(T-1,t+5)
            speed=2*math.acos(min(1.,abs(float(raw[a]@raw[b]))))/max((b-a)/fps,.001)
            still=float(stills[t])
            second=t/fps;seed=settings.natural_seed
            yaw=drift(second,seed);pitch=drift(second,seed+101)
            is_eye=name.startswith('Eye_')
            if is_eye:
                still=1/(1+(speed/math.radians(8))**2)
                away=1-contact_weight(t+1,intervals,settings.away_fade)
                q=q.slerp(Quaternion(),away)
                amount=math.radians(settings.eye_amount)*still if settings.natural_eyes else 0.
                # A shared gaze fluctuation, expressed in each eye's rest axes.
                inv=bone.bone.matrix_local.to_quaternion().inverted()
                delta=Quaternion(inv@Vector((0,0,1)),yaw*amount)@Quaternion(inv@Vector((1,0,0)),pitch*amount*.65)
            else:
                yaw=intermittent_sway(second,seed);pitch=0.
                amount=math.radians(settings.body_amount)*still if settings.natural_body else 0.
                if name.startswith(('Arm_','Elbow_')):
                    side=0 if name.endswith('L') else 1
                    lo=max(0,t-8);hi=min(T,t+9)
                    amount*=1-float(np.mean(visibility[lo:hi,side]))
                    amount*=.65 if name.startswith('Arm_') else .35
                    # All unseen body parts follow the same quiet excursion.
                else:amount/=3.
                inv=bone.bone.matrix_local.to_quaternion().inverted()
                delta=Quaternion(inv@Vector((0,1,0)),yaw*amount)@Quaternion(inv@Vector((1,0,0)),pitch*amount*.35)
            q=(q@delta).normalized()
            if last is not None and last.dot(q)<0:q.negate()
            output.append(list(q));last=q
        output=np.array(output)
        for k in range(4):write(action,rig,path,k,output[:,k])
    rig.animation_data.action=action
    if base_slot:
        rig.animation_data.action_slot=next(s for s in action.slots if s.identifier==base_slot.identifier)
    rig['mocap_v2_assisted_action']=action
    scene.frame_set(scene.frame_current)
    if previous.get('mocap_v2_generated_assistance') and previous.users==0:
        import bpy
        bpy.data.actions.remove(previous)
    action['mocap_v2_generated_assistance']=True
    return action

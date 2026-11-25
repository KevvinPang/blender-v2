"""Simple gesture-range UI on the existing non-destructive arm/finger solver."""
import bpy
from pathlib import Path
import numpy as np
from bpy.props import EnumProperty,IntProperty
from mathutils import Quaternion,Vector,Euler
from motion_tweak_segments import episodes

PARTS=('Wrist','Thumb0','IndexFinger1','MiddleFinger1','RingFinger1','LittleFinger1')


def mt():
    import motion_tweak
    return motion_tweak


def ensure_part(clip,bone):
    part=next((x for x in clip.parts if x.bone==bone),None)
    if part is None:part=clip.parts.add();part.bone=bone
    return part


def select(scene,jump=True):
    p=scene.motion_tweak;c=mt().current_clip(scene)
    if c and c.simple_section:
        p.edit_mode='RANGE';p.selected_bone=p.segment_part+'_'+c.side
        if jump:scene.frame_set((c.start+c.end)//2)


def add_section(scene,start,end,side,automatic=False):
    p=scene.motion_tweak;c=p.clips.add()
    c.start,c.end,c.side=start,end,side
    c.name=('左手' if side=='L' else '右手')+'动作 '+str(start)+'–'+str(end)
    c.auto_section=automatic;c.simple_section=True
    c.fade_in=c.fade_out=min(6,max(1,(end-start)//4))
    for stem in PARTS:ensure_part(c,stem+'_'+side)
    frame=scene.frame_current;physics=getattr(scene,'wiggle_enable',None)
    try:
        if physics is not None:scene.wiggle_enable=False
        scene.frame_set((start+end)//2);bpy.context.view_layer.update()
        # Position controls keep the same reference axes throughout the gesture.
        camera=scene.camera
        c.segment_basis=(p.rig.matrix_world.to_quaternion().inverted()@camera.matrix_world.to_quaternion()) if camera else Quaternion()
    finally:
        if physics is not None:scene.wiggle_enable=physics
        scene.frame_set(frame)
    return c


def sample(scene):
    p=scene.motion_tweak;m=mt();m.validate(p)
    slot=p.evaluator.animation_data.action_slot_handle
    curves={(c.data_path,c.array_index):c for c in m.curves(p.source,slot)}
    start=max(1,scene.frame_start);end=scene.frame_end
    if p.rig.get('mocap_v2_frame_count'):end=min(end,int(p.rig['mocap_v2_frame_count']))
    if end-start<2:raise ValueError('当前动作至少需要 3 帧')
    tracks={}
    for side in ('L','R'):
        names=['Arm_'+side,'Elbow_'+side,'Wrist_'+side]+[f'{stem}_{side}' for stem in PARTS[1:]]
        rows=[]
        for name in names:
            bone=p.rig.pose.bones.get(name)
            if bone is None:continue
            path=bone.path_from_id('rotation_quaternion')
            if bone.rotation_mode!='QUATERNION':raise ValueError('请先将角色动作烘焙为四元数旋转')
            rows.append([[curves[path,k].evaluate(f) if (path,k) in curves else bone.rotation_quaternion[k]
                          for k in range(4)] for f in range(start,end+1)])
        tracks[side]=np.transpose(rows,(1,0,2))
    return start,tracks


def automatic(scene):
    p=scene.motion_tweak;m=mt()
    if m._drag:raise ValueError('请先确认或取消正在拖动的调整')
    if not p.rig:
        p.rig=getattr(getattr(scene,'mocap_v2',None),'rig',None) or getattr(getattr(scene,'sam_ellen',None),'rig',None)
    if not p.source:m.setup(scene,p.rig)
    elif m.binding_changed(p):m.use_current_action(scene)
    start,tracks=sample(scene);fps=scene.render.fps/scene.render.fps_base
    visibility=None
    diagnostics=Path(p.rig.get('mocap_v2_diagnostics',''))
    if diagnostics.is_file() and (p.source.get('mocap_v2_generated_assistance') or p.source==p.rig.get('mocap_v2_base_action')):
        with np.load(diagnostics) as data:
            if 'hand_visibility' in data and len(data['hand_visibility'])>=start-1+len(tracks['L']):
                visibility=data['hand_visibility'][start-1:start-1+len(tracks['L'])].copy()
    suggestions=[(start+a,start+b,side) for si,(side,q) in enumerate(tracks.items())
                 for a,b in episodes(q,fps,None if visibility is None else visibility[:,si])]
    # Re-analysis replaces only empty suggestions, never manual/edited sections.
    for i in reversed(range(len(p.clips))):
        c=p.clips[i]
        if c.auto_section and c.enabled and not c.keys and not c.holds and not any(any(x.offset) or any(x.rotation) for x in c.parts):
            p.clips.remove(i)
    added=0
    for a,b,side in sorted(suggestions):
        if any(c.side==side and max(a,c.start)<min(b,c.end) for c in p.clips):continue
        add_section(scene,a,b,side,True);added+=1
    if p.clips:
        p.active_clip=next((i for i,c in enumerate(p.clips) if c.simple_section),0)
        select(scene)
    p.edit_mode='RANGE';p.editing=False;p.advanced_ui=False
    p.status=(f'自动找到 {added} 个动作段；边界可手动修改' if added else '未找到新的明确动作段；可手动标记，已有调整保留')
    return suggestions


def apply_section(scene):
    m=mt();p=scene.motion_tweak;m.validate(p);c=m.current_clip(scene)
    if not c or not c.simple_section:raise ValueError('请选择或手动添加一个动作段')
    if p.show_original:raise ValueError('请先切回微调结果')
    if c.start<scene.frame_start or c.end>scene.frame_end or c.end-c.start<2:
        raise ValueError('动作段需位于当前播放范围内，且至少包含 3 帧')
    previous=[{k:tuple(getattr(h,k)) if k in ('offset','rotation') else getattr(h,k) for k in m.KEY_FIELDS} for h in c.holds]
    basis=Quaternion(c.segment_basis).normalized()
    try:
        c.holds.clear()
        for part in c.parts:
            if not part.bone.endswith('_'+c.side):raise ValueError('调整部位与左右手不一致')
            if not any(part.offset) and not any(part.rotation):continue
            h=c.holds.add();h.bone=part.bone;h.frame=(c.start+c.end)//2
            h.offset=basis@Vector(part.offset)/100.
            h.rotation=basis@Euler(part.rotation,'XYZ').to_quaternion()@basis.inverted()
            h.edge_start=h.edge_end=False
        m.bake(scene)
    except Exception:
        c.holds.clear()
        for row in previous:
            h=c.holds.add()
            for key,value in row.items():setattr(h,key,value)
        raise
    p.status=(f'已统一调整第 {c.start}–{c.end} 帧；两端平滑过渡' if c.enabled else '此段调整已停用；参数保留')


def commit_drag(scene,clip,bone,frame,offset,rotation):
    if not bone.endswith('_'+clip.side):raise ValueError('请选择此动作段对应的手')
    weight=mt().hold_weight(clip,frame)
    if weight<.05:raise ValueError('请在动作段中部调整，边界留作平滑过渡')
    part=ensure_part(clip,bone);saved=tuple(part.offset),tuple(part.rotation)
    inv=Quaternion(clip.segment_basis).inverted();q=rotation.normalized()
    if q.w<0:q.negate()
    q=Quaternion(q.axis,min(q.angle/weight,1.0472)) if q.angle>1e-8 else Quaternion()
    part.offset=inv@Vector(offset)*100./weight
    part.rotation=(inv@q@inv.inverted()).to_euler('XYZ')
    try:apply_section(scene)
    except Exception:part.offset,part.rotation=saved;raise


class MTSectionAuto(bpy.types.Operator):
    bl_idname='motion_tweak.auto_sections';bl_label='读取当前动作 · 自动划分';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        try:automatic(context.scene);return {'FINISHED'}
        except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}


class MTSectionManual(bpy.types.Operator):
    bl_idname='motion_tweak.manual_section';bl_label='手动标记动作段';bl_options={'REGISTER','UNDO'}
    side:EnumProperty(items=[('L','左手',''),('R','右手','')],default='L')
    def execute(self,context):
        s=context.scene;p=s.motion_tweak
        try:
            mt().validate(p)
            start=max(s.frame_start,min(s.frame_current,s.frame_end-2))
            end=min(s.frame_end,start+30)
            add_section(s,start,end,self.side)
            p.active_clip=len(p.clips)-1;select(s)
            p.status='已添加手动区间，请设置实际动作的开始帧和结束帧'
            return {'FINISHED'}
        except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}


class MTSectionApply(bpy.types.Operator):
    bl_idname='motion_tweak.section_apply';bl_label='应用到这段动作';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        try:apply_section(context.scene);return {'FINISHED'}
        except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}


class MTSectionReset(bpy.types.Operator):
    bl_idname='motion_tweak.section_reset';bl_label='重置这段调整';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        c=mt().current_clip(context.scene)
        if not c or not c.simple_section:return {'CANCELLED'}
        old=[(tuple(x.offset),tuple(x.rotation)) for x in c.parts]
        try:
            for x in c.parts:x.offset=(0,0,0);x.rotation=(0,0,0)
            apply_section(context.scene);return {'FINISHED'}
        except Exception as exc:
            for x,values in zip(c.parts,old):x.offset,x.rotation=values
            self.report({'ERROR'},str(exc));return {'CANCELLED'}


class MTSectionMark(bpy.types.Operator):
    bl_idname='motion_tweak.section_mark';bl_label='用当前帧标记';bl_options={'REGISTER','UNDO'}
    edge:EnumProperty(items=[('START','开始',''),('END','结束','')])
    def execute(self,context):
        c=mt().current_clip(context.scene)
        if c:
            setattr(c,'start' if self.edge=='START' else 'end',context.scene.frame_current)
            c.auto_section=False
        return {'FINISHED'}


class MT_UL_Sections(bpy.types.UIList):
    def draw_item(self,context,layout,data,item,icon,active_data,active_propname,index):
        row=layout.row(align=True);row.prop(item,'enabled',text='')
        row.label(text=('左手' if item.side=='L' else '右手')+f'  {item.start}–{item.end} 帧' if item.simple_section else item.name)
        if item.holds:row.label(text='',icon='CHECKMARK')


def draw(layout,context):
    s=context.scene;p=s.motion_tweak;m=mt()
    row=layout.row();row.enabled=not bool(p.source);row.prop(p,'rig',text='角色骨架')
    layout.operator('motion_tweak.auto_sections',icon='VIEWZOOM')
    if p.source and m.binding_changed(p):
        layout.label(text='当前动作已切换，点击上方重新读取',icon='INFO')
    elif p.source:
        layout.template_list('MT_UL_Sections','',p,'clips',p,'active_clip',rows=4)
        row=layout.row(align=True)
        row.operator('motion_tweak.manual_section',text='手动左手段',icon='ADD').side='L'
        row.operator('motion_tweak.manual_section',text='手动右手段',icon='ADD').side='R'
        c=m.current_clip(s)
        if c and c.simple_section:
            row=layout.row(align=True);row.prop(c,'start',text='开始');row.prop(c,'end',text='结束')
            row=layout.row(align=True)
            row.operator('motion_tweak.section_mark',text='当前帧设为开始').edge='START'
            row.operator('motion_tweak.section_mark',text='当前帧设为结束').edge='END'
            row=layout.row(align=True);row.prop(c,'fade_in',text='渐入');row.prop(c,'fade_out',text='渐出')
            layout.prop(p,'segment_part')
            part=next((x for x in c.parts if x.bone==p.selected_bone),None)
            if part:
                box=layout.box();box.label(text=m.part_label(part.bone)+' · 整段偏移')
                for i,label in enumerate(('左右（厘米）','上下（厘米）','靠近镜头（厘米）')):box.prop(part,'offset',index=i,text=label)
                row=box.row(align=True)
                for i,label in enumerate(('俯仰','侧转','侧倾')):row.prop(part,'rotation',index=i,text=label)
            layout.prop(p,'editing',text='也可点击手部，用 G 移动 / R 旋转',toggle=True)
            layout.operator('motion_tweak.section_apply',icon='CHECKMARK')
            layout.label(text='改范围、偏移或开关后，点击应用')
            row=layout.row(align=True);row.operator('motion_tweak.preview',text='预览这段',icon='PLAY');row.operator('motion_tweak.compare',text='原动作 / 调整后')
            row=layout.row(align=True);row.operator('motion_tweak.section_reset');row.operator('motion_tweak.remove',text='删除区间',icon='X')
        elif c:layout.label(text='此段是旧版记录，可在高级面板继续编辑')
    import textwrap
    for line in textwrap.wrap(p.status,26):layout.label(text=line)
    layout.label(text='自动边界是建议，可手动改起止帧',icon='INFO')
    layout.prop(p,'advanced_ui')


CLASSES=(MTSectionAuto,MTSectionManual,MTSectionApply,MTSectionReset,MTSectionMark,MT_UL_Sections)

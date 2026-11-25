bl_info = {
    'name':'FlowHMR 2 MMR', 'author':'小二, GPT', 'version':(1,3,1),
    'blender':(3,6,0), 'location':'3D 视图 > 侧栏 > FlowHMR 2 MMR',
    'description':'将 FlowHMR 的单人或多人动作重定向到 MMR，保留原始时间和站位', 'category':'Animation',
}

import bpy
from bpy.props import BoolProperty, IntProperty, IntVectorProperty, FloatProperty, FloatVectorProperty, StringProperty, EnumProperty, PointerProperty, CollectionProperty
from . import mapping, runtime, baking, layers, presets, multiplayer


def armature_poll(self,obj):return obj.type=='ARMATURE'


def target_poll(self,obj):return mapping.is_mmr(obj) if self.source_layout else armature_poll(self,obj)


def parameter_changed(self,context):
    scene=self.id_data
    if isinstance(scene,bpy.types.Scene):
        runtime.update(scene,True,self if isinstance(self,MMRB_Settings) else None)


def source_changed(self,context):
    if not self.source_layout or self.status!='NONE' or runtime.BUSY:return
    fill_map(self)
    sync_deform_binding(self)
    try:
        self.frame_start,self.frame_end=multiplayer.source_span(self.source)
        self.last_message='已读取源动作帧范围。'
        self.action_name=f'MMR · {self.source.name}'
    except ValueError as error:self.last_message=str(error)


def sync_deform_binding(cfg):
    scene=cfg.id_data
    if runtime.BUSY or not isinstance(scene,bpy.types.Scene) or cfg.status!='NONE':return
    # Always replace the old association, including when the new rig has no match.
    matches=mapping.deform_candidates(scene,cfg.target,exclude=(cfg.source,))
    cfg.deform=matches[0] if len(matches)==1 else None
    if not cfg.target:cfg.deform_notice=''
    elif not mapping.is_mmr(cfg.target):cfg.deform_notice='请选择 MMR 控制骨架。'
    elif not matches:cfg.deform_notice='未找到有效绑定，请手动选择 MMD 骨架。'
    elif len(matches)>1:cfg.deform_notice=f'找到 {len(matches)} 套绑定，请手动选择 MMD 骨架。'
    else:cfg.deform_notice=''


def target_changed(self,context):
    sync_deform_binding(self)


def deform_changed(self,context):
    if runtime.BUSY or self.status!='NONE':return
    if self.deform:
        self.deform_notice='MMD 骨架不能与 MMR 控制骨架相同。' if self.deform==self.target else ''
    else:self.deform_notice='未指定 MMD 骨架，颈头高度调整不可用。'


class MMRB_BoneMap(bpy.types.PropertyGroup):
    target:StringProperty(name='MMR 控制骨')
    source:StringProperty(name='源骨骼',description='对应的源骨骼名称；手指及脚趾可以留空')
    expanded:BoolProperty(name='展开骨骼微调',default=False)
    offset_space:EnumProperty(name='微调坐标',items=[('LOCAL','骨骼局部','沿当前骨骼自身轴移动和旋转，局部位移随骨架缩放'),('WORLD','世界坐标','沿世界 XYZ 移动，并绕自身关节的世界轴旋转')],default='LOCAL',update=parameter_changed)
    rotation_offset:FloatVectorProperty(name='旋转微调',size=3,subtype='EULER',default=(0,0,0),description='在自动重定向之后叠加此骨骼的 XYZ 旋转；不会修改源动画',update=parameter_changed)
    location_offset:FloatVectorProperty(name='位移微调',size=3,subtype='TRANSLATION',unit='LENGTH',precision=4,default=(0,0,0),description='单独移动此 MMR 控制骨；连接骨的位移在烘焙后保留为原生约束',update=parameter_changed)


class MMRB_Visibility(bpy.types.PropertyGroup):
    obj:PointerProperty(type=bpy.types.Object)
    path:StringProperty()
    expression:StringProperty()
    modifier_name:StringProperty()
    vertex_group:StringProperty()


class MMRB_Settings(bpy.types.PropertyGroup):
    show_config:BoolProperty(name='配置保存 / 读取',default=False,description='展开或收起 JSON 配置；不改变当前参数')
    config_name:StringProperty(name='配置名称',default='默认配置',description='按名称保存到本插件的 configs 目录；同名保存会覆盖 JSON，并保留上一份 .json.bak 备份')
    config_selected:EnumProperty(name='已保存配置',items=presets.enum_items,update=presets.selection_changed,description='选择插件目录中的 JSON 配置，再点击读取；保留当前选择的源骨架和目标骨架')
    source_layout:BoolProperty(default=False,options={'HIDDEN'})
    visibility:CollectionProperty(type=MMRB_Visibility)
    source:PointerProperty(name='源动作骨架',type=bpy.types.Object,poll=armature_poll,update=source_changed)
    target:PointerProperty(name='MMR 控制骨架',type=bpy.types.Object,poll=target_poll,update=target_changed,description='选择或更换 MMR 控制器后，自动查找并填入它绑定的 MMD 变形骨架')
    deform:PointerProperty(name='MMD 变形骨架',type=bpy.types.Object,poll=armature_poll,update=deform_changed,description='随 MMR 控制器选择自动填入；找不到绑定、存在多套候选或特殊绑定时可手动修正。用于骨轴标定、颈头比例微调和脚底定位')
    deform_notice:StringProperty(default='',options={'HIDDEN'})
    mode:EnumProperty(name='输入方式',items=[('SOURCE','源骨架重定向','从独立的 SMPL/FBX 骨架建立实时约束，不立即烘焙'),('EXISTING','接管现有动作','保留已有控制器动画，仅接管可调的颈头高度层')],default='SOURCE')
    status:EnumProperty(items=[('NONE','未建立',''),('LIVE','实时预览',''),('PAUSED','预览暂停',''),('BAKED','已烘焙','')],default='NONE')
    session_id:StringProperty(options={'HIDDEN'})
    helpers:PointerProperty(type=bpy.types.Collection,options={'HIDDEN'})
    original_action:PointerProperty(type=bpy.types.Action,options={'HIDDEN'})
    last_baked_action:PointerProperty(type=bpy.types.Action,options={'HIDDEN'})
    bone_map:CollectionProperty(type=MMRB_BoneMap)
    show_mapping:BoolProperty(name='检查 / 修改骨骼映射',default=False)
    mapping_filter:StringProperty(name='筛选骨骼',description='按 MMR 控制骨名、源骨骼名或 MMD 骨名筛选')
    auto_fk:BoolProperty(name='自动切换 FK',default=True,description='MMR 默认 IK_FK=0；重定向期间切为 1，并防止原有关键帧切回 IK。拆除时恢复原状态')
    frame_start:IntProperty(name='起始帧',default=2,min=-100000,max=1000000)
    frame_end:IntProperty(name='结束帧',default=250,min=-100000,max=1000000)
    motion_scale:FloatProperty(name='位移比例',default=1.,min=.1,max=4.,precision=3,description='多人模式中 1 保留源位移；单人模式在自动腿长比例上乘以该值',update=parameter_changed)
    heading:FloatProperty(name='整体朝向',subtype='ANGLE',default=0.,description='绕世界 Z 轴旋转源动作及位移',update=parameter_changed)
    root_offset:FloatVectorProperty(name='根骨位置偏移',size=3,subtype='TRANSLATION',unit='LENGTH',precision=4,default=(0,0,0),description='世界坐标 XYZ 的整体平移；源重定向与接管现有动作都可用，烘焙时写入新动作',update=parameter_changed)
    head_angles:FloatVectorProperty(name='头部角度',size=3,subtype='EULER',default=(0,0,0),description='参考姿势坐标下的 X 俯仰、Y 侧倾、Z 偏航微调',update=parameter_changed)
    neck_angles:FloatVectorProperty(name='颈部角度',size=3,subtype='EULER',default=(0,0,0),description='在完整旋转传递之后调整颈部角度',update=parameter_changed)
    chest_angles:FloatVectorProperty(name='胸部角度',size=3,subtype='EULER',default=(0,0,0),description='调整上胸的姿势，头和手臂仍按各自映射传递完整朝向',update=parameter_changed)
    head_axis:FloatProperty(name='头骨参考轴校准',default=1.,min=0,max=1,subtype='FACTOR',description='1 消除源 Head 与 MMD 頭 在参考姿势中的骨轴差异；0 保留上一版完整参考旋转。不会改变头部大小或关节位置',update=parameter_changed)
    neck_axis:FloatProperty(name='颈段参考轴校准',default=0.,min=0,max=1,subtype='FACTOR',description='可选：使颈段方向靠近源骨架。会改变头部关节位置；默认 0 保留已经确认的颈部姿势',update=parameter_changed)
    height_cm:FloatProperty(name='颈头抬升（厘米）',default=2.,min=0,max=5,precision=2,description='沿胸颈骨链合计增加的高度；0 恢复目标原比例。保持头部尺寸与旋转',update=parameter_changed)
    height_root_ratio:FloatProperty(name='分配给颈根',default=.5,min=0,max=1,subtype='FACTOR',description='抬升分配给颈根的比例，其余增加颈段长度',update=parameter_changed)
    height_available:BoolProperty(options={'HIDDEN'})
    hand_pair_strength:FloatProperty(name='双手相对位置修正',default=1.,min=0,max=1,subtype='FACTOR',description='修正窄肩角色两手靠拢时的位置关系；0 使用纯 FK，1 使用完整配对修正',update=parameter_changed)
    finger_strength:FloatProperty(name='手指动作强度',default=1.,min=0,max=1,subtype='FACTOR',description='1 传递完整指节动作，0 保持随手掌运动的参考手型',update=parameter_changed)
    leg_position:BoolProperty(name='按源位置求解双腿',default=True,description='按目标腿长求解落脚位置，再输出到 FK 控制器，避免腿部拉伸',update=parameter_changed)
    ground:BoolProperty(name='脚底贴合地面',default=True,description='根据源脚踝最低高度和目标脚底几何计算接触高度，保留抬脚动作',update=parameter_changed)
    floor_height:FloatProperty(name='地面高度',subtype='DISTANCE',default=0.,description='目标脚底接触平面的世界 Z 坐标',update=parameter_changed)
    foot_lock:BoolProperty(name='支撑段防滑',default=True,description='使用源骨架低高度、低速度区间锁定脚底接触点；可用重新分析刷新',update=parameter_changed)
    contact_height_cm:FloatProperty(name='接触高度阈值（厘米）',default=3.,min=.1,max=15.,description='源脚踝离最低高度小于此值时，才可能判为支撑；修改后重新分析')
    contact_speed_cm:FloatProperty(name='接触速度阈值（厘米/秒）',default=20.,min=1,max=100,description='源脚踝水平速度低于此值时，才可能判为支撑；修改后重新分析')
    contact_count:IntProperty(default=0,options={'HIDDEN'})
    bake_step:IntProperty(name='帧步长',default=1,min=1,max=10,description='1 保留每一帧；更大值减少关键帧，但会引入插值误差')
    action_name:StringProperty(name='新动作名称',default='MMR_动捕重定向_已烘焙')
    last_message:StringProperty(default='默认只建立实时预览，点击烘焙后才写入新动作。')
    last_bake_error:FloatProperty(options={'HIDDEN'})

    @property
    def mapped_targets(self):return {e.target for e in self.bone_map}|{'root','torso'}


class MMRB_MultiSettings(bpy.types.PropertyGroup):
    enabled:BoolProperty(name='多人模式',default=False)
    pairs:CollectionProperty(type=MMRB_Settings)
    active_index:IntProperty(default=0,min=0)
    last_message:StringProperty()
    changed_range:BoolProperty(default=False,options={'HIDDEN'})
    original_range:IntVectorProperty(size=2,options={'HIDDEN'})
    expanded_range:IntVectorProperty(size=2,options={'HIDDEN'})


def selected_settings(scene):
    group=scene.mmrb_multi
    if not group.enabled:return scene.mmrb
    return group.pairs[group.active_index] if group.active_index<len(group.pairs) else None


def add_pair(scene,source=None):
    group=scene.mmrb_multi
    cfg=group.pairs.add()
    cfg.source_layout=True
    # Copy pose/ground preferences. Source timing and scene placement belong to this pair.
    for key in presets.SETTINGS:
        if key not in {'mode','frame_start','frame_end','motion_scale','heading','root_offset','action_name'}:
            value=getattr(scene.mmrb,key)
            setattr(cfg,key,value[:] if hasattr(value,'__len__') and not isinstance(value,str) else value)
    if source is not None:cfg.source=source
    return cfg


def fill_map(cfg):
    offsets={e.target:(tuple(e.rotation_offset),tuple(e.location_offset),e.offset_space,e.expanded) for e in cfg.bone_map}
    cfg.bone_map.clear()
    for target,source in mapping.auto_map(cfg.source).items():
        e=cfg.bone_map.add();e.target=target;e.source=source
        if target in offsets:e.rotation_offset,e.location_offset,e.offset_space,e.expanded=offsets[target]


def detect(scene,active=None):
    cfg=scene.mmrb
    if cfg.status!='NONE':raise ValueError('请先拆除当前会话再重新识别。')
    rigs=[o for o in scene.objects if mapping.is_mmr(o)]
    if active in rigs:cfg.target=active
    elif not cfg.target and rigs:cfg.target=rigs[0]
    sync_deform_binding(cfg)
    sources=[o for o in scene.objects if o.type=='ARMATURE' and o not in [cfg.target,cfg.deform] and mapping.auto_map(o).get('spine_fk')]
    sources=[o for o in sources if not o.name.startswith('源骨架对照')]
    if not cfg.source and sources:
        cfg.source=next((o for o in sources if o.name=='Person 1'),sources[0])
    fill_map(cfg)
    cfg.frame_start=scene.frame_start;cfg.frame_end=scene.frame_end
    if cfg.source and cfg.source.animation_data and cfg.source.animation_data.action:
        a,b=cfg.source.animation_data.action.frame_range
        cfg.frame_start=max(int(a),scene.frame_start);cfg.frame_end=int(b)
        # RGB MoCap FBX has a rest-pose frame 1. Its motion starts at frame 2.
        if cfg.source.name.startswith('Person ') and int(a)==1:cfg.frame_start=max(2,cfg.frame_start)
    if '颈头抬升_厘米' in scene:
        cfg.height_cm=float(scene['颈头抬升_厘米']);cfg.height_root_ratio=float(scene.get('抬升分配_颈根占比',.5))
    cfg.last_message='已识别骨架；请检查对象和映射，再建立预览。'


class MMRB_OT_save_config(bpy.types.Operator):
    bl_idname='mmrb.save_config';bl_label='保存配置';bl_description='把映射、逐骨旋转/位移、根骨偏移及其他参数保存为插件目录中的 JSON；同名文件保留上一份备份'
    def execute(self,context):
        c=context.scene.mmrb
        try:
            path=presets.save(c,c.config_name)
            c.config_selected=path.name;c.config_name=path.stem
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
        c.last_message='已保存配置：'+path.name
        self.report({'INFO'},'配置已保存：'+str(path));return {'FINISHED'}


class MMRB_OT_load_config(bpy.types.Operator):
    bl_idname='mmrb.load_config';bl_label='读取配置';bl_description='读取选中的 JSON，恢复参数与逐骨微调；保留当前骨架选择。已有会话会重建预览，已烘焙的动作保留，不自动烘焙';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        c=context.scene.mmrb
        try:
            name=c.config_selected if c.config_selected and c.config_selected!=presets.NONE else c.config_name
            path,values=presets.read(c,name)
            active=presets.apply(context.scene,values)
            presets.refresh();c.config_selected=path.name;c.config_name=path.stem
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
        c.last_message='已读取配置：'+path.name+('；预览已更新，未自动烘焙。' if active else '；可建立实时预览。')
        self.report({'INFO'},c.last_message);return {'FINISHED'}


class MMRB_OT_refresh_configs(bpy.types.Operator):
    bl_idname='mmrb.refresh_configs';bl_label='刷新配置列表';bl_description='手动复制或修改 JSON 后，刷新插件 configs 目录中的配置列表'
    def execute(self,context):
        c=context.scene.mmrb
        selected=c.config_selected
        try:items=presets.refresh()
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
        c.config_selected=selected if selected in {item[0] for item in items} else presets.NONE
        return {'FINISHED'}


class MMRB_OT_open_configs(bpy.types.Operator):
    bl_idname='mmrb.open_configs';bl_label='打开配置目录';bl_description='打开当前插件安装目录下的 configs 文件夹，可复制、分享或备份其中的 JSON 配置'
    def execute(self,context):
        try:
            path=presets.directory();path.mkdir(parents=True,exist_ok=True)
            return bpy.ops.wm.path_open(filepath=str(path))
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}


class MMRB_OT_detect(bpy.types.Operator):
    bl_idname='mmrb.detect';bl_label='自动识别骨架';bl_description='识别源骨架、MMR 控制器及其实际 MMD 变形骨架';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        try:detect(context.scene,context.active_object)
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
        return {'FINISHED'}


class MMRB_OT_map(bpy.types.Operator):
    bl_idname='mmrb.auto_map';bl_label='重新自动映射';bl_description='支持 RGB MoCap/SMPL FBX 与带命名空间的 Mixamo 名称';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        cfg=selected_settings(context.scene)
        if cfg is None or cfg.status!='NONE':return {'CANCELLED'}
        fill_map(cfg);return {'FINISHED'}


class MMRB_OT_collapse(bpy.types.Operator):
    bl_idname='mmrb.collapse_mapping';bl_label='全部收起';bl_description='收起映射列表及每根骨骼的细节，不改变任何参数'
    def execute(self,context):
        c=selected_settings(context.scene)
        if c is None:return {'CANCELLED'}
        for e in c.bone_map:e.expanded=False
        c.show_mapping=False
        return {'FINISHED'}


class MMRB_OT_reset_offset(bpy.types.Operator):
    bl_idname='mmrb.reset_offset';bl_label='重置此骨微调';bl_description='只清零选中骨骼的旋转和位移，其他骨骼保持不变';bl_options={'REGISTER','UNDO'}
    target:StringProperty()
    def execute(self,context):
        c=selected_settings(context.scene)
        if c is None:return {'CANCELLED'}
        if c.status=='BAKED':self.report({'WARNING'},'请先恢复实时预览再调整。');return {'CANCELLED'}
        if self.target=='root':c.root_offset=(0,0,0)
        else:
            e=next((e for e in c.bone_map if e.target==self.target),None)
            if e:e.rotation_offset=(0,0,0);e.location_offset=(0,0,0)
        return {'FINISHED'}


class MMRB_OT_start(bpy.types.Operator):
    bl_idname='mmrb.start';bl_label='建立实时预览';bl_description='只建立可调预览；不烘焙、不覆盖原动作';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        try:runtime.start(context.scene)
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
        self.report({'INFO'},'实时预览已建立，尚未烘焙。');return {'FINISHED'}


class MMRB_OT_pause(bpy.types.Operator):
    bl_idname='mmrb.pause';bl_label='暂停 / 恢复实时预览';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        try:baking.resume(context.scene)
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
        return {'FINISHED'}


class MMRB_OT_refresh(bpy.types.Operator):
    bl_idname='mmrb.refresh';bl_label='刷新当前帧';bl_description='手动修改源骨骼后，重新计算当前帧的预览；播放或切换时间轴时会自动计算'
    def execute(self,context):
        cfg=selected_settings(context.scene)
        if cfg is None:return {'CANCELLED'}
        runtime.update(context.scene,True,cfg)
        return {'FINISHED'}


class MMRB_OT_remove(bpy.types.Operator):
    bl_idname='mmrb.remove';bl_label='拆除并还原';bl_description='移除本插件的预览层，恢复原动作和原 IK/FK 状态；另存的烘焙动作仍保留';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        try:runtime.remove(context.scene)
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
        return {'FINISHED'}


class MMRB_OT_analyze(bpy.types.Operator):
    bl_idname='mmrb.analyze';bl_label='重新分析脚部支撑';bl_description='读取源动画更新接触区间，不写入动画关键帧'
    def execute(self,context):
        scene=context.scene
        cfg=selected_settings(scene)
        if cfg is None:return {'CANCELLED'}
        oldframe,subframe=scene.frame_current,scene.frame_subframe
        try:
            runtime.BUSY=True
            if cfg.source_layout and not runtime.in_range(scene,cfg):scene.frame_set(cfg.frame_start)
            session=runtime.get_session(scene,cfg)
            if not session:raise ValueError('请先建立预览。')
            session.analyze_contacts()
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
        finally:
            scene.frame_set(oldframe,subframe=subframe)
            runtime.BUSY=False
        runtime.update(scene,True,cfg);return {'FINISHED'}


class MMRB_OT_bake(bpy.types.Operator):
    bl_idname='mmrb.bake';bl_label='烘焙到新动作';bl_description='显式烘焙 MMR 控制器并写入 FK 开关；保留原动作和可调颈头高度层';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        try:a=baking.bake(context.scene)
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
        self.report({'INFO'},'已生成新动作：'+a.name);return {'FINISHED'}


class MMRB_OT_multi_manage(bpy.types.Operator):
    bl_idname='mmrb.multi_manage';bl_label='管理人物配对';bl_options={'REGISTER','UNDO'}
    action:EnumProperty(items=[('DETECT','识别源骨架','从合并 FBX 中添加人物'),
                               ('ADD','添加配对','手动添加一对骨架'),('REMOVE','移除配对','移除当前未运行的配对')])
    def execute(self,context):
        scene=context.scene;group=scene.mmrb_multi
        if scene.mmrb.status!='NONE' or any(cfg.status!='NONE' for cfg in group.pairs):
            self.report({'WARNING'},'请先拆除当前会话再更换配对。');return {'CANCELLED'}
        if self.action=='DETECT':
            sources=multiplayer.discover_sources(scene)
            if not sources:
                self.report({'WARNING'},'没有找到带动作的源骨架，请先导入 FBX。');return {'CANCELLED'}
            existing={cfg.source for cfg in group.pairs if cfg.source}
            for source in sources:
                if source not in existing:add_pair(scene,source)
            group.last_message=f'已识别 {len(sources)} 副源骨架，请给每位人物选择目标。'
        elif self.action=='ADD':
            add_pair(scene)
            group.active_index=len(group.pairs)-1
        elif group.active_index<len(group.pairs):
            group.pairs.remove(group.active_index)
            group.active_index=min(group.active_index,max(0,len(group.pairs)-1))
        return {'FINISHED'}


class MMRB_OT_multi_run(bpy.types.Operator):
    bl_idname='mmrb.multi_run';bl_label='运行多人重定向';bl_options={'REGISTER','UNDO'}
    action:EnumProperty(items=[('PREVIEW','预览全部','在当前场景同时建立或恢复每人的实时预览'),
                               ('BAKE','一键烘焙全部','建立未运行的配对，按每人的原始帧号烘焙全部动作'),
                               ('PAUSE','暂停预览','暂停所有实时预览'),('REMOVE','拆除并还原','恢复目标原设置，保留已生成的动作数据')])
    def execute(self,context):
        scene=context.scene;group=scene.mmrb_multi
        try:
            if self.action=='PREVIEW':message=f'{multiplayer.preview_all(scene)} 人已建立同屏预览。'
            elif self.action=='BAKE':message=f'{multiplayer.bake_all(scene)} 人已烘焙，动作保留在各自的目标骨架。'
            elif self.action=='REMOVE':
                multiplayer.remove_all(scene);message='已恢复各目标的原设置；已烘焙的动作仍保留。'
            else:
                for cfg in group.pairs:
                    if cfg.status=='LIVE':cfg.status='PAUSED'
                message='已暂停多人预览。'
        except (ValueError,RuntimeError,ReferenceError,KeyError,TypeError) as error:
            group.last_message=str(error);self.report({'ERROR'},str(error));return {'CANCELLED'}
        group.last_message=message;self.report({'INFO'},message);return {'FINISHED'}


_STATUS_TEXT={'NONE':'待设置','LIVE':'实时预览','PAUSED':'已暂停','BAKED':'已烘焙'}


class MMRB_UL_pairs(bpy.types.UIList):
    def draw_item(self,context,layout,data,item,icon,active_data,active_propname,index=0,flt_flag=0):
        row=layout.row(align=True)
        row.label(text=item.source.name if item.source else f'配对 {index+1}',icon='ARMATURE_DATA')
        row.label(text=_STATUS_TEXT[item.status])


def draw_mapping(layout,cfg):
    row=layout.row(align=True)
    row.prop(cfg,'show_mapping',icon='TRIA_DOWN' if cfg.show_mapping else 'TRIA_RIGHT',emboss=False)
    if not cfg.show_mapping:return
    row.operator('mmrb.collapse_mapping',text='',icon='TRIA_UP')
    layout.prop(cfg,'mapping_filter',text='',icon='VIEWZOOM')
    remap=layout.row();remap.enabled=cfg.status=='NONE';remap.operator('mmrb.auto_map')
    if cfg.status=='BAKED':layout.label(text='恢复实时预览后可继续微调。',icon='INFO')
    for entry in cfg.bone_map:
        label=mapping.DEFORM.get(entry.target,entry.target)
        if cfg.mapping_filter and cfg.mapping_filter.casefold() not in (entry.target+' '+entry.source+' '+label).casefold():continue
        box=layout.box();row=box.row(align=True)
        row.prop(entry,'expanded',text=label+'  /  '+entry.target,icon='TRIA_DOWN' if entry.expanded else 'TRIA_RIGHT',emboss=False)
        row=box.row();row.enabled=cfg.status=='NONE'
        if cfg.source:row.prop_search(entry,'source',cfg.source.data,'bones',text='')
        else:row.prop(entry,'source',text='')
        if entry.expanded:
            col=box.column();col.enabled=cfg.status!='BAKED'
            col.prop(entry,'offset_space');col.prop(entry,'rotation_offset');col.prop(entry,'location_offset')
            col.operator('mmrb.reset_offset',icon='LOOP_BACK').target=entry.target
    layout.operator('mmrb.collapse_mapping',icon='TRIA_UP')


def draw_multi(layout,scene):
    group=scene.mmrb_multi
    active=any(cfg.status!='NONE' for cfg in group.pairs)
    ready=bool(group.pairs) and all(cfg.source and cfg.target for cfg in group.pairs)
    header=layout.row(align=True);header.enabled=not active
    header.operator('mmrb.multi_manage',text='识别源骨架',icon='VIEWZOOM').action='DETECT'
    header.operator('mmrb.multi_manage',text='',icon='ADD').action='ADD'
    header.operator('mmrb.multi_manage',text='',icon='REMOVE').action='REMOVE'
    layout.template_list('MMRB_UL_pairs','',group,'pairs',group,'active_index',rows=3)
    cfg=selected_settings(scene)
    if cfg is not None:
        box=layout.box();box.use_property_split=True
        pair=box.column(align=True);pair.enabled=not active
        pair.prop(cfg,'source',text='源骨架');pair.prop(cfg,'target',text='目标');pair.prop(cfg,'deform',text='变形骨架')
        if cfg.deform_notice:
            for i in range(0,len(cfg.deform_notice),22):box.label(text=cfg.deform_notice[i:i+22],icon='INFO' if i==0 else 'NONE')
        if cfg.source:box.label(text=f'源动作 · {cfg.frame_start}–{cfg.frame_end} 帧')
    run=layout.row(align=True);run.enabled=ready
    run.operator('mmrb.multi_run',text='预览全部',icon='PLAY').action='PREVIEW'
    bake=run.row(align=True);bake.enabled=any(c.status!='BAKED' for c in group.pairs)
    bake.operator('mmrb.multi_run',text='一键烘焙全部',icon='ACTION').action='BAKE'
    if active:
        row=layout.row(align=True)
        if any(c.status=='LIVE' for c in group.pairs):row.operator('mmrb.multi_run',text='暂停预览',icon='PAUSE').action='PAUSE'
        row.operator('mmrb.multi_run',text='拆除并还原',icon='LOOP_BACK').action='REMOVE'
    if group.last_message:
        for i in range(0,len(group.last_message),25):layout.label(text=group.last_message[i:i+25])
    if cfg is not None:
        if cfg.status=='PAUSED' and cfg.last_message:
            for i in range(0,len(cfg.last_message),25):layout.label(text=cfg.last_message[i:i+25])
        draw_mapping(layout,cfg)


class MMRB_PT_main(bpy.types.Panel):
    bl_label='FlowHMR 2 MMR';bl_idname='MMRB_PT_main';bl_space_type='VIEW_3D';bl_region_type='UI';bl_category='FlowHMR 2 MMR'
    def draw(self,context):
        l=self.layout;c=context.scene.mmrb
        group=context.scene.mmrb_multi
        mode=l.row();mode.enabled=c.status=='NONE' and not any(p.status!='NONE' for p in group.pairs)
        mode.prop(group,'enabled',toggle=True,icon='COMMUNITY')
        if group.enabled:draw_multi(l,context.scene);return
        box=l.box()
        box.prop(c,'show_config',icon='TRIA_DOWN' if c.show_config else 'TRIA_RIGHT',emboss=False)
        if c.show_config:
            box.prop(c,'config_name')
            row=box.row(align=True);row.prop(c,'config_selected',text='')
            row.operator('mmrb.refresh_configs',text='',icon='FILE_REFRESH')
            row=box.row(align=True)
            row.operator('mmrb.save_config',icon='FILE_TICK');row.operator('mmrb.load_config',icon='IMPORT')
        setup=l.column();setup.enabled=c.status=='NONE'
        setup.operator('mmrb.detect',icon='VIEWZOOM')
        setup.prop(c,'mode');setup.prop(c,'source');setup.prop(c,'target');setup.prop(c,'deform')
        if c.deform_notice:
            for i in range(0,len(c.deform_notice),22):setup.label(text=c.deform_notice[i:i+22],icon='INFO' if i==0 else 'NONE')
        if c.mode=='SOURCE':setup.prop(c,'auto_fk')
        draw_mapping(l,c)
        row=l.row(align=True)
        if c.status=='NONE':row.operator('mmrb.start',icon='PLAY')
        else:
            row.operator('mmrb.pause',text='暂停预览' if c.status=='LIVE' else '恢复实时预览',icon='PAUSE' if c.status=='LIVE' else 'PLAY')
            row.operator('mmrb.remove',text='拆除并还原',icon='LOOP_BACK')
        l.label(text={'NONE':'尚未建立','LIVE':'实时预览 · 未烘焙','PAUSED':'预览已暂停','BAKED':'已生成独立控制器动作'}[c.status],icon='INFO')
        if c.last_message:
            # Avoid a single very long status row in a narrow sidebar.
            for i in range(0,len(c.last_message),25):l.label(text=c.last_message[i:i+25])
        if c.status in {'LIVE','PAUSED'} and c.mode=='SOURCE':l.operator('mmrb.refresh',icon='FILE_REFRESH')


# A new UI-only ID lets older projects pick up the closed default, too.
class MMRB_PT_pose_proportions(bpy.types.Panel):
    bl_label='姿态与比例';bl_idname='MMRB_PT_pose_proportions';bl_parent_id='MMRB_PT_main';bl_space_type='VIEW_3D';bl_region_type='UI';bl_category='FlowHMR 2 MMR';bl_options={'DEFAULT_CLOSED'}
    @classmethod
    def poll(cls,context):return selected_settings(context.scene) is not None
    def draw(self,context):
        l=self.layout;c=selected_settings(context.scene)
        root=l.box();root.enabled=c.status!='BAKED';root.label(text='根骨位置偏移 · 世界 XYZ')
        root.prop(c,'root_offset',text='')
        root.operator('mmrb.reset_offset',text='根骨偏移归零',icon='LOOP_BACK').target='root'
        box=l.box();box.label(text='颈头高度 · 烘焙后仍可调')
        box.enabled=c.status=='NONE' or c.height_available
        box.prop(c,'height_cm');box.prop(c,'height_root_ratio')
        axis=l.column();axis.enabled=c.status!='BAKED' or c.mode=='EXISTING'
        axis.prop(c,'head_axis');axis.prop(c,'neck_axis')
        axis.prop(c,'head_angles');axis.prop(c,'neck_angles')
        if c.mode=='EXISTING':l.label(text='接管模式保留现有全身动作。');return
        col=l.column();col.enabled=c.status!='BAKED'
        col.prop(c,'motion_scale');col.prop(c,'heading')
        col.prop(c,'chest_angles')
        col.prop(c,'hand_pair_strength');col.prop(c,'finger_strength')


class MMRB_PT_feet(bpy.types.Panel):
    bl_label='脚部与地面';bl_idname='MMRB_PT_feet';bl_parent_id='MMRB_PT_main';bl_space_type='VIEW_3D';bl_region_type='UI';bl_category='FlowHMR 2 MMR';bl_options={'DEFAULT_CLOSED'}
    @classmethod
    def poll(cls,context):
        cfg=selected_settings(context.scene)
        return cfg is not None and cfg.mode=='SOURCE'
    def draw(self,context):
        l=self.layout;c=selected_settings(context.scene);l.enabled=c.status!='BAKED'
        l.prop(c,'leg_position');l.prop(c,'ground');l.prop(c,'floor_height');l.prop(c,'foot_lock')
        l.prop(c,'contact_height_cm');l.prop(c,'contact_speed_cm')
        row=l.row();row.enabled=c.status in {'LIVE','PAUSED'};row.operator('mmrb.analyze')
        l.label(text=f'已分析 {c.contact_count} 个支撑区间')


class MMRB_PT_bake(bpy.types.Panel):
    bl_label='烘焙';bl_idname='MMRB_PT_bake';bl_parent_id='MMRB_PT_main';bl_space_type='VIEW_3D';bl_region_type='UI';bl_category='FlowHMR 2 MMR'
    @classmethod
    def poll(cls,context):return selected_settings(context.scene) is not None
    def draw(self,context):
        l=self.layout;c=selected_settings(context.scene)
        if c.source_layout:l.label(text=f'当前人物 · 原片 {c.frame_start}–{c.frame_end} 帧')
        else:
            row=l.row(align=True);row.prop(c,'frame_start');row.prop(c,'frame_end')
        l.prop(c,'bake_step');l.prop(c,'action_name')
        if not c.source_layout:
            row=l.row();row.enabled=c.status in {'LIVE','PAUSED'};row.operator('mmrb.bake',icon='ACTION')
        l.label(text='新动作 + FK 开关；原动作保留。' if c.mode=='SOURCE' else '新动作保留原有 IK/FK 状态。')
        l.label(text='颈头微调保留为原生约束。')


CLASSES=(MMRB_BoneMap,MMRB_Visibility,MMRB_Settings,MMRB_MultiSettings,MMRB_OT_save_config,MMRB_OT_load_config,MMRB_OT_refresh_configs,MMRB_OT_open_configs,MMRB_OT_detect,MMRB_OT_map,MMRB_OT_collapse,MMRB_OT_reset_offset,MMRB_OT_start,MMRB_OT_pause,MMRB_OT_refresh,MMRB_OT_remove,MMRB_OT_analyze,MMRB_OT_bake,MMRB_OT_multi_manage,MMRB_OT_multi_run,MMRB_UL_pairs,MMRB_PT_main,MMRB_PT_pose_proportions,MMRB_PT_feet,MMRB_PT_bake)


def register():
    presets.refresh()
    for cls in CLASSES:bpy.utils.register_class(cls)
    bpy.types.Scene.mmrb=PointerProperty(type=MMRB_Settings)
    bpy.types.Scene.mmrb_multi=PointerProperty(type=MMRB_MultiSettings)
    runtime.register_handlers()
    runtime.load_finished(None)


def unregister():
    runtime.unregister_handlers()
    del bpy.types.Scene.mmrb_multi
    del bpy.types.Scene.mmrb
    for cls in reversed(CLASSES):bpy.utils.unregister_class(cls)

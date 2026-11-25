bl_info={'name':'动捕','author':'Ellen V2','version':(2,1,1),'blender':(5,2,0),
         'location':'3D 视图 > 侧栏 > 动捕','description':'V2 视频动捕、自然辅助及原表情工作流兼容','category':'Animation'}
import hashlib,json,subprocess,sys,time,uuid
from pathlib import Path
from types import SimpleNamespace
import bpy
import numpy as np
from bpy_extras.io_utils import ImportHelper
from bpy.props import BoolProperty,IntProperty,FloatProperty,StringProperty,EnumProperty,PointerProperty,CollectionProperty
from bpy.app.handlers import persistent

ROOT=Path(__file__).resolve().parents[1]
V2=ROOT/'新逻辑/v2'
for path in (ROOT/'tools',V2/'blender_bridge'):
    if str(path) not in sys.path:sys.path.insert(0,str(path))
_job=None


def rig_changed(settings,context):
    if settings.rig and hasattr(settings.id_data,'sam_ellen'):
        settings.id_data.sam_ellen.rig=settings.rig


def fingerprint(rig):
    from natural_motion import curve_map
    ad=rig.animation_data
    if not ad or not ad.action:return ''
    digest=hashlib.sha256(ad.action.name.encode())
    for key,fc in sorted(curve_map(ad.action,ad.action_slot).items()):
        digest.update(str(key).encode());coords=np.empty(len(fc.keyframe_points)*2,np.float32)
        fc.keyframe_points.foreach_get('co',coords);digest.update(coords.tobytes())
    return digest.hexdigest()


def validate(scene,rig):
    if not rig or rig.type!='ARMATURE':raise ValueError('请选择艾莲骨架')
    missing=[n for n in ('Head','Neck','Arm_L','Arm_R','Eye_L','Eye_R') if n not in rig.pose.bones]
    if missing:raise ValueError('骨架缺少所需骨骼：'+', '.join(missing))
    if rig.animation_data and any(not t.mute for t in rig.animation_data.nla_tracks):
        raise ValueError('此骨架有启用的 NLA 轨道，请先停用叠加轨道后生成，避免动作重复叠加')
    if any(r.enabled and r.end_frame<r.start_frame for r in scene.mocap_v2.away_ranges):
        raise ValueError('不看镜头区间结束帧早于开始帧')
    from sam_ellen_phone_ranges import ranges
    ranges(scene.sam_ellen)


def assist(scene,rig):
    from natural_motion import apply
    visibility=None;path=Path(rig.get('mocap_v2_diagnostics',''))
    if path.is_file():
        with np.load(path) as data:
            if 'hand_visibility' in data:visibility=data['hand_visibility'].copy()
    return apply(scene,rig,scene.mocap_v2,visibility)


def import_result(scene,rig,manifest_path,video,options=None):
    """Commit only target bone curves and cameras; facial datablocks stay in place."""
    from natural_motion import curve_map
    from presentation_camera import ensure_presentation_camera
    validate(scene,rig)
    manifest=json.loads(Path(manifest_path).read_text(encoding='utf8'))
    with bpy.data.libraries.load(manifest['blend'],link=False) as (src,dst):
        if manifest['action'] not in src.actions:raise ValueError('动作包缺少骨架动作')
        dst.actions=[manifest['action']];dst.objects=list(manifest['cameras'])
    incoming=dst.actions[0];objects=dst.objects
    if any(o is None for o in objects):raise ValueError('动作包摄像机不完整')
    original=rig.animation_data.action if rig.animation_data else None
    slot=rig.animation_data.action_slot if rig.animation_data else None
    original_camera=scene.camera;original_frame=scene.frame_current
    timing=(scene.frame_start,scene.frame_end,scene.render.fps,scene.render.fps_base,
            scene.render.resolution_x,scene.render.resolution_y)
    constraints=[(c,c.mute) for b in rig.pose.bones for c in b.constraints]
    modes=[(b,b.rotation_mode) for b in rig.pose.bones]
    props={k:rig.get(k) for k in ('mocap_v2_base_action','mocap_v2_assisted_action','mocap_v2_diagnostics','mocap_v2_frame_count','mocap_v2_hand_tracks','mocap_v2_hamer_fingers')}
    action=original.copy() if original else bpy.data.actions.new('V2 动捕')
    action.name='V2 动捕 · '+Path(video).stem[:16]
    try:
        rig.animation_data_create().action=action
        if slot:
            rig.animation_data.action_slot=next(s for s in action.slots if s.identifier==slot.identifier)
        for (path,index),source in curve_map(incoming,None).items():
            if not (path.startswith('pose.bones[') or path=='location'):continue
            target=action.fcurve_ensure_for_datablock(rig,path,index=index)
            for modifier in list(target.modifiers):target.modifiers.remove(modifier)
            target.keyframe_points.clear();target.keyframe_points.add(len(source.keyframe_points))
            coords=np.empty(len(source.keyframe_points)*2,np.float32)
            source.keyframe_points.foreach_get('co',coords);target.keyframe_points.foreach_set('co',coords)
            for p in target.keyframe_points:p.interpolation='LINEAR'
            target.extrapolation=source.extrapolation;target.update()
        for b in rig.pose.bones:
            b.rotation_mode=manifest['rotation_modes'].get(b.name,b.rotation_mode)
            states={c['name']:c['mute'] for c in manifest['constraints'].get(b.name,[])}
            for c in b.constraints:
                if c.name in states:c.mute=states[c.name]
        for obj in objects:scene.collection.objects.link(obj)
        # Requested objects are returned in requested order, even on name collisions.
        scene.camera=objects[manifest['cameras'].index(manifest['camera'])]
        scene.render.fps=manifest['fps'];scene.render.fps_base=manifest['fps_base']
        scene.render.resolution_x,scene.render.resolution_y=manifest['resolution']
        scene.frame_start=1;scene.frame_end=manifest['frames']
        rig['mocap_v2_frame_count']=manifest['frames'];rig['mocap_v2_diagnostics']=manifest['diagnostics']
        fingers=Path(manifest_path).parent/'hamer_fingers.npz'
        if fingers.is_file():
            rig['mocap_v2_hamer_fingers']=str(fingers.resolve())
        elif 'mocap_v2_hamer_fingers' in rig:del rig['mocap_v2_hamer_fingers']
        if manifest.get('hand_tracks'):
            rig['mocap_v2_hand_tracks']=manifest['hand_tracks']
        elif 'mocap_v2_hand_tracks' in rig:del rig['mocap_v2_hand_tracks']
        for k in ('mocap_v2_base_action','mocap_v2_assisted_action'):
            if k in rig:del rig[k]
        assist(scene,rig)
        ensure_presentation_camera(scene)
        scene['mocap_v2_validation']=manifest['validation']
        scene['mocap_v2_stationary_denoise_version']=1
        if original:original.use_fake_user=True
        scene.frame_set(1)
        from shoulder_frills import stabilize
        stabilize(rig)
    except Exception:
        rig.animation_data.action=original
        if original and slot:rig.animation_data.action_slot=slot
        for c,mute in constraints:c.mute=mute
        for b,mode in modes:b.rotation_mode=mode
        for k,value in props.items():
            if value is None:
                if k in rig:del rig[k]
            else:rig[k]=value
        scene.camera=original_camera
        (scene.frame_start,scene.frame_end,scene.render.fps,scene.render.fps_base,
         scene.render.resolution_x,scene.render.resolution_y)=timing
        for obj in objects:bpy.data.objects.remove(obj,do_unlink=True)
        scene.frame_set(original_frame)
        raise
    finally:
        if incoming.users==0:bpy.data.actions.remove(incoming)
    # Auxiliary failures are visible, and never discard a successful body take.
    notices=[];scene.sam_ellen.rig=rig
    if options:
        for key,value in options.items():setattr(scene.sam_ellen_editing,key,value)
    try:
        from sam_ellen_phone_ranges import apply
        apply(scene)
    except Exception as exc:notices.append('持手机：'+str(exc))
    try:
        from sam_ellen_video_audio import attach
        strip=attach(scene,rig,{'video':video,'frames':[{'time':0},{'time':(manifest['frames']-1)/(manifest['fps']/manifest['fps_base'])}]},rig.animation_data.action,original)
        if scene.sam_ellen_editing.video_audio:notices.append(scene.sam_ellen_editing.status)
        if scene.sam_ellen_editing.video_audio_lipsync and strip:
            from sam_ellen_audio_lipsync import bake_capture
            bake_capture(scene,rig,bpy.path.abspath(strip.sound.filepath))
            notices.append('声音口型已生成，原表情区间保留')
    except Exception as exc:notices.append('声音/口型：'+str(exc))
    # Retain sound ownership when reapplying assistance later.
    base=rig.get('mocap_v2_base_action')
    if base:base['sam_audio_tracks']=rig.animation_data.action.get('sam_audio_tracks','[]')
    scene.mocap_v2.status='完成：'+str(manifest['frames'])+' 帧。'+'；'.join(notices)
    scene.mocap_v2.result_path=str(manifest_path)
    return rig.animation_data.action


def stop_job():
    global _job
    if _job:
        (_job['directory']/'cancel').touch()
        _job=None


@persistent
def before_load(_):stop_job()


def poll():
    global _job
    if not _job:return None
    job=_job;scene=job['scene'];s=scene.mocap_v2
    try:
        status_path=job['directory']/'status.json'
        status=json.loads(status_path.read_text(encoding='utf8')) if status_path.exists() else {}
        elapsed=int(time.time()-job['started'])
        s.status=f"{status.get('stage','启动后台任务')}（{elapsed//60}分{elapsed%60}秒）"
        if status.get('detail'):s.status+=' '+status['detail']
        if status.get('state') in ('error','cancelled'):
            s.status=status['stage'];_job=None;return None
        if status.get('state')=='complete':
            s.result_path=status['result'];_job=None
            if bpy.context.scene!=scene or fingerprint(job['rig'])!=job['fingerprint']:
                s.status='识别已完成；场景或动作已变化，请点击“导入已完成结果”'
                return None
            import_result(scene,job['rig'],s.result_path,job['video'],job['audio'])
            return None
        if job['process'].poll() is not None:
            s.status='后台进程已退出，原动作未改；请打开任务日志查看原因';_job=None;return None
        for screen in bpy.data.screens:
            for area in screen.areas:
                if area.type=='VIEW_3D':area.tag_redraw()
        return .5
    except Exception as exc:
        s.status='未导入：'+str(exc);_job=None
        import traceback;traceback.print_exc()
        return None


class V2Range(bpy.types.PropertyGroup):
    enabled:BoolProperty(name='启用',default=True)
    start_frame:IntProperty(name='开始帧',default=1,min=1)
    end_frame:IntProperty(name='结束帧',default=30,min=1)


class V2Settings(bpy.types.PropertyGroup):
    video:StringProperty(name='动捕视频',subtype='FILE_PATH')
    rig:PointerProperty(name='艾莲骨架',type=bpy.types.Object,poll=lambda s,o:o.type=='ARMATURE',update=rig_changed)
    camera_mode:EnumProperty(name='拍摄机位',items=[('STATIC','固定机位','摄像机固定，角色负责位移'),('DYNAMIC','移动机位','复现连续单镜头的摄像机运动')],default='STATIC')
    framing:EnumProperty(name='构图',items=[('face','按视频脸部比例','脸部大小与位置优先'),('portrait','近景','更紧凑的自拍构图')],default='face')
    gaze_reference:IntProperty(name='看镜头标定帧',default=1,min=1,description='填写视频中明确看拍摄镜头的帧，30 帧/秒')
    natural_body:BoolProperty(name='身体与未捕获手臂偶尔轻微左右摆',default=True)
    body_amount:FloatProperty(name='身体幅度（度）',default=.6,min=0,max=2)
    natural_eyes:BoolProperty(name='眼睛轻微自然晃动',default=True)
    eye_amount:FloatProperty(name='眼神幅度（度）',default=.25,min=0,max=1)
    natural_seed:IntProperty(name='自然变化种子',default=17,min=0)
    away_ranges:CollectionProperty(type=V2Range)
    away_fade:IntProperty(name='视线过渡帧数',default=8,min=1,max=60)
    status:StringProperty(name='状态',default='选择视频后生成 V2 动捕；原表情插件可继续使用')
    result_path:StringProperty(subtype='FILE_PATH')
    job_path:StringProperty(subtype='DIR_PATH')


class V2Generate(bpy.types.Operator):
    bl_idname='mocap_v2.generate';bl_label='生成 V2 动捕'
    def execute(self,context):
        global _job
        scene=context.scene;s=scene.mocap_v2;rig=s.rig
        try:
            if _job:raise ValueError('已有 V2 任务正在进行')
            import sam_ellen_bridge
            if sam_ellen_bridge._work:raise ValueError('旧动捕正在生成，请先完成或取消旧任务')
            validate(scene,rig)
            video=Path(bpy.path.abspath(s.video))
            if not video.is_file():raise ValueError('请选择存在的视频文件')
            blender=Path(bpy.app.binary_path)
            python=ROOT.parent/'sam-3d-body-main/.venv/Scripts/python.exe'
            blender_python=blender.parent/'5.2/python/bin/python.exe'
            if not python.is_file():raise ValueError('未找到 V2 模型运行环境：'+str(python))
            directory=Path('C:/mocap_v2_outputs/ui')/uuid.uuid4().hex;directory.mkdir(parents=True)
            snapshot=directory/'snapshot.blend'
            bpy.ops.wm.save_as_mainfile(filepath=str(snapshot),copy=True)
            request={'video':str(video),'snapshot':str(snapshot),'rig':rig.name,
                'camera_mode':s.camera_mode,'framing':s.framing,'gaze_reference':s.gaze_reference,
                'python':str(python),'blender':str(blender),'blender_python':str(blender_python)}
            path=directory/'request.json';path.write_text(json.dumps(request,ensure_ascii=False),encoding='utf8')
            with (directory/'launcher.log').open('w',encoding='utf8') as log:
                process=subprocess.Popen([str(python),str(V2/'jobs/blender_job.py'),str(path)],
                    stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            _job={'scene':scene,'rig':rig,'fingerprint':fingerprint(rig),'directory':directory,
                'process':process,'started':time.time(),'video':str(video),
                'audio':{k:getattr(scene.sam_ellen_editing,k) for k in ('video_audio','video_audio_lipsync','movie_audio','lipsync_gain','lipsync_lead')}}
            s.job_path=str(directory);s.result_path='';s.status='已启动 V2 后台识别'
            bpy.app.timers.register(poll,first_interval=.5)
            return {'FINISHED'}
        except Exception as exc:self.report({'ERROR'},str(exc));s.status=str(exc);return {'CANCELLED'}


class V2Cancel(bpy.types.Operator):
    bl_idname='mocap_v2.cancel';bl_label='取消生成'
    def execute(self,context):
        if _job:(_job['directory']/'cancel').touch();context.scene.mocap_v2.status='正在停止后台任务；原动作保留'
        return {'FINISHED'}


class V2Assist(bpy.types.Operator):
    bl_idname='mocap_v2.assist';bl_label='更新自然摇摆与视线区间';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        s=context.scene.mocap_v2
        try:
            validate(context.scene,s.rig);assist(context.scene,s.rig)
            s.status='已更新自然辅助；视频视线与原表情保留'
            return {'FINISHED'}
        except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}


class V2Import(bpy.types.Operator):
    bl_idname='mocap_v2.import_result';bl_label='导入已完成结果';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        s=context.scene.mocap_v2
        try:
            request=json.loads((Path(s.result_path).parent/'request.json').read_text(encoding='utf8'))
            import_result(context.scene,s.rig,s.result_path,request['video']);return {'FINISHED'}
        except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}


class V2RefitHands(bpy.types.Operator):
    bl_idname='mocap_v2.refit_hands';bl_label='按当前摄像机重新适配手部'
    bl_description='保留摄像机微调、骨长与源动作肘部弯曲，重新拟合手的位置和手势；可撤销'
    bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        scene=context.scene;rig=scene.mocap_v2.rig
        old=None;props={};frame=scene.frame_current
        try:
            if _job:raise ValueError('请先完成或取消当前生成任务')
            validate(scene,rig)
            path=Path(rig.get('mocap_v2_hand_tracks',''))
            if not path.is_file():raise ValueError('此工程没有新版手部观测，请先生成新版动捕')
            from hand_refinement import load_tracks,apply
            tracks=load_tracks(path)
            old=rig.animation_data.action
            props={k:rig.get(k) for k in ('mocap_v2_base_action','mocap_v2_assisted_action')}
            base=rig.get('mocap_v2_base_action',old).copy()
            rig.animation_data.action=base
            for key in props:
                if key in rig:del rig[key]
            apply(scene,rig,tracks)
            fingers=Path(bpy.path.abspath(rig.get('mocap_v2_hamer_fingers',str(path.parent/'hamer_fingers.npz'))))
            if fingers.is_file():
                from hamer_pose import apply_hamer_pose
                with np.load(fingers) as data:
                    apply_hamer_pose(scene,rig,data['points'],data['valid'],tracks)
            from part_projection import align_parts
            align_parts(scene,rig,tracks)
            from temporal_denoise import stabilize_final_hands
            stabilize_final_hands(scene,rig,tracks)
            assist(scene,rig)
            old.use_fake_user=True
            from sam_ellen_phone_ranges import apply as apply_phone
            apply_phone(scene)
            scene.mocap_v2.status='已按当前摄像机重新适配手部，保留源动作肘部弯曲'
            return {'FINISHED'}
        except Exception as exc:
            if old is not None:
                rig.animation_data.action=old
                for key,value in props.items():
                    if value is None:
                        if key in rig:del rig[key]
                    else:rig[key]=value
            self.report({'ERROR'},str(exc));return {'CANCELLED'}
        finally:scene.frame_set(frame)


def replace_current_lipsync(scene,filepath):
    from sam_ellen_audio_lipsync import bake
    from sam_ellen_face import face_object
    if _job:raise ValueError('请先完成或取消 V2 动捕任务')
    rig=scene.mocap_v2.rig
    if not rig or not rig.animation_data or not rig.animation_data.action or not rig.get('mocap_v2_frame_count'):
        raise ValueError('请先选择已有 V2 动作的艾莲骨架')
    mesh=face_object(scene,rig)
    if mesh is None:raise ValueError('找不到当前 V2 角色的面部网格')
    selected=scene.ellen_expression_object
    scene.ellen_expression_object=mesh
    settings=scene.sam_ellen_editing
    try:
        result=bake(scene,filepath,1,settings.lipsync_gain,settings.lipsync_lead,
                    settings.lipsync_import_audio,replace_end=int(rig['mocap_v2_frame_count']),audio_owner=rig)
    except Exception:
        scene.ellen_expression_object=selected
        raise
    scene.mocap_v2.status=f"已覆盖当前动捕第 1–{int(rig['mocap_v2_frame_count'])} 帧口型；动画时长不变"
    settings.status=scene.mocap_v2.status
    return result


class V2AudioLipSync(bpy.types.Operator,ImportHelper):
    bl_idname='mocap_v2.audio_lipsync';bl_label='选择配音并覆盖当前口型'
    bl_description='从当前 V2 动捕第一帧覆盖口型和对应配音；不追加动画帧，保留身体动作和表情区间'
    bl_options={'REGISTER','UNDO'}
    filter_glob:StringProperty(default='*.wav;*.mp3;*.flac;*.ogg;*.m4a;*.aac;*.mp4',options={'HIDDEN'})
    def execute(self,context):
        try:
            replace_current_lipsync(context.scene,self.filepath)
            self.report({'INFO'},context.scene.mocap_v2.status)
            return {'FINISHED'}
        except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}


class V2EditRange(bpy.types.Operator):
    bl_idname='mocap_v2.range';bl_label='编辑不看镜头时间段';bl_options={'REGISTER','UNDO'}
    index:IntProperty(default=-1)
    def execute(self,context):
        rows=context.scene.mocap_v2.away_ranges
        if self.index<0:
            item=rows.add();item.start_frame=context.scene.frame_current;item.end_frame=item.start_frame+30
        elif self.index<len(rows):rows.remove(self.index)
        return {'FINISHED'}


class V2Log(bpy.types.Operator):
    bl_idname='mocap_v2.log';bl_label='打开任务日志'
    def execute(self,context):
        path=Path(context.scene.mocap_v2.job_path)/'worker.log'
        if path.is_file():
            text=bpy.data.texts.get('V2 任务日志') or bpy.data.texts.new('V2 任务日志')
            text.clear();text.write(path.read_text(encoding='utf8',errors='replace'))
            context.area.type='TEXT_EDITOR';context.area.spaces.active.text=text
        else:self.report({'INFO'},'尚未生成日志')
        return {'FINISHED'}


class V2Panel(bpy.types.Panel):
    bl_label='动捕 V2';bl_idname='MOCAPV2_PT_main';bl_space_type='VIEW_3D';bl_region_type='UI';bl_category='动捕'
    def draw(self,context):
        scene=context.scene;s=scene.mocap_v2;layout=self.layout
        col=layout.column();col.enabled=not bool(_job)
        for key in ('rig','video','camera_mode','framing','gaze_reference'):col.prop(s,key)
        from sam_ellen_editing import draw_video_audio_options
        draw_video_audio_options(col,scene)
        col.operator('mocap_v2.generate',icon='TRACKING')
        if _job:layout.operator('mocap_v2.cancel',icon='CANCEL')
        import textwrap
        for line in textwrap.wrap(s.status,25):layout.label(text=line)
        if s.job_path:layout.operator('mocap_v2.log',icon='TEXT')
        col=layout.column();col.enabled=not bool(_job)
        if s.result_path:col.operator('mocap_v2.import_result',icon='IMPORT')
        if s.rig and s.rig.get('mocap_v2_hand_tracks'):
            col.operator('mocap_v2.refit_hands',icon='CAMERA_DATA')
        box=col.box();box.label(text='自然辅助 · 保留有效捕获动作')
        box.prop(s,'natural_body');box.prop(s,'body_amount')
        box.prop(s,'natural_eyes');box.prop(s,'eye_amount')
        box=col.box();box.label(text='不看镜头时间段（帧）')
        box.label(text='区间外保留视频视线意图')
        for i,item in enumerate(s.away_ranges):
            row=box.row(align=True);row.prop(item,'enabled',text='');row.prop(item,'start_frame',text='起');row.prop(item,'end_frame',text='止')
            row.operator('mocap_v2.range',text='',icon='X').index=i
        box.operator('mocap_v2.range',text='添加时间段',icon='ADD');box.prop(s,'away_fade')
        box.label(text='区间内视线随头部自然朝前')
        col.operator('mocap_v2.assist',icon='FILE_REFRESH')
        box=col.box();box.label(text='声音生成口型（兼容原表情）')
        box.prop(scene.sam_ellen_editing,'lipsync_gain');box.prop(scene.sam_ellen_editing,'lipsync_lead')
        box.prop(scene.sam_ellen_editing,'lipsync_import_audio')
        box.label(text='覆盖当前动捕口型，不追加动画帧')
        box.operator('mocap_v2.audio_lipsync',icon='SOUND')
        box.operator('sam_ellen.restore_audio_lipsync',icon='LOOP_BACK')
        from sam_ellen_phone_ranges import draw
        draw(col,scene.sam_ellen)
        import ellen_physics
        if scene.sam_ellen.rig:ellen_physics.draw(col,scene)
        else:col.label(text='选择骨架后显示人物物理选项')
        col.label(text='表情请继续使用原表情插件',icon='INFO')


CLASSES=(V2Range,V2Settings,V2Generate,V2Cancel,V2Assist,V2Import,V2RefitHands,V2AudioLipSync,V2EditRange,V2Log,V2Panel)


def register():
    import addon_utils
    if not hasattr(bpy.types.Scene,'sam_ellen'):addon_utils.enable('sam_ellen_bridge',default_set=True,persistent=True)
    if not hasattr(bpy.types.Scene,'sam_ellen'):raise RuntimeError('请先启用旧版艾莲插件，V2 复用其音频/物理接口')
    for cls in CLASSES:bpy.utils.register_class(cls)
    bpy.types.Scene.mocap_v2=PointerProperty(type=V2Settings)
    bpy.app.handlers.load_pre.append(before_load)


def unregister():
    stop_job()
    if bpy.app.timers.is_registered(poll):bpy.app.timers.unregister(poll)
    if before_load in bpy.app.handlers.load_pre:bpy.app.handlers.load_pre.remove(before_load)
    del bpy.types.Scene.mocap_v2
    for cls in reversed(CLASSES):bpy.utils.unregister_class(cls)

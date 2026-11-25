"""Named JSON configurations stored beside this installed add-on's code."""
import json,math,os,re,shutil,tempfile,zlib
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
from . import runtime,layers,baking,compat,adjustments,mapping

FORMAT='mmr_motion_bridge_config'
SCHEMA=1
SETTINGS=('mode','auto_fk','frame_start','frame_end','motion_scale','heading','root_offset',
          'head_angles','neck_angles','chest_angles','head_axis','neck_axis','height_cm',
          'height_root_ratio','hand_pair_strength','finger_strength','leg_position','ground',
          'floor_height','foot_lock','contact_height_cm','contact_speed_cm','bake_step','action_name')
BONE_FIELDS=('target','source','offset_space','rotation_offset','location_offset')
NONE='__NONE__'
_items=[]
_strings={}


def directory():return Path(__file__).resolve().parent/'configs'


def filename(name):
    name=name.strip()
    if name.lower().endswith('.json'):name=name[:-5]
    if not name or len(name)>80 or name.endswith(('.', ' ')) or re.search(r'[<>:"/\\|?*\x00-\x1f]',name):
        raise ValueError('配置名称不能为空，也不能包含路径、特殊文件名字符或末尾的点。')
    if name in {'.','..'} or re.fullmatch(r'(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?',name):
        raise ValueError('这个名称不能用作 Windows 文件名，请换一个配置名称。')
    return name+'.json'


def refresh():
    global _items
    def keep(s):return _strings.setdefault(s,s)
    result=[(NONE,'选择已保存的配置','选择配置后点击“读取配置”',0)]
    if directory().exists():
        used={0}
        for p in sorted(directory().glob('*.json'),key=lambda p:p.name.casefold()):
            number=zlib.crc32(p.name.encode('utf8'))&0x7fffffff
            while number in used:number=(number+1)&0x7fffffff
            used.add(number)
            result.append((keep(p.name),keep(p.stem),keep(str(p)),number))
    _items=result
    return _items


def enum_items(self,context):return _items or refresh()


def selection_changed(self,context):
    if self.config_selected!=NONE:self.config_name=Path(self.config_selected).stem


def plain(value):
    return value if isinstance(value,(str,bool,int,float)) else list(value)


def export(cfg):
    from . import bl_info
    return {'format':FORMAT,'schema_version':SCHEMA,'addon_version':list(bl_info['version']),
            'saved_at_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'units':{'rotation':'radians','translation':'Blender scene units','height_cm':'centimeters'},
            'settings':{n:plain(getattr(cfg,n)) for n in SETTINGS},
            'bone_map':[{n:plain(getattr(e,n)) for n in BONE_FIELDS} for e in cfg.bone_map]}


def save(cfg,name):
    path=directory()/filename(name);payload=export(cfg)
    text=json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    directory().mkdir(parents=True,exist_ok=True)
    temp=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=directory(),prefix='._config_',suffix='.tmp',delete=False) as f:
            temp=Path(f.name);f.write(text);f.flush();os.fsync(f.fileno())
        if path.exists():shutil.copy2(path,path.with_suffix('.json.bak'))
        os.replace(temp,path);temp=None
    finally:
        if temp and temp.exists():temp.unlink()
    refresh()
    return path


def checked(value,prop,label):
    if getattr(prop,'is_array',False):
        if not isinstance(value,list) or len(value)!=prop.array_length:
            raise ValueError(label+' 必须是 '+str(prop.array_length)+' 个数值。')
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<prop.hard_min or v>prop.hard_max for v in value):
            raise ValueError(label+' 含有无效或越界数值。')
        return [float(v) for v in value]
    if prop.type=='BOOLEAN':valid=type(value) is bool
    elif prop.type in {'INT','FLOAT'}:
        valid=type(value) in ((int,) if prop.type=='INT' else (int,float)) and math.isfinite(value) and prop.hard_min<=value<=prop.hard_max
    elif prop.type=='ENUM':valid=isinstance(value,str) and value in prop.enum_items.keys()
    elif prop.type=='STRING':valid=isinstance(value,str) and len(value)<=1024
    else:valid=False
    if not valid:raise ValueError(label+' 的类型或取值无效。')
    return value


def validate(cfg,data):
    if not isinstance(data,dict) or data.get('format')!=FORMAT or type(data.get('schema_version')) is not int or data['schema_version']!=SCHEMA:
        raise ValueError('不是受支持的 FlowHMR 2 MMR 配置文件。')
    if not isinstance(data.get('settings'),dict) or not isinstance(data.get('bone_map',[]),list):
        raise ValueError('配置缺少 settings 参数对象，或 bone_map 格式不正确。')
    settings={n:plain(getattr(cfg,n)) for n in SETTINGS}
    for n,value in data['settings'].items():
        if n in SETTINGS:settings[n]=checked(value,cfg.bl_rna.properties[n],n)
    if settings['frame_end']<settings['frame_start']:raise ValueError('配置的结束帧小于起始帧。')
    from . import MMRB_BoneMap
    # A partial JSON may override a few bones; a saved full configuration
    # restores every entry. Missing optional fields keep the current values.
    entries={e.target:{n:plain(getattr(e,n)) for n in BONE_FIELDS} for e in cfg.bone_map}
    seen=set()
    for row in data.get('bone_map',[]):
        if not isinstance(row,dict) or not isinstance(row.get('target'),str) or row['target'] not in mapping.DEFAULT:raise ValueError('配置包含未知的 MMR 控制骨。')
        name=row['target']
        if name in seen:raise ValueError('配置重复定义骨骼：'+name)
        seen.add(name)
        base=entries.setdefault(name,{'target':name,'source':mapping.DEFAULT[name],'offset_space':'LOCAL','rotation_offset':[0.,0.,0.],'location_offset':[0.,0.,0.]})
        for n,value in row.items():
            if n in BONE_FIELDS:base[n]=checked(value,MMRB_BoneMap.bl_rna.properties[n],name+'.'+n)
    return settings,list(entries.values())


def read(cfg,name):
    path=directory()/filename(name)
    if not path.is_file():raise ValueError('找不到配置文件：'+path.name)
    def bad_constant(value):raise ValueError('配置不能包含 '+value+'。')
    try:data=json.loads(path.read_text(encoding='utf-8-sig'),parse_constant=bad_constant)
    except (UnicodeError,json.JSONDecodeError) as e:raise ValueError('JSON 文件格式不正确：'+str(e)) from e
    return path,validate(cfg,data)


def assign(cfg,settings,entries):
    previous=runtime.BUSY;runtime.BUSY=True
    try:
        for n,value in settings.items():setattr(cfg,n,value)
        expanded={e.target:e.expanded for e in cfg.bone_map}
        cfg.bone_map.clear()
        for row in entries:
            e=cfg.bone_map.add()
            for n,value in row.items():setattr(e,n,value)
            e.expanded=expanded.get(e.target,False)
    finally:runtime.BUSY=previous


def apply(scene,values):
    cfg=scene.mmrb;settings,entries=values;old=export(cfg);status=cfg.status
    active=status!='NONE';frame=scene.frame_current
    old_action=cfg.target.animation_data.action if active and cfg.target.animation_data else None
    old_slot=compat.slot_id(cfg.target) if active else ''
    if active:
        candidate=SimpleNamespace(**settings,source=cfg.source,target=cfg.target,deform=cfg.deform,
                                  bone_map=[SimpleNamespace(**row) for row in entries])
        runtime.validate(scene,candidate,allowed_driver_paths=layers.read_state(cfg).get('switch_drivers',()))
    try:
        if active:runtime.remove(scene)
        assign(cfg,settings,entries)
        if active:
            runtime.start(scene)
            if status=='PAUSED':cfg.status='PAUSED'
        scene.frame_set(frame)
        return active
    except Exception as error:
        # Reconstruct the previous preview from its original Action, never
        # bake implicitly or overwrite an existing Action during loading.
        try:
            if cfg.helpers:runtime.remove(scene)
            assign(cfg,old['settings'],old['bone_map'])
            if active:
                runtime.start(scene)
                if status=='BAKED':
                    cfg.status='BAKED';baking.set_retarget_constraints(cfg,True)
                    layers.release_fk_drivers(cfg);layers.source_dependency(cfg,False)
                    compat.assign_action(cfg.target,old_action,old_slot)
                    cfg.target.animation_data.use_nla=False
                    cfg.target.animation_data.action_blend_type='REPLACE';cfg.target.animation_data.action_influence=1.
                    for t in cfg.target.animation_data.nla_tracks:t.mute=True
                    adjustments.sync(cfg)
                elif status=='PAUSED':cfg.status='PAUSED'
            scene.frame_set(frame)
        except Exception as rollback:
            raise RuntimeError('读取失败，恢复预览也失败：'+str(rollback)+'；原动作仍保留。') from error
        raise RuntimeError('读取失败，已恢复原设置：'+str(error)) from error

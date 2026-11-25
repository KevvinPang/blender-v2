"""Isolated V2 inference worker. Blender UI only polls atomic status files."""
import json, os, subprocess, sys, time, traceback
from pathlib import Path

V2=Path(__file__).resolve().parents[1]


def atomic(path,data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
    for attempt in range(12):
        try:
            os.replace(tmp,path)
            return
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(.05)


def main(request_path):
    request_path=Path(request_path);root=request_path.parent
    request=json.loads(request_path.read_text(encoding='utf8'))
    status=root/'status.json';log=root/'worker.log';started=time.time()
    def report(stage,state='running',**extra):
        atomic(status,dict(stage=stage,state=state,updated=time.time(),started=started,**extra))
    def run(stage,command):
        if (root/'cancel').exists():raise InterruptedError('用户取消；原动作保持不变')
        report(stage)
        with log.open('a',encoding='utf8') as out:
            out.write('\n'+stage+'\n');out.flush()
            process=subprocess.Popen(list(map(str,command)),stdout=out,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1'),
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            try:
                while process.poll() is None:
                    if (root/'cancel').exists():
                        subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,
                            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                        raise InterruptedError('用户取消；原动作保持不变')
                    # Include the latest real progress line, not a fabricated percentage.
                    with log.open('rb') as tail:
                        tail.seek(max(0,log.stat().st_size-2048))
                        lines=tail.read().decode('utf8',errors='replace').splitlines()
                    detail=next((line for line in reversed(lines) if line.startswith(('[v2]','V2_PROGRESS','OBS '))), '')
                    report(stage,child_pid=process.pid,detail=detail[:180])
                    time.sleep(.5)
                if process.returncode:raise RuntimeError(f'{stage}失败（退出码 {process.returncode}）；详见 {log}')
            finally:
                if process.poll() is None:process.terminate()
        if (root/'cancel').exists():raise InterruptedError('用户取消；原动作保持不变')
    try:
        if (root/'cancel').exists():raise InterruptedError('用户取消；原动作保持不变')
        for field in ('video','snapshot','python','blender','blender_python'):
            if not Path(request[field]).is_file():raise FileNotFoundError(f'{field} 文件不存在：{request[field]}')
        if request.get('cache'):
            # Explicit developer validation only; the panel always creates a fresh job.
            job=Path(request['cache'])
        else:
            command=[request['python'],V2/'run_dynamic_camera.py',request['video'],'--output',root/'inference']
            if request['camera_mode']=='STATIC':command+=['--static-camera']
            if request.get('max_frames'):command+=['--max-frames',str(request['max_frames'])]
            run('1/5 身体与摄像机识别',command)
            job=root/'inference'/Path(request['video']).stem
            run('2/5 身体位置校验',[request['python'],V2/'validation/extract_observations.py',job,'--stride','3'])
            run('3/5 逐帧手部识别',[request['blender_python'],V2/'validation/extract_hand_observations.py',job,'--mode','image'])
            run('4/5 头部与视线识别',[request['blender_python'],V2/'validation/extract_face_observations.py',job])
        camera=list(job.glob('*_static_camera.npz' if request['camera_mode']=='STATIC' else '*_vggt_camera.npz'))
        motion=list(job.glob('*_seed0_origin.npz'))
        if len(camera)!=1 or len(motion)!=1:raise ValueError('缓存的摄像机模式或动作文件不唯一')
        if request['camera_mode']=='DYNAMIC':
            processed=job/'camera_joint.npz'
            run('4/5 背景运动与人物定位协调',[request['python'],V2/'camera/prepare_joint_camera.py',
                camera[0],job/(Path(request['video']).stem+'_30fps.mp4'),
                job/'face_observations.npz',job/'sam_observations.npz',processed,job/'background_motion.npz'])
            camera=[processed]
        hand_python=Path(request.get('hamer_python', Path.home()/'Downloads/WiLoR-main/.venv/Scripts/python.exe'))
        fingers=root/'hamer_fingers.npz'
        run('4/5 HaMeR 完整指形',[hand_python,V2/'jobs/hamer_fingers.py',
            job/(Path(request['video']).stem+'_30fps.mp4'),fingers])
        parts=root/'hand_parts.npz'
        run('4/5 手部出入镜逐点追踪',[hand_python,V2/'jobs/hand_parts.py',
            job/(Path(request['video']).stem+'_30fps.mp4'),job/'hand_observations.npz',parts])
        command=[request['blender'],'--background','--factory-startup',request['snapshot'],
            '--python-exit-code','1','--python',V2/'jobs/export_take.py','--',motion[0],root/'result.blend',
            '--camera-track',camera[0],'--observations',job/'sam_observations.npz',
            '--hands',job/'hand_observations.npz','--face-observations',job/'face_observations.npz','--hamer-fingers',fingers,
            '--hand-parts',parts,
            '--head-mode','face','--gaze-mode','video','--gaze-reference-frame',str(request['gaze_reference']),
            '--rig-name',request['rig'],'--framing',request['framing'],'--preserve-expressions']
        run('5/5 艾莲重定向与连续性处理',command)
        if not (root/'result.json').is_file():raise RuntimeError('未产生完整动作包')
        report('识别完成，等待导入',state='complete',result=str(root/'result.json'))
    except InterruptedError as exc:report(str(exc),state='cancelled')
    except Exception as exc:
        with log.open('a',encoding='utf8') as out:out.write(traceback.format_exc())
        report(str(exc),state='error')
        return 1
    return 0


if __name__=='__main__':raise SystemExit(main(sys.argv[1]))

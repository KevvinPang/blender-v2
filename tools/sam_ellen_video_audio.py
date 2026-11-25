"""Video sound follows successful SAM takes, independently of manual dubbing."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import bpy


def executable(name):
    candidates = [shutil.which(name), f'F:/RVC-WebUI/{name}.exe',
                  f'F:/Applio-3.6.6/{name}.exe']
    return next((str(p) for p in candidates if p and Path(p).is_file()), None)


def run(name, args):
    program = executable(name)
    if not program:
        raise ValueError(f'未找到 {name}，请安装 FFmpeg')
    result = subprocess.run([program, *map(str, args)], capture_output=True,
                            encoding='utf8', errors='replace',
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode:
        raise ValueError(result.stderr[-1200:])
    return result.stdout


def prepare_movie(scene):
    # A sound-only sequencer leaves the camera image intact.
    scene.render.image_settings.media_type = 'VIDEO'
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.audio_codec = 'AAC'
    scene.render.ffmpeg.audio_bitrate = 192
    scene.render.ffmpeg.audio_mixrate = 48000
    scene.render.ffmpeg.audio_channels = 'STEREO'
    scene.render.use_sequencer = True
    scene.use_audio = True


def track_names(action):
    return json.loads(action.get('sam_audio_tracks', '[]')) if action else []


def activate(scene, rig, action):
    from sam_ellen_editing import sound_strips
    names = set(track_names(action))
    for strip in sound_strips(scene):
        if strip.get('sam_audio_owner') == rig.name:
            strip.mute = strip.name not in names


def extract(video, begin, duration):
    path = Path(bpy.path.abspath(video)).resolve()
    if not path.is_file():
        raise ValueError('原视频已移动，无法导入声音')
    info = json.loads(run('ffprobe', ['-v', 'error', '-show_streams', '-of', 'json', path]))
    audio = next((s for s in info['streams'] if s['codec_type'] == 'audio'), None)
    if not audio:
        return None
    video_stream = next((s for s in info['streams'] if s['codec_type'] == 'video'), {})
    offset = float(audio.get('start_time', 0))-float(video_stream.get('start_time', 0))
    signature = f'{path}|{path.stat().st_mtime_ns}|{path.stat().st_size}|{begin:.9f}|{duration:.9f}'
    key = hashlib.sha256(signature.encode('utf8')).hexdigest()[:24]
    cache = Path(bpy.utils.user_resource('DATAFILES', path='sam_video_audio', create=True))
    target = cache / (key+'.wav')
    if not target.is_file():
        temporary = cache / (key+'.partial.wav')
        try:
            # Trim in the original video timebase, including intentional silence.
            filters = (f'asetpts=PTS-STARTPTS+({offset})/TB,aresample=async=1:first_pts=0,'
                       f'apad,atrim=start={begin:.9f}:duration={duration:.9f},asetpts=PTS-STARTPTS')
            run('ffmpeg', ['-v', 'error', '-y', '-i', path, '-map', f'0:{audio["index"]}',
                           '-vn', '-af', filters, '-ar', 48000, '-ac', 2,
                           '-c:a', 'pcm_s16le', temporary])
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return target


def attach(scene, rig, data, action, previous, timeline=None):
    from sam_ellen_editing import import_audio, sound_strips
    settings = scene.sam_ellen_editing
    start = int(timeline['start']) if timeline else 1
    rate = scene.render.fps / scene.render.fps_base
    count = max(1, math.ceil((data['frames'][-1]['time']-data['frames'][0]['time'])*rate-1e-7)+1)
    # Keep preceding sound when appending, including a reapplication of that segment.
    names = [strip.name for strip in sound_strips(scene)
             if timeline and strip.name in track_names(previous)
             and strip.frame_final_start < start]
    created = None
    message = '未导入视频声音'
    if settings.video_audio:
        try:
            path = extract(data.get('video', ''), data['frames'][0]['time'], count/rate)
            if path:
                created = next((strip for strip in sound_strips(scene)
                                if strip.get('sam_audio_owner') == rig.name
                                and strip.get('sam_audio_source') == str(path)
                                and strip.frame_final_start == start
                                and strip.frame_final_duration == count), None)
                if created is None:
                    created = import_audio(scene, str(path), start)
                    created.name = '视频声音 - '+Path(data['video']).stem
                    created['sam_audio_owner'] = rig.name
                    created['sam_audio_source'] = str(path)
                    created.frame_final_end = start+count
                names.append(created.name)
                message = f'视频声音已同步至第 {start}–{start+count-1} 帧'
            else:
                message = '视频没有音轨，已仅导入动作'
        except Exception as error:
            # Sound import must not discard a successfully generated performance.
            message = '动作已保留，但视频声音导入失败：'+str(error)
    action['sam_audio_tracks'] = json.dumps(names, ensure_ascii=False)
    activate(scene, rig, action)
    if names and settings.movie_audio:
        prepare_movie(scene)
    settings.status = message
    return created

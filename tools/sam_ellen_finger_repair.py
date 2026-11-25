"""Blender-side optional worker adapter; OFF never launches any repair process."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile


def restore_experiment(data):
    if '_finger_reliability_original' not in data and 'finger_reliability' not in data:
        return data
    result = copy.deepcopy(data)
    for item in result.pop('_finger_reliability_original', []):
        for k, value in item['positions'].items():
            result['frames'][item['row']]['positions'][int(k)] = value
    result.pop('finger_reliability', None)
    return result


def prepare_experiment(data, path, enabled=False, side='L'):
    baseline = restore_experiment(data)
    if not enabled:
        return baseline
    root = Path.home()/'Desktop/小小布朗尼/tools'
    predictions = Path(path).with_name('wilor_predictions.json')
    if not predictions.is_file():
        raise ValueError('实验修复需要原始 WiLoR 缓存，请先补充 / 重新识别手掌与手指')
    with tempfile.TemporaryDirectory(prefix='sam_finger_repair_') as directory:
        source, output = Path(directory)/'input.json', Path(directory)/'output.json'
        source.write_text(json.dumps(baseline, ensure_ascii=False, allow_nan=False), encoding='utf8')
        job = subprocess.run([str(root/'motion_venv/Scripts/python.exe'),
            str(root/'finger_reliability_repair.py'), '--take', str(source),
            '--predictions', str(predictions), '--output', str(output), '--side', side],
            capture_output=True, timeout=90, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if job.returncode:
            error = job.stderr.decode('utf8', errors='replace')[-600:]
            raise RuntimeError('实验性手指修复未完成，已保留旧动作：'+error)
        return json.loads(output.read_text(encoding='utf8'))

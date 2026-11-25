"""Offline adaptation of LivLiv Player's speech envelope and spectral vowel cue.

Copyright 2026 LivLiv. Licensed under Apache-2.0; see vendor/livliv/.
Modified for this project: Python/NumPy, configurable lead and Blender shape map.
Upstream: 83ecd9c48f33d9439411e596841193c24dc7b005, runtime/speech.ts,
runtime/vowel.ts and vrm.ts. This estimates three mouth families, not phonemes.
"""
import math
import numpy as np

RATE = 100
VOWELS = ('あ', 'い', 'う', 'え', 'お', 'ワ')


def envelope(samples, sr):
    hop = max(1, math.floor(sr / RATE + .5))
    n = math.ceil(len(samples) / hop)
    if not n:
        return np.zeros(0)
    power = [0.0] * n
    lp, hp = (1 - math.exp(-2 * math.pi * f / sr) for f in (1500, 150))
    a = b = 0.0
    for i, x in enumerate(samples.tolist()):
        a += (x-a)*lp
        b += (a-b)*hp
        power[i // hop] += (a-b)**2
    power = np.asarray(power)
    energy = np.sqrt((power + np.r_[power[1:], 0]) / (2*hop))
    peak = float(energy.max())
    if peak == 0:
        return np.zeros(n)
    voiced = np.sort(energy[energy > peak*.05])
    ref = voiced[math.floor(len(voiced)*.9)] if len(voiced) else peak
    gate = ref*.1
    level = np.clip((energy-gate)/max(ref-gate, 1e-6), 0, 1)**.75
    average = np.convolve(np.pad(level, 10, mode='edge'), np.ones(21)/21, mode='valid')
    target = np.clip(level*.6 + (level-average)*1.2, 0, 1)
    out = np.empty(n)
    value = 0.0
    for i, goal in enumerate(target):
        value += (goal-value)*(.6 if goal > value else .45)
        out[i] = value
    return out


class VowelCue:
    def __init__(self):
        self.center, self.value, self.seen = 15.0, 0.0, 0.0

    def push(self, power, hz, level, dt):
        def band(a, b):
            return 1e-12 + power[math.ceil(a/hz):min(len(power), math.ceil(b/hz))].sum()
        back = 10*math.log10(band(300, 900)/band(1800, 3500))
        close = 10*math.log10(band(250, 650)/band(650, 1300))
        target = 0.0
        if level > .15:
            if level > .3:
                self.seen += dt
                k = min(1, dt/min(3, .3+self.seen))
                self.center += (min(24, max(8, back))-self.center)*k
            rounded = (back-self.center)/10
            wide = ((self.center-back)/3)*min(1, max(0, (close-8+4)/4))
            target = max(-1, min(1, wide if wide > 0 else -max(0, rounded)))
        self.value += (target-self.value)*min(1, dt/.035)
        return self.value


def analyze(samples, sr=48000, lead_ms=30):
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('音频为空或包含无效采样')
    if sr < 8000 or not math.isfinite(lead_ms) or not -200 <= lead_ms <= 200:
        raise ValueError('无效采样率或口型提前量')
    raw = envelope(x, sr)
    times = np.arange(len(raw))/RATE
    ahead = times + lead_ms/1000
    env = np.interp(ahead, times, raw, left=0, right=0)
    size = 1 << math.floor(math.log2(.0213*sr)+.5)
    window = np.blackman(size)
    cue = VowelCue()
    vowel = np.empty(len(env))
    for f, level in enumerate(env):
        center = math.floor(ahead[f]*sr+.5)
        first = center-size//2
        frame = np.zeros(size)
        lo, hi = max(0, first), min(len(x), first+size)
        if hi > lo:
            frame[lo-first:hi-first] = x[lo:hi]
        spectrum = np.fft.rfft(frame*window)[:size//2]
        vowel[f] = cue.push(np.abs(spectrum)**2, sr/size, level, 1/RATE)
    # A final neutral sample prevents a held-open mouth after playback.
    return {'times': np.r_[times, len(x)/sr], 'open': np.r_[env, 0],
            'vowel': np.r_[vowel, 0], 'duration': len(x)/sr}


def mouth_weights(opening, vowel, gain=1):
    opening = min(1, max(0, opening*gain))
    vowel = min(1, max(-1, vowel))
    result = {n: 0.0 for n in VOWELS}
    result['あ'] = opening*(1-abs(vowel)*.6)
    result['え'] = opening*max(0, vowel)*.7
    result['う'] = opening*max(0, -vowel)*.7
    budget = max(1, sum(result.values()))
    return {n: value/budget for n, value in result.items()}

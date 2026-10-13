"""
Day 22: 优化版动作库
- 幅度统一（<= 0.35）
- 频率统一（<= 1.5 Hz）
- 平衡优先
"""
import numpy as np

STAND = np.array([0.0, 0.8, -1.5] * 4)

def fade(t_local, duration, fade_time=0.4):
    fi = min(t_local / fade_time, 1.0)
    fo = min((duration - t_local) / fade_time, 1.0)
    return min(fi, fo)

def act_stand(t, d):
    return STAND.copy()

def act_nod(t, d):
    target = STAND.copy()
    f = fade(t, d)
    swing = 0.30 * np.sin(2 * np.pi * 1.0 * t) * f
    target[2] = -1.5 + swing
    target[5] = -1.5 + swing
    return target

def act_sway(t, d):
    target = STAND.copy()
    f = fade(t, d)
    sway = 0.25 * np.sin(2 * np.pi * 0.5 * t) * f
    target[1] = 0.8 + sway
    target[4] = 0.8 - sway
    return target

def act_shake(t, d):
    target = STAND.copy()
    f = fade(t, d)
    shake = 0.20 * np.sin(2 * np.pi * 1.5 * t) * f
    target[0] = shake
    target[3] = -shake
    target[6] = -shake
    target[9] = shake
    return target

def act_stretch(t, d):
    target = STAND.copy()
    f = fade(t, d)
    phase = t / d
    s = np.sin(np.pi * phase) * f
    for leg in range(4):
        target[leg * 3 + 2] = -1.5 + 0.30 * s
    return target

def act_squat(t, d):
    target = STAND.copy()
    f = fade(t, d)
    phase = t / d
    s = np.sin(np.pi * phase) * f
    for leg in range(4):
        target[leg * 3 + 1] = 0.8 + 0.35 * s
        target[leg * 3 + 2] = -1.5 - 0.30 * s
    return target

def act_dance_spin(t, d):
    target = STAND.copy()
    f = fade(t, d)
    spin = 0.30 * np.sin(2 * np.pi * 1.0 * t) * f
    for leg in range(4):
        target[leg * 3 + 0] = spin
    return target

def act_bow(t, d):
    target = STAND.copy()
    f = fade(t, d)
    phase = t / d
    s = np.sin(np.pi * phase) * f
    target[2] = -1.5 + 0.30 * s
    target[5] = -1.5 + 0.30 * s
    target[8] = -1.5 - 0.15 * s
    target[11] = -1.5 - 0.15 * s
    return target

ACTIONS_V2 = {
    "stand": act_stand,
    "nod": act_nod,
    "sway": act_sway,
    "shake": act_shake,
    "stretch": act_stretch,
    "squat": act_squat,
    "dance_spin": act_dance_spin,
    "bow": act_bow,
}

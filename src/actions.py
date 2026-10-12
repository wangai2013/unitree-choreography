"""
Day 7: 动作库（含"可视化跳跃"）
"""
import numpy as np

STAND = np.array([0.0, 0.8, -1.5] * 4)

def fade(t_local, duration, fade_time=0.3):
    fi = min(t_local / fade_time, 1.0)
    fo = min((duration - t_local) / fade_time, 1.0)
    return min(fi, fo)

def act_stand(t_local, duration):
    return STAND.copy()

def act_nod(t_local, duration):
    target = STAND.copy()
    f = fade(t_local, duration)
    swing = 0.4 * np.sin(2 * np.pi * 1.5 * t_local) * f
    target[2] = -1.5 + swing
    target[5] = -1.5 + swing
    return target

def act_sway(t_local, duration):
    target = STAND.copy()
    f = fade(t_local, duration)
    sway = 0.15 * np.sin(2 * np.pi * 1.0 * t_local) * f
    target[1] = 0.8 + sway
    target[4] = 0.8 - sway
    return target

def act_bow(t_local, duration):
    target = STAND.copy()
    f = fade(t_local, duration)
    phase = t_local / duration
    bow_curve = np.sin(np.pi * phase) * f
    target[2] = -1.5 + 0.5 * bow_curve
    target[5] = -1.5 + 0.5 * bow_curve
    target[8] = -1.5 - 0.2 * bow_curve
    target[11] = -1.5 - 0.2 * bow_curve
    return target

def act_squat(t_local, duration):
    target = STAND.copy()
    f = fade(t_local, duration)
    phase = t_local / duration
    squat_curve = np.sin(np.pi * phase) * f
    for leg in range(4):
        target[leg * 3 + 1] = 0.8 + 0.6 * squat_curve
        target[leg * 3 + 2] = -1.5 - 0.5 * squat_curve
    return target

def act_jump(t_local, duration):
    """跳跃：蹲 → 蹬地 → 腾空 → 落地"""
    target = STAND.copy()
    f = fade(t_local, duration)
    phase = t_local / duration

    # 阶段划分：0~0.3 蹲，0.3~0.5 蹬，0.5~0.8 空中，0.8~1 落地
    if phase < 0.3:
        # 蹲
        p = phase / 0.3
        for leg in range(4):
            target[leg * 3 + 1] = 0.8 + 0.7 * p * f
            target[leg * 3 + 2] = -1.5 - 0.6 * p * f
    elif phase < 0.5:
        # 蹬地（快速伸展）
        for leg in range(4):
            target[leg * 3 + 1] = 0.8 - 0.5 * f
            target[leg * 3 + 2] = -1.5 + 0.5 * f
    else:
        # 腾空（保持收腿）
        for leg in range(4):
            target[leg * 3 + 1] = 0.8 + 0.3 * f
            target[leg * 3 + 2] = -1.5 - 0.3 * f
    return target

ACTIONS = {
    "stand": act_stand,
    "nod":   act_nod,
    "sway":  act_sway,
    "bow":   act_bow,
    "squat": act_squat,
    "jump":  act_jump,
}

# ===== Day 17: 新增动作 =====

def act_wave_hi(t_local, duration):
    """挥手打招呼：前腿抬起 + 挥动"""
    target = STAND.copy()
    f = fade(t_local, duration)
    phase = t_local / duration
    lift_curve = np.sin(np.pi * phase)
    lift = 0.4 * lift_curve * f
    target[1] = 0.8 + lift * 0.6     # FR 大腿抬起
    target[2] = -1.5 + lift * 1.2    # FR 小腿伸展
    wave = 0.3 * np.sin(2 * np.pi * 3.0 * t_local) * f
    target[2] += wave
    # 其他三条腿撑地
    target[4] += 0.08 * f
    target[7] += 0.08 * f
    target[10] += 0.08 * f
    return target

def act_kick(t_local, duration):
    """踢腿：后腿往前踢"""
    target = STAND.copy()
    f = fade(t_local, duration)
    phase = t_local / duration
    kick_curve = np.sin(np.pi * phase) * f
    # RR 腿踢出
    target[7] = 0.8 - 0.5 * kick_curve   # RR 大腿往前
    target[8] = -1.5 + 0.8 * kick_curve  # RR 小腿伸直
    return target

def act_spin(t_local, duration):
    """转圈（简化）：髋关节摆动 + 机身转向"""
    target = STAND.copy()
    f = fade(t_local, duration)
    phase = t_local / duration
    spin_curve = np.sin(2 * np.pi * 2.0 * t_local) * f
    # 四条腿同时摆动
    for leg in range(4):
        target[leg * 3 + 0] = 0.3 * spin_curve
    return target

ACTIONS["wave_hi"] = act_wave_hi
ACTIONS["kick"] = act_kick
ACTIONS["spin"] = act_spin

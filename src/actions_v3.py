"""
Day 23: 全身动作库
- 腿动 + 机身动
- 每个动作返回 (target_joints, body_pitch, body_roll, body_yaw)
"""
import numpy as np

STAND = np.array([0.0, 0.8, -1.5] * 4)

def fade(t_local, duration, fade_time=0.3):
    fi = min(t_local / fade_time, 1.0)
    fo = min((duration - t_local) / fade_time, 1.0)
    return min(fi, fo)

def act_stand(t, d):
    return STAND.copy(), 0.0, 0.0, 0.0

def act_nod(t, d):
    """点头：前腿折叠 + 后腿伸展 + 机身前倾"""
    target = STAND.copy()
    f = fade(t, d)
    swing = 0.30 * np.sin(2 * np.pi * 1.5 * t) * f

    # 前腿：小腿折叠（缩短）→ 机身前端下降
    target[2] = -1.5 - swing * 0.8
    target[5] = -1.5 - swing * 0.8
    # 后腿：小腿伸展（加长）→ 机身后端抬高
    target[8] = -1.5 + swing * 0.4
    target[11] = -1.5 + swing * 0.4

    # 机身：俯仰（pitch）—— 前倾
    pitch = 0.15 * np.sin(2 * np.pi * 1.5 * t) * f

    return target, pitch, 0.0, 0.0

def act_sway(t, d):
    """摇摆：左右腿交替 + 机身侧倾"""
    target = STAND.copy()
    f = fade(t, d)
    sway = 0.25 * np.sin(2 * np.pi * 1.0 * t) * f

    # 左侧腿（FL, RL）：伸展
    # 右侧腿（FR, RR）：收缩
    target[4] = 0.8 - sway * 0.5   # FL thigh
    target[10] = 0.8 - sway * 0.5  # RL thigh
    target[1] = 0.8 + sway * 0.5   # FR thigh
    target[7] = 0.8 + sway * 0.5   # RR thigh

    # 机身：侧倾（roll）
    roll = 0.18 * np.sin(2 * np.pi * 1.0 * t) * f

    return target, 0.0, roll, 0.0

def act_shake(t, d):
    """摇头：髋关节 + 机身偏航"""
    target = STAND.copy()
    f = fade(t, d)
    shake = 0.20 * np.sin(2 * np.pi * 2.0 * t) * f

    target[0] = shake
    target[3] = -shake
    target[6] = -shake
    target[9] = shake

    # 机身：偏航（yaw）—— 左右转
    yaw = 0.15 * np.sin(2 * np.pi * 2.0 * t) * f

    return target, 0.0, 0.0, yaw

def act_stretch(t, d):
    """伸展：腿伸直 + 机身上升"""
    target = STAND.copy()
    f = fade(t, d)
    phase = t / d
    s = np.sin(np.pi * phase) * f

    for leg in range(4):
        target[leg * 3 + 2] = -1.5 + 0.30 * s

    # 机身：稍微前倾（前腿伸直会让重心前移）
    pitch = 0.05 * s

    return target, pitch, 0.0, 0.0

def act_squat(t, d):
    """下蹲：腿折叠 + 机身下沉 + 前倾"""
    target = STAND.copy()
    f = fade(t, d)
    phase = t / d
    s = np.sin(np.pi * phase) * f

    for leg in range(4):
        target[leg * 3 + 1] = 0.8 + 0.35 * s   # 大腿
        target[leg * 3 + 2] = -1.5 - 0.30 * s  # 小腿

    # 机身：前倾（蹲下时"重心前移"是自然的）
    pitch = 0.10 * s

    return target, pitch, 0.0, 0.0

def act_dance_spin(t, d):
    """转圈：髋关节摆动 + 机身偏航"""
    target = STAND.copy()
    f = fade(t, d)
    spin = 0.30 * np.sin(2 * np.pi * 1.0 * t) * f

    for leg in range(4):
        target[leg * 3 + 0] = spin

    # 机身：yaw（绕 z 轴）—— 转向
    yaw = 0.25 * np.sin(2 * np.pi * 1.0 * t) * f

    return target, 0.0, 0.0, yaw

def act_bow(t, d):
    """鞠躬：前腿伸展 + 后腿折叠 + 机身大幅前倾"""
    target = STAND.copy()
    f = fade(t, d)
    phase = t / d
    s = np.sin(np.pi * phase) * f

    # 前腿：折叠（缩短）
    target[2] = -1.5 - 0.20 * s
    target[5] = -1.5 - 0.20 * s
    # 后腿：伸展（加长）
    target[8] = -1.5 + 0.20 * s
    target[11] = -1.5 + 0.20 * s

    # 机身：大幅前倾（鞠躬）
    pitch = 0.30 * s

    return target, pitch, 0.0, 0.0

ACTIONS_V3 = {
    "stand": act_stand,
    "nod": act_nod,
    "sway": act_sway,
    "shake": act_shake,
    "stretch": act_stretch,
    "squat": act_squat,
    "dance_spin": act_dance_spin,
    "bow": act_bow,
}

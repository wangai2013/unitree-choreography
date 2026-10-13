"""
Day 28: G1 动作库（29dof 版）
- waist_pitch / waist_roll 是"真关节"
"""
import numpy as np

STAND = np.zeros(29)
# 腿
STAND[0] = -0.2; STAND[3] = 0.4; STAND[4] = -0.2
STAND[6] = -0.2; STAND[9] = 0.4; STAND[10] = -0.2

# 手臂"自然下垂"
STAND[15] = 0.2; STAND[16] = -0.1; STAND[18] = 0.8   # 左臂
STAND[22] = 0.2; STAND[23] = -0.1; STAND[25] = 0.8   # 右臂

def fade(t_local, duration, fade_time=0.3):
    fi = min(t_local / fade_time, 1.0)
    fo = min((duration - t_local) / fade_time, 1.0)
    return min(fi, fo)

def act_stand(t, d):
    return STAND.copy()

def act_wave_hand(t, d):
    target = STAND.copy()
    f = fade(t, d)
    target[22] = 0.0
    target[23] = 1.2
    target[25] = -1.2
    target[26] = -1.57
    wave = 0.5 * np.sin(2 * np.pi * 2.0 * t) * f
    target[24] = wave
    return target

def act_bow(t, d):
    """鞠躬（29dof 的 waist_pitch = 索引 14）"""
    target = STAND.copy()
    f = fade(t, d)
    phase = t / d
    s = np.sin(np.pi * phase) * f
    target[14] = +1.0 * s   # ⭐ 正 = 前倾（鞠躬）
    return target

def act_turn_left(t, d):
    target = STAND.copy()
    f = fade(t, d)
    phase = t / d
    s = np.sin(np.pi * phase) * f
    target[12] = 0.7 * s
    return target

def act_wave_both(t, d):
    target = STAND.copy()
    f = fade(t, d)
    target[15] = 0.0; target[16] = -1.2; target[18] = -1.2; target[19] = +1.57
    target[22] = 0.0; target[23] =  1.2; target[25] = -1.2; target[26] = -1.57
    wave = 0.5 * np.sin(2 * np.pi * 2.0 * t) * f
    target[17] = -wave
    target[24] =  wave
    return target

ACTIONS = {
    "stand":      act_stand,
    "wave_hand":  act_wave_hand,
    "bow":        act_bow,
    "turn_left":  act_turn_left,
    "wave_both":  act_wave_both,
}


# ========== Day 30: 新增动作 ==========

def act_wave_slow(t, d):
    """慢挥手（更优雅）"""
    target = STAND.copy()
    f = fade(t, d)
    target[22] = 0.0
    target[23] = 1.3
    target[25] = -1.0
    target[26] = -1.57
    wave = 0.6 * np.sin(2 * np.pi * 0.8 * t) * f  # 慢频
    target[24] = wave
    return target

def act_nod_double(t, d):
    """双点头（快速两次）"""
    target = STAND.copy()
    f = fade(t, d)
    swing = 0.3 * np.sin(2 * np.pi * 1.5 * t) * f
    target[14] = swing   # waist_pitch
    return target

def act_cheer(t, d):
    """欢呼（双手向上）"""
    target = STAND.copy()
    f = fade(t, d)
    phase = t / d
    s = np.sin(np.pi * phase) * f

    # 双臂向上
    target[15] = 0.0; target[16] = -1.8; target[18] = -1.5; target[19] = 1.57
    target[22] = 0.0; target[23] =  1.8; target[25] = -1.5; target[26] = -1.57

    # 腰 pitch 前倾
    target[14] = 0.2 * s

    return target

# 注册
ACTIONS["wave_slow"] = act_wave_slow
ACTIONS["nod_double"] = act_nod_double
ACTIONS["cheer"] = act_cheer

print("✅ 新增 3 个动作: wave_slow, nod_double, cheer")

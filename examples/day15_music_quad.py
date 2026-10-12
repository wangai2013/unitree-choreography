"""
Day 15: 4 台 Go2 音乐同步舞蹈
- librosa 分析音乐节拍
- pygame 播放音乐
- 4 台按节拍做动作
"""
import mujoco
import mujoco.viewer
import numpy as np
import time
import librosa
import pygame

MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene_quad_final.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# ===== 1. 分析音乐 =====
MUSIC_FILE = "music.mp3"
print(f"🎵 加载音乐: {MUSIC_FILE}")

pygame.mixer.init()
pygame.mixer.music.load(MUSIC_FILE)

y, sr = librosa.load(MUSIC_FILE, sr=None)
tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")

print(f"🥁 BPM: {float(tempo):.1f}")
print(f"🎯 节拍数: {len(beats)}")

# ===== 2. 4 台设置 =====
N = 4
FREE_JOINTS = [i for i in range(model.njnt) if model.jnt_type[i] == 0]
QPOS_BASES = [model.jnt_qposadr[j] for j in FREE_JOINTS]

CTRL_TO_QPOS = []
CTRL_TO_QVEL = []
for r in range(N):
    qpos_r = np.zeros(12, dtype=int)
    qvel_r = np.zeros(12, dtype=int)
    for i in range(12):
        jid = model.actuator_trnid[r * 12 + i][0]
        qpos_r[i] = model.jnt_qposadr[jid]
        qvel_r[i] = model.jnt_dofadr[jid]
    CTRL_TO_QPOS.append(qpos_r)
    CTRL_TO_QVEL.append(qvel_r)

STAND = np.array([0.0, 0.8, -1.5] * 4)
for r in range(N):
    for i in range(12):
        data.qpos[CTRL_TO_QPOS[r][i]] = STAND[i]
mujoco.mj_forward(model, data)

KP, KD = 300.0, 15.0
HEIGHT = 0.32

# 初始站位（方阵）
FORMATION = [(-1.2, -0.6), (1.2, -0.6), (-1.2, 0.6), (1.2, 0.6)]

# ===== 3. 舞蹈编排（按拍数）=====
BEATS_PER_ACTION = 4
DANCE_ACTIONS = [
    "stand", "sync_nod", "wave_nod", "mirror_sway",
    "sync_nod", "wave_nod", "mirror_sway", "stand",
]

def get_beat_action(t):
    """根据音乐时间 t，返回 (动作, 局部时间, 时长)"""
    for i, action in enumerate(DANCE_ACTIONS):
        beat_start = i * BEATS_PER_ACTION
        beat_end = (i + 1) * BEATS_PER_ACTION
        if beat_start >= len(beats):
            break
        t_start = beats[beat_start]
        t_end = beats[beat_end] if beat_end < len(beats) else len(y) / sr
        if t_start <= t < t_end:
            return action, t - t_start, t_end - t_start
    return None, None, None

# ===== 4. 主循环 =====
step = 0
music_started = False
last_action = None

print("=" * 60)
print("🎬 4 台音乐群控舞蹈")
print("  先站 2 秒，然后音乐开始")
print("=" * 60)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001

        # 2 秒后启动音乐
        if t >= 2.0 and not music_started:
            pygame.mixer.music.play()
            music_started = True
            print("🎵 音乐开始！")

        music_t = t - 2.0

        # ===== 默认：站立 =====
        targets = [STAND.copy() for _ in range(N)]

        if music_t > 0:
            action, t_local, dur = get_beat_action(music_t)

            if action != last_action and action is not None:
                print(f"🎬 音乐 {music_t:.2f}s → {action}")
                last_action = action

            if action == "sync_nod":
                swing = 0.4 * np.sin(2 * np.pi * 1.5 * t_local)
                for r in range(N):
                    targets[r][2] = -1.5 + swing
                    targets[r][5] = -1.5 + swing

            elif action == "wave_nod":
                # 波浪：每台延迟 0.3 秒
                for r in range(N):
                    t_r = max(0, t_local - r * 0.3)
                    swing = 0.4 * np.sin(2 * np.pi * 1.5 * t_r)
                    targets[r][2] = -1.5 + swing
                    targets[r][5] = -1.5 + swing

            elif action == "mirror_sway":
                sway = 0.2 * np.sin(2 * np.pi * 1.0 * t_local)
                for r in range(N):
                    sign = +1 if r % 2 == 0 else -1
                    targets[r][1] = 0.8 + sign * sway
                    targets[r][4] = 0.8 - sign * sway

        # ===== 应用 =====
        for r in range(N):
            # 站位
            data.qpos[QPOS_BASES[r] + 0] = FORMATION[r][0]
            data.qpos[QPOS_BASES[r] + 1] = FORMATION[r][1]
            data.qpos[QPOS_BASES[r] + 2] = HEIGHT
            data.qvel[QPOS_BASES[r]:QPOS_BASES[r] + 6] = 0

            # PD 控制
            for i in range(12):
                q = data.qpos[CTRL_TO_QPOS[r][i]]
                dq = data.qvel[CTRL_TO_QVEL[r][i]]
                data.ctrl[r * 12 + i] = KP * (targets[r][i] - q) - KD * dq

        mujoco.mj_step(model, data)
        viewer.sync()
        step += 1
        time.sleep(0.001)

pygame.mixer.music.stop()

"""
Day 21: 作品级舞蹈（全身版）
腿动 + 机身晃 = 真跳舞
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import mujoco
import mujoco.viewer
import numpy as np
import time
import json
import pygame

# 加载舞蹈
DANCE_FILE = "dances/masterpiece.json"
with open(DANCE_FILE, "r", encoding="utf-8") as f:
    dance = json.load(f)

print("=" * 60)
print(f"🎬 {dance['name']}")
print(f"📝 {dance['description']}")
print("=" * 60)

# 音乐
pygame.mixer.init()
pygame.mixer.music.load(dance.get("music", "music.mp3"))
print("🎵 音乐就绪")

# 模型
N = 4
MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene_quad_final.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

CTRL_TO_QPOS = []
for r in range(N):
    qpos_r = np.zeros(12, dtype=int)
    for i in range(12):
        jid = model.actuator_trnid[r * 12 + i][0]
        qpos_r[i] = model.jnt_qposadr[jid]
    CTRL_TO_QPOS.append(qpos_r)

FREE_JOINTS = [i for i in range(model.njnt) if model.jnt_type[i] == 0]
QPOS_BASES = [model.jnt_qposadr[j] for j in FREE_JOINTS]

STAND = np.array([0.0, 0.8, -1.5] * 4)
HEIGHT = 0.32
FORMATION = dance["formation"]

BPM = dance["bpm"]
BEAT_INTERVAL = 60.0 / BPM

timeline = []
acc = 0.0
for seg in dance["sequence"]:
    dur = seg["beats"] * BEAT_INTERVAL
    timeline.append((seg["action"], acc, acc + dur))
    acc += dur
total = acc

print(f"⏱  总时长: {total:.1f} 秒")
print("=" * 60)

def get_action(t):
    for action, t0, t1 in timeline:
        if t0 <= t < t1:
            return action, t - t0, t1 - t0
    return None, None, None

def fade(t_local, duration, fade_time=0.3):
    fi = min(t_local / fade_time, 1.0)
    fo = min((duration - t_local) / fade_time, 1.0)
    return min(fi, fo)

def compute_target(action, t_local, duration):
    target = STAND.copy()
    f = fade(t_local, duration)
    
    if action == "nod":
        swing = 0.4 * np.sin(2 * np.pi * 1.5 * t_local) * f
        target[2] = -1.5 + swing
        target[5] = -1.5 + swing
    elif action == "shake":
        shake = 0.3 * np.sin(2 * np.pi * 3.0 * t_local) * f
        target[0] = shake
        target[3] = -shake
        target[6] = -shake
        target[9] = shake
    elif action == "stretch":
        phase = t_local / duration
        s = np.sin(np.pi * phase) * f
        for leg in range(4):
            target[leg * 3 + 2] = -1.5 + 0.5 * s
    elif action == "squat":
        phase = t_local / duration
        s = np.sin(np.pi * phase) * f
        for leg in range(4):
            target[leg * 3 + 1] = 0.8 + 0.6 * s
            target[leg * 3 + 2] = -1.5 - 0.5 * s
    elif action == "dance_spin":
        spin = 0.4 * np.sin(2 * np.pi * 1.5 * t_local) * f
        for leg in range(4):
            target[leg * 3 + 0] = spin
    elif action == "bow":
        phase = t_local / duration
        s = np.sin(np.pi * phase) * f
        target[2] = -1.5 + 0.5 * s
        target[5] = -1.5 + 0.5 * s
        target[8] = -1.5 - 0.2 * s
        target[11] = -1.5 - 0.2 * s
    elif action == "sit":
        phase = t_local / duration
        s = np.sin(np.pi * phase) * f
        target[7] = 0.8 + 0.6 * s
        target[10] = 0.8 + 0.6 * s
        target[8] = -1.5 - 0.4 * s
        target[11] = -1.5 - 0.4 * s
    return target

# ⭐ 全身摇摆函数
def compute_body_sway(action, t_local, duration, music_t):
    """返回 (sway_x, sway_y, sway_z, body_angle)"""
    if action == "nod":
        return 0.0, 0.0, 0.03 * np.sin(2 * np.pi * 1.5 * t_local), 0.0
    elif action == "shake":
        return 0.04 * np.sin(2 * np.pi * 3.0 * t_local), 0.0, 0.0, 0.05 * np.sin(2 * np.pi * 3.0 * t_local)
    elif action == "sway":
        return 0.06 * np.sin(2 * np.pi * 1.0 * t_local), 0.0, 0.01 * np.sin(2 * np.pi * 2.0 * t_local), 0.05 * np.sin(2 * np.pi * 1.0 * t_local)
    elif action == "stretch":
        phase = t_local / duration if duration > 0 else 0
        s = np.sin(np.pi * phase)
        return 0.0, 0.0, 0.05 * s, 0.0
    elif action == "squat":
        phase = t_local / duration if duration > 0 else 0
        s = np.sin(np.pi * phase)
        return 0.0, 0.0, -0.08 * s, 0.0
    elif action == "dance_spin":
        angle = 2 * np.pi * 1.0 * t_local
        return 0.04 * np.sin(angle), 0.04 * np.cos(angle), 0.01 * np.sin(2 * angle), angle * 0.3
    elif action == "bow":
        phase = t_local / duration if duration > 0 else 0
        s = np.sin(np.pi * phase)
        return 0.0, 0.05 * s, -0.02 * s, 0.0
    elif action == "sit":
        phase = t_local / duration if duration > 0 else 0
        s = np.sin(np.pi * phase)
        return 0.0, -0.04 * s, -0.06 * s, 0.0
    else:
        # 默认：呼吸
        return 0.0, 0.0, 0.01 * np.sin(2 * np.pi * 0.5 * music_t), 0.0

# 主循环
step = 0
music_started = False
last_action = None

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001

        if t >= 2.0 and not music_started:
            pygame.mixer.music.play()
            music_started = True
            print("🎵 音乐开始")

        music_t = t - 2.0

        targets = [STAND.copy() for _ in range(N)]
        positions = [list(FORMATION[r]) for r in range(N)]
        action = None
        t_local = 0
        dur = 1

        if music_t > 0:
            action, t_local, dur = get_action(music_t)

            if action != last_action and action is not None:
                print(f"🎬 {music_t:.2f}s → {action}")
                last_action = action

            if action is not None:
                target = compute_target(action, t_local, dur)
                for r in range(N):
                    targets[r] = target

        # ⭐ 全身摇晃
        sway_x, sway_y, sway_z, body_angle = compute_body_sway(action, t_local, dur, music_t)

        # 应用
        for r in range(N):
            data.qpos[QPOS_BASES[r] + 0] = positions[r][0] + sway_x
            data.qpos[QPOS_BASES[r] + 1] = positions[r][1] + sway_y
            data.qpos[QPOS_BASES[r] + 2] = HEIGHT + sway_z

            # 姿态
            data.qpos[QPOS_BASES[r] + 3] = np.cos(body_angle / 2)
            data.qpos[QPOS_BASES[r] + 4] = 0.0
            data.qpos[QPOS_BASES[r] + 5] = 0.0
            data.qpos[QPOS_BASES[r] + 6] = np.sin(body_angle / 2)

            for i in range(12):
                data.qpos[CTRL_TO_QPOS[r][i]] = targets[r][i]

        data.qvel[:] = 0
        mujoco.mj_forward(model, data)

        viewer.sync()
        step += 1
        time.sleep(0.001)

if music_started:
    pygame.mixer.music.stop()

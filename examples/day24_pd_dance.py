"""
Day 24: PD 驱动 + 混合约束（锁 x/y，z 自由）
- PD 控制腿
- 物理驱动（mj_step）
- 锁 x/y（防碰撞），z 自由（腿撑）
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

with open("dances/masterpiece_v3.json", "r", encoding="utf-8") as f:
    dance = json.load(f)

print("=" * 60)
print(f"🎬 {dance['name']} (PD + 混合约束)")
print("=" * 60)

pygame.mixer.init()
pygame.mixer.music.load(dance["music"])
print("🎵 音乐就绪")

N = 4
MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene_quad_final.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

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

FREE_JOINTS = [i for i in range(model.njnt) if model.jnt_type[i] == 0]
QPOS_BASES = [model.jnt_qposadr[j] for j in FREE_JOINTS]
QVEL_BASES = [model.jnt_dofadr[j] for j in FREE_JOINTS]

HEIGHT = 0.32

FORMATIONS = {
    "square":  [(-1.5, -0.8), (1.5, -0.8), (-1.5, 0.8), (1.5, 0.8)],
    "diamond": [(0.0, -1.5), (1.5, 0.0), (-1.5, 0.0), (0.0, 1.5)],
    "circle":  [(1.5, 0.0), (0.0, 1.5), (-1.5, 0.0), (0.0, -1.5)],
}

STAND = np.array([0.0, 0.8, -1.5] * 4)

for r in range(N):
    for i in range(12):
        data.qpos[CTRL_TO_QPOS[r][i]] = STAND[i]
    data.qpos[QPOS_BASES[r] + 0] = FORMATIONS["square"][r][0]
    data.qpos[QPOS_BASES[r] + 1] = FORMATIONS["square"][r][1]
    data.qpos[QPOS_BASES[r] + 2] = HEIGHT
    data.qpos[QPOS_BASES[r] + 3] = 1.0

mujoco.mj_forward(model, data)

BPM = dance["bpm"]
BEAT_INTERVAL = 60.0 / BPM

timeline = []
acc = 0.0
for seg in dance["sequence"]:
    dur = seg["beats"] * BEAT_INTERVAL
    timeline.append((seg["action"], seg["formation"], acc, acc + dur))
    acc += dur
total = acc

print(f"⏱  总时长: {total:.1f} 秒")
print("=" * 60)

def get_segment(t):
    for action, formation, t0, t1 in timeline:
        if t0 <= t < t1:
            return action, formation, t - t0, t1 - t0
    return None, None, None, None

def fade(t_local, duration, fade_time=0.3):
    fi = min(t_local / fade_time, 1.0)
    fo = min((duration - t_local) / fade_time, 1.0)
    return min(fi, fo)

def act_stand(t, d):
    return STAND.copy()

def act_nod(t, d):
    target = STAND.copy()
    f = fade(t, d)
    swing = 0.30 * np.sin(2 * np.pi * 1.5 * t) * f
    target[2] = -1.5 - swing * 0.8
    target[5] = -1.5 - swing * 0.8
    target[8] = -1.5 + swing * 0.4
    target[11] = -1.5 + swing * 0.4
    return target

def act_sway(t, d):
    target = STAND.copy()
    f = fade(t, d)
    sway = 0.25 * np.sin(2 * np.pi * 1.0 * t) * f
    target[4] = 0.8 - sway * 0.5
    target[10] = 0.8 - sway * 0.5
    target[1] = 0.8 + sway * 0.5
    target[7] = 0.8 + sway * 0.5
    return target

def act_shake(t, d):
    target = STAND.copy()
    f = fade(t, d)
    shake = 0.20 * np.sin(2 * np.pi * 2.0 * t) * f
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
    target[2] = -1.5 - 0.20 * s
    target[5] = -1.5 - 0.20 * s
    target[8] = -1.5 + 0.20 * s
    target[11] = -1.5 + 0.20 * s
    return target

ACTIONS = {
    "stand": act_stand,
    "nod": act_nod,
    "sway": act_sway,
    "shake": act_shake,
    "stretch": act_stretch,
    "squat": act_squat,
    "dance_spin": act_dance_spin,
    "bow": act_bow,
}

KP = 150.0
KD = 8.0

step = 0
music_started = False
last_action = None

target_x = [FORMATIONS["square"][r][0] for r in range(N)]
target_y = [FORMATIONS["square"][r][1] for r in range(N)]

print("🎬 混合模式启动...")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001

        if t >= 2.0 and not music_started:
            pygame.mixer.music.play()
            music_started = True
            print("🎵 音乐开始")

        music_t = t - 2.0

        targets = [STAND.copy() for _ in range(N)]

        if music_t > 0:
            action, formation, t_local, dur = get_segment(music_t)

            if action is not None and action != last_action:
                print(f"🎬 {music_t:.2f}s -> {action} @ {formation}")
                last_action = action

            if action is not None:
                target = ACTIONS[action](t_local, dur)
                for r in range(N):
                    targets[r] = target

                for r in range(N):
                    target_x[r] = FORMATIONS[formation][r][0]
                    target_y[r] = FORMATIONS[formation][r][1]

        # 1. PD 控制腿
        for r in range(N):
            for i in range(12):
                q = data.qpos[CTRL_TO_QPOS[r][i]]
                dq = data.qvel[CTRL_TO_QVEL[r][i]]
                data.ctrl[r * 12 + i] = KP * (targets[r][i] - q) - KD * dq

        # 2. 物理步进
        mujoco.mj_step(model, data)

        # 3. 锁 x/y（防碰撞），保留 z + 姿态
        for r in range(N):
            base = QPOS_BASES[r]
            data.qpos[base + 0] = target_x[r]
            data.qpos[base + 1] = target_y[r]
            data.qvel[QVEL_BASES[r] + 0] = 0
            data.qvel[QVEL_BASES[r] + 1] = 0

        mujoco.mj_forward(model, data)

        viewer.sync()
        step += 1
        time.sleep(0.001)

if music_started:
    pygame.mixer.music.stop()

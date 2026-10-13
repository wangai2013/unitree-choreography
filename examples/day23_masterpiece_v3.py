"""
Day 23: 全身动作版编舞
- 腿动 + 机身动（俯仰/侧倾/偏航）
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
import math
import pygame

from actions_v3 import ACTIONS_V3, STAND

with open("dances/masterpiece_v3.json", "r", encoding="utf-8") as f:
    dance = json.load(f)

print("=" * 60)
print(f"🎬 {dance['name']}")
print(f"📝 {dance['description']}")
print("=" * 60)

pygame.mixer.init()
pygame.mixer.music.load(dance["music"])
print("🎵 音乐就绪")

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

HEIGHT = 0.32

FORMATIONS = {
    "square":  [(-1.2, -0.6), (1.2, -0.6), (-1.2, 0.6), (1.2, 0.6)],
    "diamond": [(0.0, -1.2), (1.2, 0.0), (-1.2, 0.0), (0.0, 1.2)],
    "circle":  [(1.2, 0.0), (0.0, 1.2), (-1.2, 0.0), (0.0, -1.2)],
}

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

def euler_to_quat(pitch, roll, yaw):
    """Euler (rad) -> quaternion (w, x, y, z)"""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr
    return (w, x, y, z)

step = 0
music_started = False
last_action = None
last_target = STAND.copy()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001

        if t >= 2.0 and not music_started:
            pygame.mixer.music.play()
            music_started = True
            print("🎵 音乐开始")

        music_t = t - 2.0

        targets = [STAND.copy() for _ in range(N)]
        body_poses = [(0.0, 0.0, 0.0) for _ in range(N)]  # (pitch, roll, yaw)
        positions = [list(FORMATIONS["square"][r]) for r in range(N)]

        if music_t > 0:
            action, formation, t_local, dur = get_segment(music_t)

            if action is not None and action != last_action:
                print(f"🎬 {music_t:.2f}s -> {action} @ {formation}")
                last_action = action

            if action is not None:
                # ⭐ 动作 + 机身姿态
                target_raw, pitch, roll, yaw = ACTIONS_V3[action](t_local, dur)

                for r in range(N):
                    targets[r] = target_raw
                    body_poses[r] = (pitch, roll, yaw)

                for r in range(N):
                    positions[r] = list(FORMATIONS[formation][r])

        # 应用
        for r in range(N):
            data.qpos[QPOS_BASES[r] + 0] = positions[r][0]
            data.qpos[QPOS_BASES[r] + 1] = positions[r][1]
            data.qpos[QPOS_BASES[r] + 2] = HEIGHT

            # ⭐ 机身姿态（pitch / roll / yaw）
            pitch, roll, yaw = body_poses[r]
            qw, qx, qy, qz = euler_to_quat(pitch, roll, yaw)
            data.qpos[QPOS_BASES[r] + 3] = qw
            data.qpos[QPOS_BASES[r] + 4] = qx
            data.qpos[QPOS_BASES[r] + 5] = qy
            data.qpos[QPOS_BASES[r] + 6] = qz

            for i in range(12):
                data.qpos[CTRL_TO_QPOS[r][i]] = targets[r][i]

        data.qvel[:] = 0
        mujoco.mj_forward(model, data)

        viewer.sync()
        step += 1
        time.sleep(0.001)

if music_started:
    pygame.mixer.music.stop()

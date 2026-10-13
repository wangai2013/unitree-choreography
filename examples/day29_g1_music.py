"""
Day 29: G1 音乐编舞
- 音乐节拍驱动
- 每 N 拍切换动作
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import mujoco
import mujoco.viewer
import numpy as np
import time
import librosa
import pygame
from g1_actions import ACTIONS, STAND

# ========== 1. 音乐 ==========
MUSIC_FILE = "music.mp3"

pygame.mixer.init()
pygame.mixer.music.load(MUSIC_FILE)

y, sr = librosa.load(MUSIC_FILE, sr=None)
tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")

BPM = float(tempo)
BEAT_INTERVAL = 60.0 / BPM

print("=" * 60)
print(f"🎵 音乐: {MUSIC_FILE}")
print(f"🥁 BPM: {BPM:.1f}")
print(f"🎯 节拍数: {len(beats)}")
print("=" * 60)

# ========== 2. 模型 ==========
model = mujoco.MjModel.from_xml_path("unitree_mujoco/unitree_robots/g1/scene_29dof.xml")
data = mujoco.MjData(model)

CTRL_TO_QPOS = np.zeros(model.nu, dtype=int)
CTRL_TO_QVEL = np.zeros(model.nu, dtype=int)
for i in range(model.nu):
    jid = model.actuator_trnid[i][0]
    CTRL_TO_QPOS[i] = model.jnt_qposadr[jid]
    CTRL_TO_QVEL[i] = model.jnt_dofadr[jid]

for i in range(model.nu):
    data.qpos[CTRL_TO_QPOS[i]] = STAND[i]

data.qpos[2] = 0.8
data.qpos[3] = 1.0
mujoco.mj_forward(model, data)

KP = 300.0
KD = 10.0

# ========== 3. 编舞（按拍数） ==========
BEATS_PER_ACTION = 4   # 每 4 拍换一个动作

DANCE_ACTIONS = [
    ("stand",       4),
    ("wave_hand",   8),
    ("bow",         8),
    ("turn_left",   8),
    ("wave_both",   8),
    ("wave_hand",   8),
    ("stand",       4),
]

# 转成"时间轴"
timeline = []
acc_beats = 0
for action, beats_count in DANCE_ACTIONS:
    duration = beats_count * BEAT_INTERVAL
    timeline.append((action, acc_beats * BEAT_INTERVAL, (acc_beats + beats_count) * BEAT_INTERVAL))
    acc_beats += beats_count

total = acc_beats * BEAT_INTERVAL
print(f"⏱  编舞时长: {total:.1f} 秒（{acc_beats} 拍）")
print("=" * 60)

def get_action(t):
    for action, t0, t1 in timeline:
        if t0 <= t < t1:
            return action, t - t0, t1 - t0
    return None, None, None

# ========== 4. 主循环 ==========
step = 0
music_started = False
last_action = None

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001

        # 2 秒后启动音乐
        if t >= 2.0 and not music_started:
            pygame.mixer.music.play()
            music_started = True
            print("🎵 音乐开始")

        music_t = t - 2.0
        target = STAND.copy()

        if music_t > 0:
            action, t_local, dur = get_action(music_t)

            if action is not None and action != last_action:
                print(f"🎬 {music_t:.2f}s → {action}")
                last_action = action

            if action is not None:
                target = ACTIONS[action](t_local, dur)

        # PD
        for i in range(model.nu):
            q = data.qpos[CTRL_TO_QPOS[i]]
            dq = data.qvel[CTRL_TO_QVEL[i]]
            data.ctrl[i] = KP * (target[i] - q) - KD * dq

        mujoco.mj_step(model, data)

        # 锁机身
        data.qpos[0] = 0.0
        data.qpos[1] = 0.0
        data.qpos[2] = 0.8
        data.qpos[3] = 1.0
        data.qpos[4] = 0.0
        data.qpos[5] = 0.0
        data.qpos[6] = 0.0
        data.qvel[0:6] = 0

        mujoco.mj_forward(model, data)
        viewer.sync()
        step += 1
        time.sleep(0.001)

if music_started:
    pygame.mixer.music.stop()

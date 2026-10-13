"""
Day 37: 10 台 G1 + 音乐
- librosa 分析节拍
- 每 N 拍切换动作
- 10 台同步
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

# ========== 1. 音乐分析 ==========
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
print(f"⏱  时长: {len(y)/sr:.1f} 秒")
print("=" * 60)

# ========== 2. 模型 ==========
model = mujoco.MjModel.from_xml_path("unitree_mujoco/unitree_robots/g1/scene_deca_g1.xml")
data = mujoco.MjData(model)

print(f"✅ 加载: nq={model.nq}, nu={model.nu}")

FREE_JOINTS = [i for i in range(model.njnt) if model.jnt_type[i] == 0]
QPOS_BASES = [model.jnt_qposadr[j] for j in FREE_JOINTS]

N = 10
CTRL_TO_QPOS = []
CTRL_TO_QVEL = []
for r in range(N):
    qpos_r = np.zeros(29, dtype=int)
    qvel_r = np.zeros(29, dtype=int)
    for i in range(29):
        ctrl_idx = r * 29 + i
        jid = model.actuator_trnid[ctrl_idx][0]
        qpos_r[i] = model.jnt_qposadr[jid]
        qvel_r[i] = model.jnt_dofadr[jid]
    CTRL_TO_QPOS.append(qpos_r)
    CTRL_TO_QVEL.append(qvel_r)

COLS = 5
SPACING_X = 1.5
SPACING_Y = 1.5
positions = []
for i in range(N):
    row = i // COLS
    col = i % COLS
    x = (col - (COLS - 1) / 2) * SPACING_X
    y = (row - 0.5) * SPACING_Y
    positions.append((x, y))

for r in range(N):
    for i in range(29):
        data.qpos[CTRL_TO_QPOS[r][i]] = STAND[i]
    data.qpos[QPOS_BASES[r] + 0] = positions[r][0]
    data.qpos[QPOS_BASES[r] + 1] = positions[r][1]
    data.qpos[QPOS_BASES[r] + 2] = 0.8
    data.qpos[QPOS_BASES[r] + 3] = 1.0

mujoco.mj_forward(model, data)

KP, KD = 300.0, 10.0

# ========== 3. 编舞（按拍数） ==========
BEATS_PER_ACTION = 4

DANCE_ACTIONS = [
    ("stand",     4),
    ("wave_hand", 8),
    ("bow",       8),
    ("wave_both", 8),
    ("wave_hand", 8),
    ("stand",     4),
]

# 转"时间轴"
timeline = []
acc_beats = 0
for action, n_beats in DANCE_ACTIONS:
    duration = n_beats * BEAT_INTERVAL
    timeline.append((action, acc_beats * BEAT_INTERVAL, (acc_beats + n_beats) * BEAT_INTERVAL))
    acc_beats += n_beats

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

        # 10 台"同步"
        for r in range(N):
            for i in range(29):
                q = data.qpos[CTRL_TO_QPOS[r][i]]
                dq = data.qvel[CTRL_TO_QVEL[r][i]]
                data.ctrl[r * 29 + i] = KP * (target[i] - q) - KD * dq

        mujoco.mj_step(model, data)

        # 锁位置
        for r in range(N):
            data.qpos[QPOS_BASES[r] + 0] = positions[r][0]
            data.qpos[QPOS_BASES[r] + 1] = positions[r][1]
            data.qpos[QPOS_BASES[r] + 2] = 0.8
            data.qpos[QPOS_BASES[r] + 3] = 1.0
            data.qpos[QPOS_BASES[r] + 4] = 0.0
            data.qpos[QPOS_BASES[r] + 5] = 0.0
            data.qpos[QPOS_BASES[r] + 6] = 0.0
            for k in range(6):
                data.qvel[QPOS_BASES[r] - 0 + k] = 0

        mujoco.mj_forward(model, data)
        viewer.sync()
        step += 1
        time.sleep(0.001)

if music_started:
    pygame.mixer.music.stop()

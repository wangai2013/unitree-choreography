"""
Day 8: 音乐 + 仿真 同步舞蹈
- pygame.mixer 播放音乐
- MuJoCo 仿真机器人跳舞
- 两者同步启动
"""
import mujoco
import mujoco.viewer
import numpy as np
import time
import librosa
import pygame
from robot_utils import build_ctrl_to_qpos_map, pd_control
from actions import ACTIONS, STAND

# ===== 1. 加载音乐 =====
MUSIC_FILE = "music.mp3"
print(f"🎵 加载音乐: {MUSIC_FILE}")

pygame.mixer.init()
pygame.mixer.music.load(MUSIC_FILE)
print("✅ 音乐就绪")

# ===== 2. 分析节拍 =====
y, sr = librosa.load(MUSIC_FILE, sr=None)
tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")

print(f"🥁 BPM: {float(tempo):.1f}")
print(f"🎯 节拍数: {len(beats)}")
print(f"⏱  音乐时长: {len(y)/sr:.1f} 秒")

# ===== 3. 舞蹈编排 =====
BEATS_PER_ACTION = 4
DANCE_ACTIONS = [
    "stand", "nod", "sway", "bow", "squat",
    "jump", "nod", "sway", "bow", "stand",
]

dance_schedule = []
for i, action_name in enumerate(DANCE_ACTIONS):
    beat_start = i * BEATS_PER_ACTION
    beat_end = min((i + 1) * BEATS_PER_ACTION, len(beats))
    if beat_start >= len(beats):
        break
    t_start = beats[beat_start]
    t_end = beats[beat_end] if beat_end < len(beats) else len(y) / sr
    dance_schedule.append((t_start, t_end, action_name))

print("\n📋 舞蹈时间表：")
for i, (t_start, t_end, name) in enumerate(dance_schedule):
    print(f"  [{i+1:2d}] {t_start:6.2f}s ~ {t_end:6.2f}s | {name}")
print("=" * 60)

# ===== 4. 加载 MuJoCo =====
MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
CTRL_TO_QPOS, CTRL_TO_QVEL = build_ctrl_to_qpos_map(model)

for i in range(12):
    data.qpos[CTRL_TO_QPOS[i]] = STAND[i]
data.qpos[2] = 0.3
mujoco.mj_forward(model, data)

KP, KD = 300.0, 15.0

def get_current_action(t):
    for t_start, t_end, name in dance_schedule:
        if t_start <= t < t_end:
            return name, t - t_start, t_end - t_start
    return None, None, None

# ===== 5. 主循环 =====
step = 0
last_action = None
music_started = False
start_real_time = None

print("\n🎬 打开 MuJoCo 窗口后，机器人会先站 2 秒，然后音乐开始！")
print("=" * 60)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001

        # ⭐ 仿真时间 t 到 2 秒时，启动音乐
        if t >= 2.0 and not music_started:
            pygame.mixer.music.play()
            music_started = True
            start_real_time = time.time()
            print(f"🎵 音乐开始播放！")

        # 音乐时间 = 仿真时间 - 2 秒偏移
        music_time = t - 2.0

        if music_time < 0:
            # 还没开始，保持站立
            target = STAND.copy()
            data.xfrc_applied[1] = [0, 0, 0, 0, 0, 0]
        else:
            name, t_local, dur = get_current_action(music_time)

            if name is None:
                target = STAND.copy()
                data.xfrc_applied[1] = [0, 0, 0, 0, 0, 0]
            else:
                if name != last_action:
                    print(f"🎬 音乐 {music_time:.2f}s → {name}")
                    last_action = name
                target = ACTIONS[name](t_local, dur)

                if name == "jump":
                    phase = t_local / dur
                    if 0.3 < phase < 0.5:
                        data.xfrc_applied[1] = [0, 0, 10, 0, 0, 0]
                    else:
                        data.xfrc_applied[1] = [0, 0, 0, 0, 0, 0]
                else:
                    data.xfrc_applied[1] = [0, 0, 0, 0, 0, 0]

        pd_control(model, data, CTRL_TO_QPOS, CTRL_TO_QVEL, target, KP, KD)
        mujoco.mj_step(model, data)
        viewer.sync()
        step += 1
        time.sleep(0.001)

        if step % 2000 == 0:
            print(f"  t={t:.1f}s | 音乐 {music_time:+.1f}s | 动作: {(name if music_time > 0 else 'stand')}")

# 退出时停止音乐
pygame.mixer.music.stop()

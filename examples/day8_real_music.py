"""
Day 8: 真音乐节拍驱动
用 librosa 分析出的真实节拍时刻，驱动动作
"""
import mujoco
import mujoco.viewer
import numpy as np
import time
import librosa
from robot_utils import build_ctrl_to_qpos_map, pd_control
from actions import ACTIONS, STAND

# ===== 1. 分析音乐 =====
MUSIC_FILE = "music.mp3"
print(f"🎵 分析音乐: {MUSIC_FILE}")

y, sr = librosa.load(MUSIC_FILE, sr=None)
tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")

print(f"🥁 BPM: {float(tempo):.1f}")
print(f"🎯 节拍数: {len(beats)}")
print(f"⏱  音乐时长: {len(y)/sr:.1f} 秒")

# ===== 2. 舞蹈编排 =====
# 每 N 拍切换一个动作
BEATS_PER_ACTION = 4

DANCE_ACTIONS = [
    "stand",   # 前 4 拍
    "nod",     # 第 5~8 拍
    "sway",    # 第 9~12 拍
    "bow",     # 第 13~16 拍
    "squat",   # 第 17~20 拍
    "jump",    # 第 21~24 拍
    "nod",
    "sway",
    "bow",
    "stand",
]

# 计算每个动作的起止时间
dance_schedule = []   # [(start_time, end_time, action_name), ...]
for i, action_name in enumerate(DANCE_ACTIONS):
    beat_start = i * BEATS_PER_ACTION
    beat_end = min((i + 1) * BEATS_PER_ACTION, len(beats))
    if beat_start >= len(beats):
        break
    t_start = beats[beat_start]
    if beat_end < len(beats):
        t_end = beats[beat_end]
    else:
        t_end = len(y) / sr
    dance_schedule.append((t_start, t_end, action_name))

print("\n📋 舞蹈时间表：")
for i, (t_start, t_end, name) in enumerate(dance_schedule):
    print(f"  [{i+1:2d}] {t_start:6.2f}s ~ {t_end:6.2f}s | {name}")
print("=" * 60)

# ===== 3. 加载 MuJoCo =====
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
    """根据音乐时间 t，找到当前动作"""
    for t_start, t_end, name in dance_schedule:
        if t_start <= t < t_end:
            return name, t - t_start, t_end - t_start
    return None, None, None

step = 0
last_action = None
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001   # 仿真时间 = 音乐时间

        name, t_local, dur = get_current_action(t)

        if name is None:
            target = STAND.copy()
            data.xfrc_applied[1] = [0, 0, 0, 0, 0, 0]
        else:
            if name != last_action:
                print(f"🎬 t={t:.2f}s → {name}")
                last_action = name
            target = ACTIONS[name](t_local, dur)

            # 跳跃力
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
            print(f"  t={t:.1f}s | 动作: {(name or '结束')}")

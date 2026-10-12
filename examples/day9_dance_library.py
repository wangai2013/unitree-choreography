"""
Day 9: 舞蹈库系统
- 从 JSON 加载舞蹈
- 自动计算时间轴（BPM + 拍数）
- 支持参数切换（不同舞蹈）
- 可选：播放对应音乐
"""
import mujoco
import mujoco.viewer
import numpy as np
import time
import json
import sys
import os
from robot_utils import build_ctrl_to_qpos_map, pd_control
from actions import ACTIONS, STAND

# ===== 1. 加载舞蹈 =====
DANCE_FILE = sys.argv[1] if len(sys.argv) > 1 else "dances/medium.json"

with open(DANCE_FILE, "r", encoding="utf-8") as f:
    dance = json.load(f)

print("=" * 60)
print(f"🎬 舞蹈: {dance['name']}")
print(f"📝 {dance.get('description', '')}")
print(f"🥁 BPM: {dance['bpm']}")
print(f"🎵 音乐: {dance.get('music', '无')}")
print("=" * 60)

# ===== 2. 计算时间轴 =====
BPM = dance["bpm"]
BEAT_INTERVAL = 60.0 / BPM   # 每拍秒数

timeline = []
for seg in dance["sequence"]:
    name = seg["action"]
    beats = seg["beats"]
    duration = beats * BEAT_INTERVAL
    timeline.append((name, duration))

total_duration = sum(d for _, d in timeline)
print(f"⏱  总时长: {total_duration:.2f} 秒（{sum(s['beats'] for s in dance['sequence'])} 拍）")
print("-" * 60)
for i, (name, dur) in enumerate(timeline):
    print(f"  [{i+1:2d}] {name:8s} — {dur:.2f} 秒")
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

def get_current_segment(t):
    acc = 0.0
    for name, dur in timeline:
        if t < acc + dur:
            return name, t - acc, dur
        acc += dur
    return None, None, None

# ===== 4. 主循环 =====
step = 0
last_action = None
last_beat = -1
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001

        name, t_local, dur = get_current_segment(t)

        # 打印节拍
        beat = int(t / BEAT_INTERVAL)
        if beat != last_beat:
            print(f"🥁 拍 {beat:3d} (t={t:.2f}s) | {name or '结束'}")
            last_beat = beat

        if name is None:
            target = STAND.copy()
            data.xfrc_applied[1] = [0, 0, 0, 0, 0, 0]
        else:
            if name != last_action:
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

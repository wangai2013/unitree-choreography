"""
Day 8: 节拍驱动的舞蹈
每个动作持续 N 拍
"""
import mujoco
import mujoco.viewer
import numpy as np
import time
from robot_utils import build_ctrl_to_qpos_map, pd_control
from actions import ACTIONS, STAND

# ===== 节拍设置 =====
BPM = 120
BEAT_INTERVAL = 60.0 / BPM   # 每拍秒数
BEATS_PER_ACTION = 4         # 每个动作持续 4 拍

# ===== 舞蹈序列（按拍数） =====
DANCE_BY_BEATS = [
    ("stand", 2),   # 站立 2 拍
    ("nod",   4),   # 点头 4 拍
    ("sway",  4),   # 摇摆 4 拍
    ("bow",   4),   # 鞠躬 4 拍
    ("squat", 4),   # 下蹲 4 拍
    ("jump",  4),   # 跳跃 4 拍
    ("nod",   4),   # 点头 4 拍
    ("stand", 2),   # 站定 2 拍
]

# 转换成时间序列
timeline = [(name, beats * BEAT_INTERVAL) for name, beats in DANCE_BY_BEATS]
total_duration = sum(d for _, d in timeline)

print("=" * 60)
print(f"🎵 节拍驱动舞蹈")
print(f"🥁 BPM: {BPM}（每拍 {BEAT_INTERVAL:.3f} 秒）")
print(f"⏱  总时长: {total_duration:.2f} 秒（{sum(b for _, b in DANCE_BY_BEATS)} 拍）")
print("-" * 60)
for i, (name, dur) in enumerate(timeline):
    print(f"  [{i+1}] {name:8s} — {dur:.2f} 秒")
print("=" * 60)

# ===== 加载模型 =====
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

def get_beat_number(t):
    """当前是第几拍"""
    return int(t / BEAT_INTERVAL)

step = 0
last_action = None
last_beat = -1
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001

        name, t_local, dur = get_current_segment(t)

        # ⭐ 每拍打印一次（视觉上跟节拍对齐）
        beat = get_beat_number(t)
        if beat != last_beat:
            print(f"🥁 节拍 {beat:3d} (t={t:.2f}s) | 动作: {name or '结束'}")
            last_beat = beat

        if name is None:
            target = STAND.copy()
            data.xfrc_applied[1] = [0, 0, 0, 0, 0, 0]
        else:
            if name != last_action:
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

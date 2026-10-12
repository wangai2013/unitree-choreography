"""
Day 6: 时间轴调度器
"""
import mujoco
import mujoco.viewer
import numpy as np
import time
from robot_utils import build_ctrl_to_qpos_map, pd_control
from actions import ACTIONS, STAND

MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

CTRL_TO_QPOS, CTRL_TO_QVEL = build_ctrl_to_qpos_map(model)

# 时间轴
TIMELINE = [
    ("stand",  1.0),
    ("nod",    2.0),
    ("sway",   2.0),
    ("wave",   4.0),
    ("stand",  1.0),
]

# 初始化
for i in range(12):
    data.qpos[CTRL_TO_QPOS[i]] = STAND[i]
data.qpos[2] = 0.3
mujoco.mj_forward(model, data)

KP, KD = 300.0, 15.0

total_duration = sum(d for _, d in TIMELINE)
print("=" * 60)
print(f"🎬 舞蹈序列（总时长 {total_duration:.1f} 秒）")
for i, (name, dur) in enumerate(TIMELINE):
    print(f"  [{i+1}] {name:8s} — {dur:.1f} 秒")
print("=" * 60)

def get_current_action(t):
    acc = 0.0
    for name, dur in TIMELINE:
        if t < acc + dur:
            return name, t - acc, dur
        acc += dur
    return None, None, None

step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001

        name, t_local, dur = get_current_action(t)

        if name is None:
            target = STAND.copy()
        else:
            target = ACTIONS[name](t_local, dur)

        pd_control(model, data, CTRL_TO_QPOS, CTRL_TO_QVEL, target, KP, KD)

        mujoco.mj_step(model, data)
        viewer.sync()

        step += 1
        time.sleep(0.001)

        if step % 500 == 0:
            print(f"t={t:.1f}s | 动作: {(name or '已结束'):8s} | 高度 {data.qpos[2]:.3f}")

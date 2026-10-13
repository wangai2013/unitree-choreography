"""
Day 28: G1 编舞播放器（JSON 驱动）
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import mujoco
import mujoco.viewer
import numpy as np
import time
import json
from g1_actions import ACTIONS, STAND

# 加载舞蹈
DANCE_FILE = sys.argv[1] if len(sys.argv) > 1 else "dances/g1_greet.json"
with open(DANCE_FILE, "r", encoding="utf-8") as f:
    dance = json.load(f)

print("=" * 60)
print(f"🎬 {dance['name']}")
print(f"📝 {dance['description']}")
print("-" * 60)
for i, seg in enumerate(dance["sequence"]):
    print(f"  [{i+1}] {seg['action']:12s} | {seg['duration']:.1f}s | {seg.get('desc','')}")
print(f"  总时长: {sum(s['duration'] for s in dance['sequence']):.1f}s")
print("=" * 60)

# 模型
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

# 时间轴
timeline = []
acc = 0.0
for seg in dance["sequence"]:
    timeline.append((seg["action"], seg.get("desc",""), acc, acc + seg["duration"]))
    acc += seg["duration"]

def get_segment(t):
    for action, desc, t0, t1 in timeline:
        if t0 <= t < t1:
            return action, desc, t - t0, t1 - t0
    return None, None, None, None

step = 0
last_action = None
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001
        target = STAND.copy()

        action, desc, t_local, dur = get_segment(t)

        if action is not None and action != last_action:
            print(f"🎬 t={t:.2f}s → {action} ({desc})")
            last_action = action

        if action is not None:
            target = ACTIONS[action](t_local, dur)

        for i in range(model.nu):
            q = data.qpos[CTRL_TO_QPOS[i]]
            dq = data.qvel[CTRL_TO_QVEL[i]]
            data.ctrl[i] = KP * (target[i] - q) - KD * dq

        mujoco.mj_step(model, data)

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

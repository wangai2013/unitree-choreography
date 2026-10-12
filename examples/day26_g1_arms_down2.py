"""
Day 26: G1 手臂放下 v2
- 重点测"shoulder_roll 负值"（往下）
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path("unitree_mujoco/unitree_robots/g1/scene_23dof.xml")
data = mujoco.MjData(model)

CTRL_TO_QPOS = np.zeros(model.nu, dtype=int)
CTRL_TO_QVEL = np.zeros(model.nu, dtype=int)
for i in range(model.nu):
    jid = model.actuator_trnid[i][0]
    CTRL_TO_QPOS[i] = model.jnt_qposadr[jid]
    CTRL_TO_QVEL[i] = model.jnt_dofadr[jid]

STAND = np.zeros(model.nu)
STAND[0] = -0.2; STAND[3] = 0.4; STAND[4] = -0.2
STAND[6] = -0.2; STAND[9] = 0.4; STAND[10] = -0.2

for i in range(model.nu):
    data.qpos[CTRL_TO_QPOS[i]] = STAND[i]

data.qpos[2] = 0.8
data.qpos[3] = 1.0
mujoco.mj_forward(model, data)

KP = 300.0
KD = 10.0

print("=" * 60)
print("🎬 G1 手臂放下 v2（重点测 roll 负值）")
print("  0~3s:  A（roll=-0.5）")
print("  3~6s:  B（roll=-1.0）")
print("  6~9s:  C（roll=-1.5）")
print("  9~12s: D（roll=-1.5, pitch=0.0, elbow=0.0）")
print("=" * 60)

step = 0
last = None
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001
        target = STAND.copy()

        if t < 3:
            group = "A"
            rp, rr, re = 0.0, -0.5, 0.3
        elif t < 6:
            group = "B"
            rp, rr, re = 0.0, -1.0, 0.3
        elif t < 9:
            group = "C"
            rp, rr, re = 0.0, -1.5, 0.3
        else:
            group = "D"
            rp, rr, re = 0.0, -1.5, 0.0

        target[22] = rp
        target[23] = rr
        target[25] = re
        # 左臂同样
        target[15] = rp
        target[16] = -rr  # 左臂 roll 反号
        target[18] = re

        if group != last:
            print(f"🎬 t={t:.1f}s → {group}: pitch={rp:.2f}, roll={rr:.2f}, elbow={re:.2f}")
            last = group

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

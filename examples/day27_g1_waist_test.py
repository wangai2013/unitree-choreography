"""
Day 27: G1 腰部关节测试
- waist_yaw（12）: 左右转
- waist_roll（13）: 左右倾
- waist_pitch（14）: 前后弯
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

RP, RR, RE = 0.4, -0.1, 0.8
STAND[15] = RP; STAND[16] = -RR; STAND[18] = RE
STAND[22] = RP; STAND[23] = RR;  STAND[25] = RE

for i in range(model.nu):
    data.qpos[CTRL_TO_QPOS[i]] = STAND[i]

data.qpos[2] = 0.8
data.qpos[3] = 1.0
mujoco.mj_forward(model, data)

KP = 300.0
KD = 10.0

print("=" * 60)
print("🎬 G1 腰部测试")
print("  0~3s:  waist_yaw = +0.5（左右转）")
print("  3~6s:  waist_roll = +0.5（左右倾）")
print("  6~9s:  waist_pitch = +0.5（前后弯）")
print("=" * 60)

step = 0
last = None
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001
        target = STAND.copy()

        if t < 3:
            group = "yaw"
            target[12] = 0.5   # waist_yaw
        elif t < 6:
            group = "roll"
            target[13] = 0.5   # waist_roll
        else:
            group = "pitch"
            target[14] = 0.5   # waist_pitch

        if group != last:
            print(f"🎬 t={t:.1f}s → waist_{group}=+0.5")
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

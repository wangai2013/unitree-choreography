"""
Day 33: 4 台 G1 可视化
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path("unitree_mujoco/unitree_robots/g1/scene_quad_g1.xml")
data = mujoco.MjData(model)

print(f"✅ nq={model.nq}, nu={model.nu}")

FREE_JOINTS = [i for i in range(model.njnt) if model.jnt_type[i] == 0]
QPOS_BASES = [model.jnt_qposadr[j] for j in FREE_JOINTS]
print(f"✅ freejoint: {len(FREE_JOINTS)}")

N = 4
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

STAND = np.zeros(29)
STAND[0] = -0.2; STAND[3] = 0.4; STAND[4] = -0.2
STAND[6] = -0.2; STAND[9] = 0.4; STAND[10] = -0.2
STAND[15] = 0.2; STAND[16] = -0.1; STAND[18] = 0.8
STAND[22] = 0.2; STAND[23] = -0.1; STAND[25] = 0.8

positions = [(-1.5, -0.5), (1.5, -0.5), (-1.5, 0.8), (1.5, 0.8)]

for r in range(N):
    for i in range(29):
        data.qpos[CTRL_TO_QPOS[r][i]] = STAND[i]
    data.qpos[QPOS_BASES[r] + 0] = positions[r][0]
    data.qpos[QPOS_BASES[r] + 1] = positions[r][1]
    data.qpos[QPOS_BASES[r] + 2] = 0.8
    data.qpos[QPOS_BASES[r] + 3] = 1.0

mujoco.mj_forward(model, data)

KP, KD = 300.0, 10.0

print("🎬 4 台 G1 站立")

step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        for r in range(N):
            for i in range(29):
                q = data.qpos[CTRL_TO_QPOS[r][i]]
                dq = data.qvel[CTRL_TO_QVEL[r][i]]
                data.ctrl[r * 29 + i] = KP * (STAND[i] - q) - KD * dq

        mujoco.mj_step(model, data)

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

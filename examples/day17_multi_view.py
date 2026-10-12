"""
Day 17: 10 台 Go2 可视化
"""
import mujoco
import mujoco.viewer
import numpy as np
import time

N = 10
MODEL_PATH = f"unitree_mujoco/unitree_robots/go2/scene_multi_{N}.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print(f"✅ 加载: nq={model.nq}, nu={model.nu}, nbody={model.nbody}")

# 映射
CTRL_TO_QPOS = []
CTRL_TO_QVEL = []
for r in range(N):
    qpos_r = np.zeros(12, dtype=int)
    qvel_r = np.zeros(12, dtype=int)
    for i in range(12):
        jid = model.actuator_trnid[r * 12 + i][0]
        qpos_r[i] = model.jnt_qposadr[jid]
        qvel_r[i] = model.jnt_dofadr[jid]
    CTRL_TO_QPOS.append(qpos_r)
    CTRL_TO_QVEL.append(qvel_r)

STAND = np.array([0.0, 0.8, -1.5] * 4)
for r in range(N):
    for i in range(12):
        data.qpos[CTRL_TO_QPOS[r][i]] = STAND[i]
mujoco.mj_forward(model, data)

KP, KD = 300.0, 15.0

step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        for r in range(N):
            for i in range(12):
                q = data.qpos[CTRL_TO_QPOS[r][i]]
                dq = data.qvel[CTRL_TO_QVEL[r][i]]
                data.ctrl[r * 12 + i] = KP * (STAND[i] - q) - KD * dq

        mujoco.mj_step(model, data)
        viewer.sync()
        step += 1
        time.sleep(0.001)

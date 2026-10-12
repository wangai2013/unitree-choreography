"""
Day 12: 4 台 Go2 可视化（都站立）
"""
import mujoco
import mujoco.viewer
import numpy as np
import time

MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene_quad_final.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print(f"✅ 加载: nq={model.nq}, nu={model.nu}, nbody={model.nbody}")

# 为 4 台机器人分别建立映射
# 每台 12 个电机：A[0-11], B[12-23], C[24-35], D[36-47]
N_ROBOTS = 4
CTRL_TO_QPOS = []   # 4×12
CTRL_TO_QVEL = []

for r in range(N_ROBOTS):
    qpos_r = np.zeros(12, dtype=int)
    qvel_r = np.zeros(12, dtype=int)
    for i in range(12):
        ctrl_idx = r * 12 + i
        jid = model.actuator_trnid[ctrl_idx][0]
        qpos_r[i] = model.jnt_qposadr[jid]
        qvel_r[i] = model.jnt_dofadr[jid]
    CTRL_TO_QPOS.append(qpos_r)
    CTRL_TO_QVEL.append(qvel_r)

print(f"✅ 4 台映射已建立")
for r in range(N_ROBOTS):
    print(f"  机器人{chr(65+r)}: qpos={CTRL_TO_QPOS[r]}")

STAND = np.array([0.0, 0.8, -1.5] * 4)

# 初始化 4 台站立姿态
for r in range(N_ROBOTS):
    for i in range(12):
        data.qpos[CTRL_TO_QPOS[r][i]] = STAND[i]
mujoco.mj_forward(model, data)

KP, KD = 300.0, 15.0

step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        for r in range(N_ROBOTS):
            for i in range(12):
                q = data.qpos[CTRL_TO_QPOS[r][i]]
                dq = data.qvel[CTRL_TO_QVEL[r][i]]
                data.ctrl[r * 12 + i] = KP * (STAND[i] - q) - KD * dq

        mujoco.mj_step(model, data)
        viewer.sync()
        step += 1
        time.sleep(0.001)

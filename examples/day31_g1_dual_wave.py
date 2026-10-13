"""
Day 31: 2 台 G1 同步挥手 + 鞠躬
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import mujoco
import mujoco.viewer
import numpy as np
import time
from g1_actions import ACTIONS, STAND

model = mujoco.MjModel.from_xml_path("unitree_mujoco/unitree_robots/g1/scene_dual_g1.xml")
data = mujoco.MjData(model)

print(f"✅ 加载: nq={model.nq}, nu={model.nu}")

FREE_JOINTS = [i for i in range(model.njnt) if model.jnt_type[i] == 0]
QPOS_BASES = [model.jnt_qposadr[j] for j in FREE_JOINTS]
print(f"✅ freejoint: {len(FREE_JOINTS)} 个")

N = 2
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

for r in range(N):
    for i in range(29):
        data.qpos[CTRL_TO_QPOS[r][i]] = STAND[i]

data.qpos[QPOS_BASES[0] + 0] = -1.0
data.qpos[QPOS_BASES[0] + 2] = 0.8
data.qpos[QPOS_BASES[0] + 3] = 1.0

data.qpos[QPOS_BASES[1] + 0] = 1.0
data.qpos[QPOS_BASES[1] + 2] = 0.8
data.qpos[QPOS_BASES[1] + 3] = 1.0

mujoco.mj_forward(model, data)

KP, KD = 300.0, 10.0

print("=" * 60)
print("🎬 2 台 G1 同步编舞")
print("  0~2s:  站好")
print("  2~5s:  同步挥手")
print("  5~8s:  同步鞠躬")
print("  8s+:   站好")
print("=" * 60)

step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001

        # ⭐ 2 台"同步"：用同一个 target
        target = STAND.copy()

        if t < 2.0:
            pass
        elif t < 5.0:
            target = ACTIONS["wave_hand"](t - 2.0, 3.0)
        elif t < 8.0:
            target = ACTIONS["bow"](t - 5.0, 3.0)
        else:
            target = STAND.copy()

        # 应用
        for r in range(N):
            for i in range(29):
                q = data.qpos[CTRL_TO_QPOS[r][i]]
                dq = data.qvel[CTRL_TO_QVEL[r][i]]
                data.ctrl[r * 29 + i] = KP * (target[i] - q) - KD * dq

        mujoco.mj_step(model, data)

        # 锁位置
        for r in range(N):
            data.qpos[QPOS_BASES[r] + 0] = -1.0 + r * 2.0
            data.qpos[QPOS_BASES[r] + 1] = 0.0
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

"""
Day 26: G1 挥手 v9（肩抬 10 度）
- 手抬起时，target[23] 加 10 度
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

# ⭐ 肩抬参数
ROLL_BASE = 1.2
ROLL_EXTRA_DEG = 10
ROLL_EXTRA_RAD = np.deg2rad(ROLL_EXTRA_DEG)
ROLL_TARGET = -0.5

print("=" * 60)
print(f"🎬 G1 挥手 v9")
print(f"  肩抬参数：base={ROLL_BASE}, 加 {ROLL_EXTRA_DEG}° = {ROLL_TARGET:.4f} rad")
print("  0~2s: 自然下垂")
print("  2~3s: 抬右臂（到 1.3745）")
print("  3~4s: 小臂旋转")
print("  4s+:  大臂 yaw 摆动")
print("=" * 60)

step = 0
last_target_25 = 0.0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.002
        
        target = STAND.copy()

        if t < 2.0:
            pass
        elif t < 3.0:
            p = (t - 2.0) / 1.0
            val_25 = 0.5 + (-1.2 - 0.8) * p
            target[22] = 0.3 + (0.0 - 0.4) * p
            target[23] = -0.1 + (ROLL_TARGET - (-0.1)) * p
            target[25] = val_25
            target[26] = -1.57 * p
            last_target_25 = val_25
        else:
            target[22] = -0.1
            target[23] = ROLL_TARGET
            target[25] = last_target_25
            target[26] = -1.57

            t_wave = t - 4.0
            wave = np.sin(2 * np.pi * 1.0 * t_wave)

            target[24] = 0.7 * wave + 0.5

        if step % 200 == 0:
            print(f"t={t:5.2f}s | roll={target[23]:.4f} | yaw={target[24]:+.2f}")

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

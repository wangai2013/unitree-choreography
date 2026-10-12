"""
Day 13: 4 台 Go2 走到目标位置（路径规划 + 步态）
"""
import mujoco
import mujoco.viewer
import numpy as np
import time

MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene_quad_final.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# 找 freejoint
FREE_JOINTS = [i for i in range(model.njnt) if model.jnt_type[i] == 0]
QPOS_BASES = [model.jnt_qposadr[j] for j in FREE_JOINTS]

# 电机映射
N = 4
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
KP_POS, KD_POS = 50.0, 5.0   # 位置 PD

# 目标位置（4 台走成"菱形"）
TARGETS = [
    (0.0, -1.2),   # A
    (1.2, 0.0),    # B
    (-1.2, 0.0),   # C
    (0.0, 1.2),    # D
]

STEP_FREQ = 2.0
STEP_LIFT = 0.3
STEP_SWING = 0.25

print("=" * 60)
print("🎬 4 台 Go2 走到目标位置（菱形）")
print("=" * 60)

step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001
        phase = 2 * np.pi * STEP_FREQ * t

        for r in range(N):
            # ===== 位置 PD：计算机身需要的 x/y 速度 =====
            cur_x = data.qpos[QPOS_BASES[r] + 0]
            cur_y = data.qpos[QPOS_BASES[r] + 1]
            tgt_x, tgt_y = TARGETS[r]

            # 用速度控制（v = Kp * (目标 - 当前)）
            vx = KP_POS * (tgt_x - cur_x)
            vy = KP_POS * (tgt_y - cur_y)

            # 限制速度
            v_max = 0.8
            vx = np.clip(vx, -v_max, v_max)
            vy = np.clip(vy, -v_max, v_max)

            # ⭐ 直接设置机身速度（简化版"走路"）
            data.qvel[QPOS_BASES[r] - 0 + 0] = vx    # vx
            data.qvel[QPOS_BASES[r] - 0 + 1] = vy    # vy

            # ===== 步态（腿摆动）=====
            target_leg = STAND.copy()
            phase_A = np.sin(phase)
            phase_B = np.sin(phase + np.pi)

            # FL + RR
            target_leg[4] = 0.8 - STEP_LIFT * max(0, phase_A)
            target_leg[5] = -1.5 + STEP_SWING * phase_A
            target_leg[7] = 0.8 - STEP_LIFT * max(0, phase_A)
            target_leg[8] = -1.5 + STEP_SWING * phase_A

            # FR + RL
            target_leg[1] = 0.8 - STEP_LIFT * max(0, phase_B)
            target_leg[2] = -1.5 + STEP_SWING * phase_B
            target_leg[10] = 0.8 - STEP_LIFT * max(0, phase_B)
            target_leg[11] = -1.5 + STEP_SWING * phase_B

            # PD 控制关节
            for i in range(12):
                q = data.qpos[CTRL_TO_QPOS[r][i]]
                dq = data.qvel[CTRL_TO_QVEL[r][i]]
                data.ctrl[r * 12 + i] = KP * (target_leg[i] - q) - KD * dq

        mujoco.mj_step(model, data)
        viewer.sync()
        step += 1
        time.sleep(0.001)

        if step % 1000 == 0:
            print(f"t={t:.1f}s:")
            for r in range(N):
                x = data.qpos[QPOS_BASES[r] + 0]
                y = data.qpos[QPOS_BASES[r] + 1]
                print(f"  {chr(65+r)}: ({x:+.2f}, {y:+.2f}) → 目标 TARGETS[{r}]")

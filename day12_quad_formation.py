"""
Day 12: 4 台队形变换（方阵 ⇄ 菱形 ⇄ 一字）
"""
import mujoco
import mujoco.viewer
import numpy as np
import time

MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene_quad_final.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# 找 4 个 freejoint
FREE_JOINTS = [i for i in range(model.njnt) if model.jnt_type[i] == 0]
QPOS_BASES = [model.jnt_qposadr[j] for j in FREE_JOINTS]
print(f"✅ 4 台 qpos 起始: {QPOS_BASES}")

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
HEIGHT = 0.32

# 队形定义（4 台位置）
FORMATIONS = [
    # 方阵
    [(-1.2, -0.6), (1.2, -0.6), (-1.2, 0.6), (1.2, 0.6)],
    # 菱形
    [(0.0, -1.2), (1.2, 0.0), (-1.2, 0.0), (0.0, 1.2)],
    # 一字
    [(-1.8, 0.0), (-0.6, 0.0), (0.6, 0.0), (1.8, 0.0)],
    # 圆形
    [(1.2, 0.0), (0.0, 1.2), (-1.2, 0.0), (0.0, -1.2)],
]
NAMES = ["方阵", "菱形", "一字", "圆形"]
DURATION_PER_FORMATION = 3.0

print("=" * 60)
print("🎬 4 台队形变换")
for i, name in enumerate(NAMES):
    print(f"  [{i+1}] {name}")
print("=" * 60)

def smoothstep(x):
    x = np.clip(x, 0, 1)
    return 3*x**2 - 2*x**3

step = 0
last_idx = -1
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001
        idx = int(t / DURATION_PER_FORMATION)
        if idx >= len(FORMATIONS):
            idx = len(FORMATIONS) - 1

        # 过渡进度（每段前 0.5 秒过渡）
        t_in = t - idx * DURATION_PER_FORMATION
        p = smoothstep(t_in / 0.5)

        # 计算当前目标位置
        if idx == 0:
            prev = FORMATIONS[0]
        else:
            prev = FORMATIONS[idx - 1]
        cur = FORMATIONS[idx]

        if idx != last_idx:
            print(f"🎬 t={t:.1f}s → {NAMES[idx]}")
            last_idx = idx

        for r in range(N):
            target_x = prev[r][0] + (cur[r][0] - prev[r][0]) * p
            target_y = prev[r][1] + (cur[r][1] - prev[r][1]) * p

            # 直接设置机身位置
            data.qpos[QPOS_BASES[r] + 0] = target_x
            data.qpos[QPOS_BASES[r] + 1] = target_y
            data.qpos[QPOS_BASES[r] + 2] = HEIGHT

            # 清速度
            for k in range(6):
                data.qvel[QPOS_BASES[r] - 0 + k] = 0

        # PD 控制关节
        for r in range(N):
            for i in range(12):
                q = data.qpos[CTRL_TO_QPOS[r][i]]
                dq = data.qvel[CTRL_TO_QVEL[r][i]]
                data.ctrl[r * 12 + i] = KP * (STAND[i] - q) - KD * dq

        mujoco.mj_step(model, data)
        viewer.sync()
        step += 1
        time.sleep(0.001)

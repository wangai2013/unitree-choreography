"""
Day 17: 10 台群控舞蹈（纯动画化版）
核心：直接设置 qpos（机身 + 关节），不用 PD，不用 mj_step
"""
import mujoco
import mujoco.viewer
import numpy as np
import time
import os

if os.path.basename(os.getcwd()) == "examples":
    os.chdir("..")

N = 10
MODEL_PATH = f"unitree_mujoco/unitree_robots/go2/scene_multi_{N}.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print(f"✅ 加载 {N} 台: nq={model.nq}, nu={model.nu}")

# 映射
CTRL_TO_QPOS = []
for r in range(N):
    qpos_r = np.zeros(12, dtype=int)
    for i in range(12):
        jid = model.actuator_trnid[r * 12 + i][0]
        qpos_r[i] = model.jnt_qposadr[jid]
    CTRL_TO_QPOS.append(qpos_r)

# 自由关节（机身）
FREE_JOINTS = [i for i in range(model.njnt) if model.jnt_type[i] == 0]
QPOS_BASES = [model.jnt_qposadr[j] for j in FREE_JOINTS]

# 5×2 站位
COLS = 5
SPACING_X = 1.2
SPACING_Y = 0.8
HEIGHT = 0.32

POSITIONS = []
for r in range(N):
    row = r // COLS
    col = r % COLS
    x = (col - (COLS - 1) / 2) * SPACING_X
    y = (row - 0.5) * SPACING_Y
    POSITIONS.append((x, y))

STAND = np.array([0.0, 0.8, -1.5] * 4)

# 时间轴
DANCE = [
    ("stand",    3.0),
    ("sync_nod", 4.0),
    ("wave_nod", 6.0),
    ("mirror",   4.0),
    ("sync_nod", 4.0),
    ("stand",    2.0),
]
total = sum(d for _, d in DANCE)

def get_segment(t):
    acc = 0.0
    for action, dur in DANCE:
        if t < acc + dur:
            return action, t - acc, dur
        acc += dur
    return None, None, None

print("=" * 60)
print(f"🎬 {N} 台群控舞蹈（纯动画化）")
print("=" * 60)

step = 0
last_action = None

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001

        targets = [STAND.copy() for _ in range(N)]

        action, t_local, dur = get_segment(t)

        if action != last_action and action is not None:
            print(f"🎬 {t:.1f}s → {action}")
            last_action = action

        # ===== 动作定义（生成目标关节角度）=====
        if action == "sync_nod":
            swing = 0.4 * np.sin(2 * np.pi * 1.5 * t_local)
            for r in range(N):
                targets[r][2] = -1.5 + swing
                targets[r][5] = -1.5 + swing

        elif action == "wave_nod":
            for r in range(N):
                t_r = max(0, t_local - r * 0.15)
                swing = 0.4 * np.sin(2 * np.pi * 1.5 * t_r)
                targets[r][2] = -1.5 + swing
                targets[r][5] = -1.5 + swing

        elif action == "mirror":
            sway = 0.2 * np.sin(2 * np.pi * 1.0 * t_local)
            for r in range(N):
                col = r % COLS
                sign = +1 if (col % 2) == 0 else -1
                targets[r][1] = 0.8 + sign * sway
                targets[r][4] = 0.8 - sign * sway

        # ===== ⭐ 直接设置 qpos（跳过物理）=====
        for r in range(N):
            # 机身位置
            data.qpos[QPOS_BASES[r] + 0] = POSITIONS[r][0]
            data.qpos[QPOS_BASES[r] + 1] = POSITIONS[r][1]
            data.qpos[QPOS_BASES[r] + 2] = HEIGHT
            # 机身姿态（四元数，无旋转）
            data.qpos[QPOS_BASES[r] + 3] = 1.0   # w
            data.qpos[QPOS_BASES[r] + 4] = 0.0
            data.qpos[QPOS_BASES[r] + 5] = 0.0
            data.qpos[QPOS_BASES[r] + 6] = 0.0

            # 12 个关节
            for i in range(12):
                data.qpos[CTRL_TO_QPOS[r][i]] = targets[r][i]

        # 清空所有速度
        data.qvel[:] = 0

        # ⭐ 只用 mj_forward（不算物理）
        mujoco.mj_forward(model, data)

        viewer.sync()
        step += 1
        time.sleep(0.001)

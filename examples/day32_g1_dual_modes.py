"""
Day 32: 2 台 G1 群控 4 种模式
- 同步 / 错时 / 镜像 / 对角
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

# ========== 时间轴 ==========
TIMELINE = [
    ("sync",      3.0, "同步挥手"),
    ("stand",     1.0, "停顿"),
    ("delay",     4.0, "错时挥手"),
    ("stand",     1.0, "停顿"),
    ("mirror",    4.0, "镜像（A 挥手 B 鞠躬）"),
    ("stand",     1.0, "停顿"),
    ("diagonal",  4.0, "对角（A 左 B 右）"),
    ("stand",     2.0, "站好"),
]

total = sum(d for _, d, _ in TIMELINE)
print("=" * 60)
print(f"🎬 2 台 G1 群控（4 种模式）")
for name, dur, desc in TIMELINE:
    print(f"  {name:10s} | {dur:.1f}s | {desc}")
print(f"  总时长: {total:.1f}s")
print("=" * 60)

def get_mode(t):
    acc = 0.0
    for mode, dur, desc in TIMELINE:
        if t < acc + dur:
            return mode, t - acc, dur
        acc += dur
    return None, None, None

def compute_targets(mode, t_local, dur):
    """返回 (target_A, target_B)"""
    if mode == "stand" or mode is None:
        return STAND.copy(), STAND.copy()

    if mode == "sync":
        # 同步
        target = ACTIONS["wave_hand"](t_local, dur)
        return target.copy(), target.copy()

    elif mode == "delay":
        # 错时（B 延迟 0.5s）
        target_A = ACTIONS["wave_hand"](t_local, dur)
        target_B = ACTIONS["wave_hand"](max(0, t_local - 0.5), dur)
        return target_A, target_B

    elif mode == "mirror":
        # 镜像（A 挥手，B 鞠躬）
        target_A = ACTIONS["wave_hand"](t_local, dur)
        target_B = ACTIONS["bow"](t_local, dur)
        return target_A, target_B

    elif mode == "diagonal":
        # 对角（A 左转，B 右转）
        target_A = STAND.copy()
        target_B = STAND.copy()
        phase = t_local / dur
        s = np.sin(np.pi * phase)
        # A 左转
        target_A[12] = 0.7 * s
        # B 右转
        target_B[12] = -0.7 * s
        return target_A, target_B

    return STAND.copy(), STAND.copy()

step = 0
last_mode = None
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001

        mode, t_local, dur = get_mode(t)
        if mode is not None and mode != last_mode:
            print(f"🎬 t={t:.2f}s → {mode}")
            last_mode = mode

        target_A, target_B = compute_targets(mode, t_local, dur)

        # A
        for i in range(29):
            q = data.qpos[CTRL_TO_QPOS[0][i]]
            dq = data.qvel[CTRL_TO_QVEL[0][i]]
            data.ctrl[0 * 29 + i] = KP * (target_A[i] - q) - KD * dq

        # B
        for i in range(29):
            q = data.qpos[CTRL_TO_QPOS[1][i]]
            dq = data.qvel[CTRL_TO_QVEL[1][i]]
            data.ctrl[1 * 29 + i] = KP * (target_B[i] - q) - KD * dq

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

"""
Day 36: 10 台 G1 阅兵式
- 同步 / 波浪
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import mujoco
import mujoco.viewer
import numpy as np
import time
from g1_actions import ACTIONS, STAND

model = mujoco.MjModel.from_xml_path("unitree_mujoco/unitree_robots/g1/scene_deca_g1.xml")
data = mujoco.MjData(model)

print(f"✅ 加载: nq={model.nq}, nu={model.nu}")

FREE_JOINTS = [i for i in range(model.njnt) if model.jnt_type[i] == 0]
QPOS_BASES = [model.jnt_qposadr[j] for j in FREE_JOINTS]
print(f"✅ freejoint: {len(FREE_JOINTS)}")

N = 10
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

# 10 台位置（5×2）
COLS = 5
SPACING_X = 1.5
SPACING_Y = 1.5
positions = []
for i in range(N):
    row = i // COLS
    col = i % COLS
    x = (col - (COLS - 1) / 2) * SPACING_X
    y = (row - 0.5) * SPACING_Y
    positions.append((x, y))

for r in range(N):
    for i in range(29):
        data.qpos[CTRL_TO_QPOS[r][i]] = STAND[i]
    data.qpos[QPOS_BASES[r] + 0] = positions[r][0]
    data.qpos[QPOS_BASES[r] + 1] = positions[r][1]
    data.qpos[QPOS_BASES[r] + 2] = 0.8
    data.qpos[QPOS_BASES[r] + 3] = 1.0

mujoco.mj_forward(model, data)

KP, KD = 300.0, 10.0

# ========== 时间轴 ==========
TIMELINE = [
    ("stand",  2.0, "站好"),
    ("sync",   3.0, "同步挥手"),
    ("stand",  1.0, "停顿"),
    ("wave",   5.0, "波浪（依次）"),
    ("stand",  2.0, "站好"),
]

def get_mode(t):
    acc = 0.0
    for mode, dur, desc in TIMELINE:
        if t < acc + dur:
            return mode, t - acc, dur
        acc += dur
    return None, None, None

def compute_targets(mode, t_local, dur):
    targets = [STAND.copy() for _ in range(N)]
    if mode is None or mode == "stand":
        return targets
    if mode == "sync":
        t = ACTIONS["wave_hand"](t_local, dur)
        for r in range(N):
            targets[r] = t.copy()
    elif mode == "wave":
        for r in range(N):
            t_r = max(0, t_local - r * 0.2)
            targets[r] = ACTIONS["wave_hand"](t_r, dur).copy()
    return targets

step = 0
last_mode = None
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001
        mode, t_local, dur = get_mode(t)
        if mode is not None and mode != last_mode:
            print(f"🎬 t={t:.2f}s → {mode}")
            last_mode = mode

        targets = compute_targets(mode, t_local, dur)

        for r in range(N):
            for i in range(29):
                q = data.qpos[CTRL_TO_QPOS[r][i]]
                dq = data.qvel[CTRL_TO_QVEL[r][i]]
                data.ctrl[r * 29 + i] = KP * (targets[r][i] - q) - KD * dq

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

"""
Day 34: 4 台 G1 群控 4 种模式
- 同步 / 波浪 / 镜像 / 对角
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import mujoco
import mujoco.viewer
import numpy as np
import time
from g1_actions import ACTIONS, STAND

model = mujoco.MjModel.from_xml_path("unitree_mujoco/unitree_robots/g1/scene_quad_g1.xml")
data = mujoco.MjData(model)

print(f"✅ 加载: nq={model.nq}, nu={model.nu}")

FREE_JOINTS = [i for i in range(model.njnt) if model.jnt_type[i] == 0]
QPOS_BASES = [model.jnt_qposadr[j] for j in FREE_JOINTS]

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

# ========== 时间轴 ==========
TIMELINE = [
    ("stand",  1.5, "站好"),
    ("sync",   3.0, "同步挥手"),
    ("stand",  1.0, "停顿"),
    ("wave",   4.0, "波浪（依次）"),
    ("stand",  1.0, "停顿"),
    ("mirror", 4.0, "镜像（交替）"),
    ("stand",  1.0, "停顿"),
    ("diagonal", 4.0, "对角"),
    ("stand",  2.0, "站好"),
]

total = sum(d for _, d, _ in TIMELINE)
print("=" * 60)
print("🎬 4 台 G1 群控编舞")
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
    """返回 [target_A, target_B, target_C, target_D]"""
    targets = [STAND.copy() for _ in range(4)]

    if mode is None or mode == "stand":
        return targets

    if mode == "sync":
        # 同步挥手
        t = ACTIONS["wave_hand"](t_local, dur)
        for r in range(4):
            targets[r] = t.copy()

    elif mode == "wave":
        # 波浪（每台延迟 0.3s）
        for r in range(4):
            t_r = max(0, t_local - r * 0.3)
            targets[r] = ACTIONS["wave_hand"](t_r, dur).copy()

    elif mode == "mirror":
        # 镜像（A/C 挥手，B/D 鞠躬）
        t_wave = ACTIONS["wave_hand"](t_local, dur)
        t_bow = ACTIONS["bow"](t_local, dur)
        targets[0] = t_wave.copy()
        targets[1] = t_bow.copy()
        targets[2] = t_wave.copy()
        targets[3] = t_bow.copy()

    elif mode == "diagonal":
        # 对角（A/D 左转，B/C 右转）
        phase = t_local / dur
        s = np.sin(np.pi * phase)
        for r in range(4):
            if r in [0, 3]:
                targets[r][12] = 0.7 * s    # 左转
            else:
                targets[r][12] = -0.7 * s   # 右转

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

        # 4 台应用
        for r in range(N):
            for i in range(29):
                q = data.qpos[CTRL_TO_QPOS[r][i]]
                dq = data.qvel[CTRL_TO_QVEL[r][i]]
                data.ctrl[r * 29 + i] = KP * (targets[r][i] - q) - KD * dq

        mujoco.mj_step(model, data)

        # 锁位置
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

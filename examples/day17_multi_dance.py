"""
Day 17: 10 台群控舞蹈（规模化）
- 10 台 Go2 同步 / 波浪
- 自动排布矩阵队形
"""
import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import sys

# 路径修复
if os.path.basename(os.getcwd()) == "examples":
    os.chdir("..")

N = 10
MODEL_PATH = f"unitree_mujoco/unitree_robots/go2/scene_multi_{N}.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print(f"✅ 加载 {N} 台: nq={model.nq}, nu={model.nu}")

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

# 时间轴
DANCE = [
    ("stand",   3.0),
    ("sync_nod", 4.0),
    ("wave_nod", 6.0),   # 10 台波浪更慢
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
print(f"🎬 {N} 台群控舞蹈（总时长 {total}s）")
for i, (a, d) in enumerate(DANCE):
    print(f"  [{i+1}] {a:10s} — {d:.1f}s")
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

        if action == "sync_nod":
            swing = 0.4 * np.sin(2 * np.pi * 1.5 * t_local)
            for r in range(N):
                targets[r][2] = -1.5 + swing
                targets[r][5] = -1.5 + swing

        elif action == "wave_nod":
            # 10 台波浪：每台延迟 0.15 秒
            for r in range(N):
                t_r = max(0, t_local - r * 0.15)
                swing = 0.4 * np.sin(2 * np.pi * 1.5 * t_r)
                targets[r][2] = -1.5 + swing
                targets[r][5] = -1.5 + swing

        elif action == "mirror":
            sway = 0.2 * np.sin(2 * np.pi * 1.0 * t_local)
            for r in range(N):
                sign = +1 if (r % 2) == 0 else -1
                targets[r][1] = 0.8 + sign * sway
                targets[r][4] = 0.8 - sign * sway

        for r in range(N):
            for i in range(12):
                q = data.qpos[CTRL_TO_QPOS[r][i]]
                dq = data.qvel[CTRL_TO_QVEL[r][i]]
                data.ctrl[r * 12 + i] = KP * (targets[r][i] - q) - KD * dq

        mujoco.mj_step(model, data)
        viewer.sync()
        step += 1
        time.sleep(0.001)

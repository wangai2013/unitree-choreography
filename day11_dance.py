"""
Day 11-4: 双机群控舞蹈（同步 → 错时 → 镜像 循环）
"""
import mujoco
import mujoco.viewer
import numpy as np
import time

MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene_dual_final.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

CTRL_TO_QPOS_A = np.zeros(12, dtype=int)
CTRL_TO_QVEL_A = np.zeros(12, dtype=int)
CTRL_TO_QPOS_B = np.zeros(12, dtype=int)
CTRL_TO_QVEL_B = np.zeros(12, dtype=int)

for i in range(12):
    jid_A = model.actuator_trnid[i][0]
    CTRL_TO_QPOS_A[i] = model.jnt_qposadr[jid_A]
    CTRL_TO_QVEL_A[i] = model.jnt_dofadr[jid_A]
    jid_B = model.actuator_trnid[i + 12][0]
    CTRL_TO_QPOS_B[i] = model.jnt_qposadr[jid_B]
    CTRL_TO_QVEL_B[i] = model.jnt_dofadr[jid_B]

STAND = np.array([0.0, 0.8, -1.5] * 4)
for i in range(12):
    data.qpos[CTRL_TO_QPOS_A[i]] = STAND[i]
    data.qpos[CTRL_TO_QPOS_B[i]] = STAND[i]
mujoco.mj_forward(model, data)

KP, KD = 300.0, 15.0

# 群控时间轴
# (模式, 时长)
TIMELINE = [
    ("stand",   1.0),
    ("sync",    3.0),   # 同步点头
    ("delay",   3.0),   # 错时
    ("mirror",  3.0),   # 镜像
    ("sync",    3.0),   # 再同步
    ("stand",   1.0),
]
total = sum(d for _, d in TIMELINE)

print("=" * 60)
print("🎬 双机群控舞蹈")
for i, (mode, dur) in enumerate(TIMELINE):
    print(f"  [{i+1}] {mode:8s} — {dur:.1f}s")
print("=" * 60)

def get_mode(t):
    acc = 0.0
    for mode, dur in TIMELINE:
        if t < acc + dur:
            return mode, t - acc
        acc += dur
    return None, None

step = 0
last_mode = None
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001
        mode, t_local = get_mode(t)

        if mode != last_mode:
            print(f"🎬 t={t:.1f}s → {mode}")
            last_mode = mode

        if mode is None:
            target_A = STAND.copy()
            target_B = STAND.copy()
        elif mode == "stand":
            target_A = STAND.copy()
            target_B = STAND.copy()
        elif mode == "sync":
            # 同步点头
            target_A = STAND.copy()
            target_B = STAND.copy()
            swing = 0.4 * np.sin(2 * np.pi * 1.5 * t_local)
            target_A[2] = -1.5 + swing
            target_A[5] = -1.5 + swing
            target_B[2] = -1.5 + swing
            target_B[5] = -1.5 + swing
        elif mode == "delay":
            # 错时
            swing_A = 0.4 * np.sin(2 * np.pi * 1.5 * t_local)
            swing_B = 0.4 * np.sin(2 * np.pi * 1.5 * max(0, t_local - 0.5))
            target_A = STAND.copy()
            target_B = STAND.copy()
            target_A[2] = -1.5 + swing_A
            target_A[5] = -1.5 + swing_A
            target_B[2] = -1.5 + swing_B
            target_B[5] = -1.5 + swing_B
        elif mode == "mirror":
            # 镜像摇摆
            target_A = STAND.copy()
            target_B = STAND.copy()
            sway = 0.2 * np.sin(2 * np.pi * 1.0 * t_local)
            target_A[1] = 0.8 + sway
            target_A[4] = 0.8 - sway
            target_B[1] = 0.8 - sway
            target_B[4] = 0.8 + sway

        for i in range(12):
            q = data.qpos[CTRL_TO_QPOS_A[i]]
            dq = data.qvel[CTRL_TO_QVEL_A[i]]
            data.ctrl[i] = KP * (target_A[i] - q) - KD * dq

        for i in range(12):
            q = data.qpos[CTRL_TO_QPOS_B[i]]
            dq = data.qvel[CTRL_TO_QVEL_B[i]]
            data.ctrl[i + 12] = KP * (target_B[i] - q) - KD * dq

        mujoco.mj_step(model, data)
        viewer.sync()
        step += 1
        time.sleep(0.001)

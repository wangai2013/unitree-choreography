"""
Day 5-1 修正版: Go2 点头（正确的 ctrl ↔ qpos 映射）
关键：ctrl[i] 和 qpos 的对应关系不是简单的 7+i
"""
import mujoco
import mujoco.viewer
import numpy as np
import time

MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# ⭐ 关键：ctrl[i] 对应的 qpos 索引（从诊断结果手动填入）
CTRL_TO_QPOS = np.array([
    10, 11, 12,   # ctrl[0,1,2]  → FR_hip, FR_thigh, FR_calf
     7,  8,  9,   # ctrl[3,4,5]  → FL_hip, FL_thigh, FL_calf
    16, 17, 18,   # ctrl[6,7,8]  → RR_hip, RR_thigh, RR_calf
    13, 14, 15,   # ctrl[9,10,11] → RL_hip, RL_thigh, RL_calf
])

# 站立姿态（顺序按 ctrl 索引：FR, FL, RR, RL）
STAND = np.array([0.0, 0.8, -1.5] * 4)

# ⭐ 直接用正确映射设置初始姿态
for i in range(12):
    data.qpos[CTRL_TO_QPOS[i]] = STAND[i]
data.qpos[2] = 0.3   # 机身高度

mujoco.mj_forward(model, data)

KP, KD = 300.0, 15.0

print("=" * 60)
print("🎬 Go2 点头（修正映射版）")
print(f"  电机顺序: FR(0-2), FL(3-5), RR(6-8), RL(9-11)")
print(f"  ctrl→qpos 映射: {CTRL_TO_QPOS}")
print("=" * 60)

step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001

        target = STAND.copy()
        swing = 0.3 * np.sin(2 * np.pi * 0.8 * t)
        # 前腿小腿点头（ctrl[2]=FR_calf, ctrl[5]=FL_calf）
        target[2] = -1.5 + swing
        target[5] = -1.5 + swing

        for i in range(12):
            q_idx = CTRL_TO_QPOS[i]   # ⭐ 用正确映射
            q  = data.qpos[q_idx]
            dq = data.qvel[q_idx - 1]  # qvel 是 qpos 偏移 -1（free joint 占 6 个 qvel）
            data.ctrl[i] = KP * (target[i] - q) - KD * dq

        mujoco.mj_step(model, data)
        viewer.sync()

        step += 1
        time.sleep(0.001)

        if step % 500 == 0:
            # 打印真实状态
            fr_thigh_q = data.qpos[11]
            fl_thigh_q = data.qpos[8]
            print(f"t={t:.2f}s | 高度 {data.qpos[2]:.3f} | FR大腿 {fr_thigh_q:+.3f} | FL大腿 {fl_thigh_q:+.3f}")

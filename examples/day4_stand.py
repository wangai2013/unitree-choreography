"""
Day 4: 让 Go2 站立（PD 控制）
原理：给 12 个关节设置站立姿态角度，用 PD 控制器逐步收敛
"""
import mujoco
import mujoco.viewer
import numpy as np
import time

MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# ========== 先看 XML 里的限制 ==========
print("=" * 60)
print("📋 12 个执行器的力矩范围（ctrlrange）：")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    lo, hi = model.actuator_ctrlrange[i]
    print(f"  [{i:2d}] {name:20s} 力矩范围: [{lo:7.2f}, {hi:7.2f}]")
print("=" * 60)

# ========== 核心参数 ==========
# 站立姿态的关节角度（弧度），顺序：FR, FL, RR, RL，每腿 3 关节
STAND_ANGLES = np.array([
    0.0,  1.5, -2.5,   # FR: 髋, 大腿, 小腿
    0.0,  0.8, -1.5,   # FL
    0.0,  0.8, -1.5,   # RR
    0.0,  0.8, -1.5,   # RL
])

KP = 600.0   # 位置增益
KD = 15    # 阻尼增益

# ========== 初始化 ==========
print(f"📂 加载模型: {MODEL_PATH}")
print(f"🎛️  PD 参数: Kp={KP}, Kd={KD}")
print(f"📐 目标姿态: {STAND_ANGLES}")
print("-" * 50)

# 抬高一点，避免一开始就卡地
data.qpos[2] = 0.4
mujoco.mj_forward(model, data)

step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # ---- PD 控制 ----
        for i in range(12):
            q  = data.qpos[7 + i]   # 关节角度（前 7 个是基座）
            dq = data.qvel[6 + i]   # 关节角速度（前 6 个是基座速度）

            torque = KP * (STAND_ANGLES[i] - q) - KD * dq
            data.ctrl[i] = torque

        mujoco.mj_step(model, data)
        step += 1

        if step % 500 == 0:
            h = data.qpos[2]
            print(f"步数 {step:5d} | 高度 {h:.3f} m")

        time.sleep(0.001)

print("\n👋 仿真结束")

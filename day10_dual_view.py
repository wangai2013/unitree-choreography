"""Day 10: 双机 Go2 可视化"""
import mujoco
import mujoco.viewer
import numpy as np
import time

MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene_dual_final.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print(f"✅ 加载: nq={model.nq}, nu={model.nu}, nbody={model.nbody}")

# 打印所有 freejoint（两台机器人的机身）
print("\n📋 所有 freejoint:")
for i in range(model.njnt):
    jtype = model.jnt_type[i]
    if jtype == 0:  # free joint
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"  joint[{i}]: {name}, qpos起始={model.jnt_qposadr[i]}")

# 站立姿态
STAND = np.array([0.0, 0.8, -1.5] * 4)
KP, KD = 300.0, 15.0

step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # 简单：所有电机 PD 控制到 STAND（按 i % 12 循环，两台机器人用同一目标）
        for i in range(model.nu):
            joint_id = model.actuator_trnid[i][0]
            qpos_adr = model.jnt_qposadr[joint_id]
            dof_adr = model.jnt_dofadr[joint_id]
            target = STAND[i % 12]
            q = data.qpos[qpos_adr]
            dq = data.qvel[dof_adr]
            data.ctrl[i] = KP * (target - q) - KD * dq

        mujoco.mj_step(model, data)
        viewer.sync()
        step += 1
        time.sleep(0.001)

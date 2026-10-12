"""
Day 3-1: 加载宇树 Go2 模型（修正版）
"""
import mujoco
import mujoco.viewer
import time

MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene.xml"

print(f"📂 加载模型: {MODEL_PATH}")
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print("=" * 50)
print(f"✅ 模型加载成功！")
print(f"  关节总数 (nq): {model.nq}")
print(f"  电机数 (nu):   {model.nu}")
print(f"  刚体数 (nbody): {model.nbody}")
print("=" * 50)

print("\n📋 所有关节：")
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    print(f"  [{i}] {name}")

print("\n📋 所有电机：")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"  [{i}] {name}")

print("\n🎬 启动查看器（关闭窗口或 Ctrl+C 退出）...")
with mujoco.viewer.launch_passive(model, data):
    while True:
        mujoco.mj_step(model, data)
        time.sleep(0.001)

"""
诊断 v2: 查看 Go2 的 qpos 真实结构
"""
import mujoco
import numpy as np

MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print("=" * 70)
print(f"模型信息:")
print(f"  nq (qpos 总长度)  = {model.nq}")
print(f"  nv (qvel 总长度)  = {model.nv}")
print(f"  nu (ctrl 总长度)  = {model.nu}")
print(f"  njnt (关节总数)   = {model.njnt}")
print("=" * 70)

# 关节类型映射
JTYPE = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}

print("\n📋 所有关节 (按 qpos 索引):")
print(f"  {'id':>3} | {'name':25s} | {'type':>6} | {'qpos起始':>8} | {'qpos长度':>8} | {'qvel起始':>8}")
print("-" * 80)
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if name is None:
        name = "(无名)"
    jtype = JTYPE.get(model.jnt_type[i], "?")
    qposadr = model.jnt_qposadr[i]
    qposlen = model.jnt_qposadr[i+1] - qposadr if i+1 < model.njnt else model.nq - qposadr
    dofadr = model.jnt_dofadr[i]
    print(f"  {i:>3} | {name:25s} | {jtype:>6} | {qposadr:>8} | {qposlen:>8} | {dofadr:>8}")

print("\n📋 所有执行器 (按 ctrl 索引):")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    if name is None:
        name = "(无名)"
    trnid = model.actuator_trnid[i][0]
    jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, trnid)
    if jname is None:
        jname = f"关节{trnid}"
    print(f"  电机{i:2d}: {name:20s} → 控制关节 '{jname}'")

print("\n" + "=" * 70)

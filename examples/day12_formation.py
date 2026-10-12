"""
Day 12: 双机队形变换
A、B 平滑切换多种队形
"""
import mujoco
import mujoco.viewer
import numpy as np
import time

MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene_dual_final.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# 找两台机器人的 freejoint 的 qpos 起始
FREE_JOINTS = []
for i in range(model.njnt):
    if model.jnt_type[i] == 0:   # free joint
        FREE_JOINTS.append(i)

print(f"✅ 发现 {len(FREE_JOINTS)} 个 freejoint")

A_qpos = model.jnt_qposadr[FREE_JOINTS[0]]
B_qpos = model.jnt_qposadr[FREE_JOINTS[1]]
print(f"A qpos起始: {A_qpos}")
print(f"B qpos起始: {B_qpos}")

# 建立 ctrl 映射
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

# ===== 队形定义：每 3 秒切换一个队形 =====
# (A位置(x,y), B位置(x,y), 描述)
FORMATIONS = [
    ((-1.0, 0.0), ( 1.0, 0.0), "左右并排"),
    (( 0.0,-1.0), ( 0.0, 1.0), "前后排列"),
    ((-1.0,-1.0), ( 1.0, 1.0), "对角"),
    (( 0.0, 0.0), ( 0.0, 0.0), "重叠（测试）"),
    ((-1.0, 0.0), ( 1.0, 0.0), "回左右"),
]

FORMATION_DURATION = 3.0   # 每个队形 3 秒
HEIGHT = 0.32              # 机器人站立高度

print("=" * 60)
print("🎬 双机队形变换")
for i, (pos_A, pos_B, desc) in enumerate(FORMATIONS):
    print(f"  [{i+1}] {desc:12s} | A{pos_A} B{pos_B}")
print("=" * 60)

def smoothstep(x):
    """平滑过渡 0→1"""
    x = np.clip(x, 0, 1)
    return 3*x**2 - 2*x**3

step = 0
current_formation = 0
last_formation = -1
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001

        # 当前是第几个队形
        idx = int(t / FORMATION_DURATION)
        if idx >= len(FORMATIONS):
            idx = len(FORMATIONS) - 1
        
        # 队形切换进度
        t_in_formation = t - idx * FORMATION_DURATION
        transition_progress = smoothstep(t_in_formation / 0.5)  # 前 0.5 秒过渡

        # 计算当前目标位置
        # 从上个队形平滑过渡到当前队形
        if idx == 0:
            prev_A, prev_B = FORMATIONS[0][0], FORMATIONS[0][1]
        else:
            prev_A, prev_B = FORMATIONS[idx-1][0], FORMATIONS[idx-1][1]
        cur_A, cur_B = FORMATIONS[idx][0], FORMATIONS[idx][1]

        pos_A_x = prev_A[0] + (cur_A[0] - prev_A[0]) * transition_progress
        pos_A_y = prev_A[1] + (cur_A[1] - prev_A[1]) * transition_progress
        pos_B_x = prev_B[0] + (cur_B[0] - prev_B[0]) * transition_progress
        pos_B_y = prev_B[1] + (cur_B[1] - prev_B[1]) * transition_progress

        # 打印队形切换
        if idx != last_formation:
            print(f"🎬 t={t:.1f}s → {FORMATIONS[idx][2]}")
            last_formation = idx

        # ⭐ 直接设置机身位置
        # qpos[A_qpos+0] = x, +1 = y, +2 = z
        # qpos[A_qpos+3:7] = 四元数（保持默认）
        data.qpos[A_qpos + 0] = pos_A_x
        data.qpos[A_qpos + 1] = pos_A_y
        data.qpos[A_qpos + 2] = HEIGHT

        data.qpos[B_qpos + 0] = pos_B_x
        data.qpos[B_qpos + 1] = pos_B_y
        data.qpos[B_qpos + 2] = HEIGHT

        # 清空机身速度（防止飘）
        data.qvel[0:6] = 0
        data.qvel[A_qpos - 1:A_qpos - 1 + 6] = 0
        # ⚠️ 上面这行取决于 qvel 索引，简化处理：

        # 站立姿态 PD 控制
        target = STAND.copy()
        for i in range(12):
            q = data.qpos[CTRL_TO_QPOS_A[i]]
            dq = data.qvel[CTRL_TO_QVEL_A[i]]
            data.ctrl[i] = KP * (target[i] - q) - KD * dq

        for i in range(12):
            q = data.qpos[CTRL_TO_QPOS_B[i]]
            dq = data.qvel[CTRL_TO_QVEL_B[i]]
            data.ctrl[i + 12] = KP * (target[i] - q) - KD * dq

        mujoco.mj_step(model, data)
        viewer.sync()
        step += 1
        time.sleep(0.001)

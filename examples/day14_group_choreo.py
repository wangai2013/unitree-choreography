"""
Day 14: 4 台 Go2 群控舞蹈（毕业作品）
30 秒完整编舞：站位 + 同步 + 波浪 + 队形 + 镜像 + 收场
"""
import mujoco
import mujoco.viewer
import numpy as np
import time

MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene_quad_final.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# ===== 1. 找 4 个 freejoint（机身位置）=====
FREE_JOINTS = [i for i in range(model.njnt) if model.jnt_type[i] == 0]
QPOS_BASES = [model.jnt_qposadr[j] for j in FREE_JOINTS]
print(f"✅ 4 台 qpos 起始: {QPOS_BASES}")

# ===== 2. 电机映射（4 台 × 12）=====
N = 4
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
HEIGHT = 0.32

# ===== 3. 队形定义 =====
FORMATION_SQUARE = [(-1.2, -0.6), (1.2, -0.6), (-1.2, 0.6), (1.2, 0.6)]  # 方阵
FORMATION_DIAMOND = [(0.0, -1.2), (1.2, 0.0), (-1.2, 0.0), (0.0, 1.2)]     # 菱形

# ===== 4. 舞蹈时间轴 =====
DANCE_TIMELINE = [
    # (模式, 时长, 参数)
    ("square",   3.0, None),                 # 站方阵
    ("sync_nod", 5.0, None),                 # 同步点头
    ("wave_nod", 5.0, 0.3),                  # 波浪点头（每台延迟 0.3s）
    ("to_diamond", 5.0, None),               # 变换队形：方阵 → 菱形
    ("mirror_sway", 6.0, None),              # 镜像摇摆
    ("to_square", 5.0, None),                # 变换回方阵
    ("final_stand", 1.0, None),              # 收场站定
]
total_duration = sum(d for _, d, _ in DANCE_TIMELINE)

print("=" * 60)
print("🎬 4 台群控舞蹈（毕业作品）")
for i, (mode, dur, _) in enumerate(DANCE_TIMELINE):
    print(f"  [{i+1}] {mode:14s} — {dur:.1f}s")
print(f"  总时长: {total_duration:.1f} 秒")
print("=" * 60)

def get_current_mode(t):
    """返回 (模式, 局部时间, 时长)"""
    acc = 0.0
    for mode, dur, param in DANCE_TIMELINE:
        if t < acc + dur:
            return mode, t - acc, dur, param
        acc += dur
    return None, None, None, None

def smoothstep(x):
    x = np.clip(x, 0, 1)
    return 3*x**2 - 2*x**3

# 记录上一步的队形位置（用于平滑过渡）
last_positions = [list(FORMATION_SQUARE[r]) for r in range(N)]

step = 0
last_mode = None
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001
        mode, t_local, dur, param = get_current_mode(t)

        if mode != last_mode:
            print(f"🎬 t={t:.1f}s → {mode}")
            last_mode = mode

        # ===== 默认目标：站立 =====
        targets = [STAND.copy() for _ in range(N)]
        positions = [list(last_positions[r]) for r in range(N)]

        if mode is None or mode == "final_stand":
            pass   # 保持默认

        elif mode == "square":
            # 站位（保持方阵）
            positions = [list(FORMATION_SQUARE[r]) for r in range(N)]

        elif mode == "sync_nod":
            # 同步点头
            swing = 0.4 * np.sin(2 * np.pi * 1.5 * t_local)
            for r in range(N):
                targets[r][2] = -1.5 + swing
                targets[r][5] = -1.5 + swing

        elif mode == "wave_nod":
            # 波浪点头（每台延迟 param 秒）
            for r in range(N):
                t_r = max(0, t_local - r * param)
                swing = 0.4 * np.sin(2 * np.pi * 1.5 * t_r)
                targets[r][2] = -1.5 + swing
                targets[r][5] = -1.5 + swing

        elif mode == "to_diamond":
            # 方阵 → 菱形（前 1 秒过渡）
            p = smoothstep(t_local / 1.0)
            for r in range(N):
                x0, y0 = FORMATION_SQUARE[r]
                x1, y1 = FORMATION_DIAMOND[r]
                positions[r] = [x0 + (x1 - x0) * p, y0 + (y1 - y0) * p]

        elif mode == "mirror_sway":
            # 镜像摇摆（保持菱形）
            positions = [list(FORMATION_DIAMOND[r]) for r in range(N)]
            sway = 0.2 * np.sin(2 * np.pi * 1.0 * t_local)
            for r in range(N):
                # 偶数台往左，奇数台往右
                sign = +1 if r % 2 == 0 else -1
                targets[r][1] = 0.8 + sign * sway
                targets[r][4] = 0.8 - sign * sway

        elif mode == "to_square":
            # 菱形 → 方阵
            p = smoothstep(t_local / 1.0)
            for r in range(N):
                x0, y0 = FORMATION_DIAMOND[r]
                x1, y1 = FORMATION_SQUARE[r]
                positions[r] = [x0 + (x1 - x0) * p, y0 + (y1 - y0) * p]

        # ===== 应用：位置 + PD 控制 =====
        for r in range(N):
            # 设置机身位置（平滑）
            data.qpos[QPOS_BASES[r] + 0] = positions[r][0]
            data.qpos[QPOS_BASES[r] + 1] = positions[r][1]
            data.qpos[QPOS_BASES[r] + 2] = HEIGHT
            # 清速度
            data.qvel[QPOS_BASES[r] - 0:QPOS_BASES[r] - 0 + 6] = 0

            # PD 控制关节
            for i in range(12):
                q = data.qpos[CTRL_TO_QPOS[r][i]]
                dq = data.qvel[CTRL_TO_QVEL[r][i]]
                data.ctrl[r * 12 + i] = KP * (targets[r][i] - q) - KD * dq

        # 记录当前位置
        for r in range(N):
            last_positions[r] = list(positions[r])

        mujoco.mj_step(model, data)
        viewer.sync()
        step += 1
        time.sleep(0.001)

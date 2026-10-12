"""
Day 16: 队形 + 音乐 + 动作 三合一
- 从 JSON 加载队形
- 从 JSON 加载舞蹈序列
- 4 台 Go2 跟着音乐跳舞 + 变换队形
"""
import mujoco
import mujoco.viewer
import numpy as np
import time
import json
import os
import sys

# ===== 路径修复 =====
if os.path.basename(os.getcwd()) == "examples":
    os.chdir("..")

# ===== 1. 加载队形 =====
with open("dances/formations.json", "r", encoding="utf-8") as f:
    formations_data = json.load(f)

print("=" * 60)
print("📋 可用队形:")
for key, val in formations_data["formations"].items():
    print(f"  - {key:10s}: {val['name']}")
print("=" * 60)

# ===== 2. 加载音乐 =====
try:
    import pygame
    pygame.mixer.init()
    pygame.mixer.music.load("music.mp3")
    music_ready = True
    print("🎵 音乐就绪")
except Exception as e:
    music_ready = False
    print(f"⚠️ 音乐加载失败: {e}")

# ===== 3. 加载模型 =====
MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene_quad_final.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# 4 台映射
N = 4
FREE_JOINTS = [i for i in range(model.njnt) if model.jnt_type[i] == 0]
QPOS_BASES = [model.jnt_qposadr[j] for j in FREE_JOINTS]

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

# ===== 4. 舞蹈时间轴 =====
# 每段格式：(动作, 持续秒, 目标队形)
DANCE = [
    ("stand",       3.0, "square"),
    ("sync_nod",    4.0, "square"),
    ("wave_nod",    4.0, "square"),
    ("to_diamond",  3.0, "diamond"),
    ("mirror_sway", 4.0, "diamond"),
    ("to_circle",   3.0, "circle"),
    ("sync_nod",    4.0, "circle"),
    ("to_line",     3.0, "line"),
    ("wave_nod",    4.0, "line"),
    ("to_square",   3.0, "square"),
    ("stand",       2.0, "square"),
]

total = sum(d for _, d, _ in DANCE)
print(f"⏱  总时长: {total:.1f} 秒")
print("=" * 60)

def get_current_segment(t):
    """返回 (动作, 局部时间, 目标队形, 进度)"""
    acc = 0.0
    for i, (action, dur, formation) in enumerate(DANCE):
        if t < acc + dur:
            return action, t - acc, dur, formation
        acc += dur
    return None, None, None, None

def smoothstep(x):
    x = np.clip(x, 0, 1)
    return 3*x**2 - 2*x**3

# 上一步的队形（用于平滑过渡）
last_positions = [list(formations_data["formations"]["square"]["positions"][r]) for r in range(N)]

# ===== 5. 主循环 =====
step = 0
music_started = False
last_action = None
last_formation = None

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = step * 0.001

        # 2 秒后启动音乐
        if t >= 2.0 and music_ready and not music_started:
            pygame.mixer.music.play()
            music_started = True
            print("🎵 音乐开始")

        music_t = t - 2.0

        # ===== 默认：站立 + 保持队形 =====
        targets = [STAND.copy() for _ in range(N)]
        positions = [list(last_positions[r]) for r in range(N)]
        current_formation = "square"

        if music_t > 0:
            action, t_local, dur, formation = get_current_segment(music_t)

            if action is None:
                action = "stand"
                t_local = 0
                dur = 1
                formation = "square"

            # 打印动作切换
            if action != last_action and action is not None:
                print(f"🎬 {music_t:.2f}s → {action} @ {formation}")
                last_action = action

            # 队形切换
            if formation != last_formation:
                print(f"   📐 队形 → {formation}")
                last_formation = formation

            current_formation = formation

            # ===== 处理动作 =====
            if action == "sync_nod":
                swing = 0.4 * np.sin(2 * np.pi * 1.5 * t_local)
                for r in range(N):
                    targets[r][2] = -1.5 + swing
                    targets[r][5] = -1.5 + swing

            elif action == "wave_nod":
                for r in range(N):
                    t_r = max(0, t_local - r * 0.3)
                    swing = 0.4 * np.sin(2 * np.pi * 1.5 * t_r)
                    targets[r][2] = -1.5 + swing
                    targets[r][5] = -1.5 + swing

            elif action == "mirror_sway":
                sway = 0.2 * np.sin(2 * np.pi * 1.0 * t_local)
                for r in range(N):
                    sign = +1 if r % 2 == 0 else -1
                    targets[r][1] = 0.8 + sign * sway
                    targets[r][4] = 0.8 - sign * sway

            elif action.startswith("to_"):
                # 队形变换（前 1 秒过渡）
                p = smoothstep(t_local / 1.0)
                target_positions = formations_data["formations"][formation]["positions"]
                for r in range(N):
                    x0, y0 = last_positions[r]
                    x1, y1 = target_positions[r]
                    positions[r] = [x0 + (x1 - x0) * p, y0 + (y1 - y0) * p]
            
            elif action == "stand":
                positions = [list(formations_data["formations"][formation]["positions"][r]) for r in range(N)]

        # 应用
        for r in range(N):
            data.qpos[QPOS_BASES[r] + 0] = positions[r][0]
            data.qpos[QPOS_BASES[r] + 1] = positions[r][1]
            data.qpos[QPOS_BASES[r] + 2] = HEIGHT
            data.qvel[QPOS_BASES[r]:QPOS_BASES[r] + 6] = 0

            for i in range(12):
                q = data.qpos[CTRL_TO_QPOS[r][i]]
                dq = data.qvel[CTRL_TO_QVEL[r][i]]
                data.ctrl[r * 12 + i] = KP * (targets[r][i] - q) - KD * dq

            last_positions[r] = list(positions[r])

        mujoco.mj_step(model, data)
        viewer.sync()
        step += 1
        time.sleep(0.001)

if music_ready:
    pygame.mixer.music.stop()

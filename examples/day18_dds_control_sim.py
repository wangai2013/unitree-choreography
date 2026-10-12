"""
Day 18: 用 DDS 控制 MuJoCo 仿真 Go2（修复版）
关键：主循环按"真实时间"跑，不按"step 数"
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import mujoco
import numpy as np
import time
import threading

from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__LowCmd_ as LowCmd_default,
    unitree_go_msg_dds__LowState_ as LowState_default,
)
from robot_utils import build_ctrl_to_qpos_map

# ===== 1. 加载 MuJoCo =====
MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
CTRL_TO_QPOS, CTRL_TO_QVEL = build_ctrl_to_qpos_map(model)
print(f"✅ MuJoCo: nq={model.nq}, nu={model.nu}")

STAND = np.array([0.0, 0.8, -1.5] * 4)
for i in range(12):
    data.qpos[CTRL_TO_QPOS[i]] = STAND[i]
data.qpos[2] = 0.3
mujoco.mj_forward(model, data)

# ===== 2. DDS 初始化 =====
print("🔌 DDS 初始化...")
ChannelFactoryInitialize(0, "lo0")

pub_lowstate = ChannelPublisher("rt/lowstate", LowState_)
pub_lowstate.Init()

latest_cmd = [None]
cmd_count = [0]

def cmd_handler(msg):
    latest_cmd[0] = msg
    cmd_count[0] += 1

sub_lowcmd = ChannelSubscriber("rt/lowcmd", LowCmd_)
sub_lowcmd.Init(cmd_handler, 10)

print("✅ DDS 就绪")

# ===== 3. 控制线程 =====
def control_thread():
    print("🎬 控制线程启动，等 1 秒...")
    time.sleep(1.0)
    print("🎬 开始发送指令...")
    
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    
    step = 0
    while True:
        t = step * 0.02
        
        cmd = LowCmd_default()
        swing = 0.4 * np.sin(2 * np.pi * 1.0 * t)
        
        for i in range(12):
            target = STAND[i]
            if i == 2 or i == 5:
                target = -1.5 + swing
            
            cmd.motor_cmd[i].q = target
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kp = 300.0
            cmd.motor_cmd[i].kd = 15.0
            cmd.motor_cmd[i].tau = 0.0
        
        pub.Write(cmd)
        step += 1
        time.sleep(0.02)

t = threading.Thread(target=control_thread, daemon=True)
t.start()

# ===== 4. 主循环（按真实时间）=====
print("\n" + "=" * 60)
print("🎬 仿真主循环启动（跑 15 秒真实时间）")
print("=" * 60)

SIM_DURATION = 15.0   # ⭐ 跑 15 秒"真实时间"
start_time = time.time()
step = 0
last_print_time = start_time

while time.time() - start_time < SIM_DURATION:
    # 应用最新指令
    if latest_cmd[0] is not None:
        cmd = latest_cmd[0]
        for i in range(12):
            q_target = cmd.motor_cmd[i].q
            kp = cmd.motor_cmd[i].kp
            kd = cmd.motor_cmd[i].kd
            
            q = data.qpos[CTRL_TO_QPOS[i]]
            dq = data.qvel[CTRL_TO_QVEL[i]]
            data.ctrl[i] = kp * (q_target - q) - kd * dq
    
    # 步进物理
    mujoco.mj_step(model, data)
    step += 1
    
    # ⭐ 每 50 步 sleep 一次（控制仿真速度）
    if step % 50 == 0:
        time.sleep(0.001)   # 让出 CPU，控制帧率
    
    # ⭐ 每 0.5 秒打印一次
    now = time.time()
    if now - last_print_time >= 0.5:
        elapsed = now - start_time
        print(f"t={elapsed:5.2f}s | step={step:6d} | "
              f"收到指令 {cmd_count[0]:5d} 条 | "
              f"高度 {data.qpos[2]:.3f} | "
              f"FR小腿 {data.qpos[CTRL_TO_QPOS[2]]:.3f}")
        last_print_time = now

print("\n✅ 仿真结束")
print(f"   总收到指令：{cmd_count[0]} 条")

"""
Day 18: DDS 回环测试
验证"发布-订阅"机制（在同一进程里）
"""
import time
import numpy as np
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

# ===== 1. 初始化 DDS =====
print("🔌 初始化 DDS...")
ChannelFactoryInitialize(0, "lo0")
print("✅ DDS 就绪")

# ===== 2. 创建发布者 & 订阅者 =====
pub = ChannelPublisher("rt/lowcmd", LowCmd_)
pub.Init()

received_count = [0]   # 用列表绕过 Python 作用域限制
last_cmd = [None]

def handler(msg):
    received_count[0] += 1
    last_cmd[0] = msg

sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
sub.Init(handler, 10)

print("📡 已订阅 rt/lowcmd")

# ===== 3. 发送测试消息 =====
print("\n📤 发送 5 条消息...")
for i in range(5):
    cmd = LowCmd_default()
    for j in range(12):
        cmd.motor_cmd[j].q = 0.8 + 0.05 * i   # 目标角度
        cmd.motor_cmd[j].kp = 60.0
        cmd.motor_cmd[j].kd = 3.5
        cmd.motor_cmd[j].dq = 0.0
        cmd.motor_cmd[j].tau = 0.0
    
    pub.Write(cmd)
    print(f"  第 {i+1} 条已发送")
    time.sleep(0.2)

# ===== 4. 等一会儿，看收到几条 =====
time.sleep(1.0)

print(f"\n📥 共收到 {received_count[0]} 条消息")

if received_count[0] > 0:
    print("🎉 DDS 发布-订阅 成功！")
    if last_cmd[0] is not None:
        print(f"   最后一条 motor_cmd[0].q = {last_cmd[0].motor_cmd[0].q:.3f}")
        print(f"   最后一条 motor_cmd[0].kp = {last_cmd[0].motor_cmd[0].kp:.3f}")
else:
    print("❌ 没收到消息")

"""
Day 2 练习：读取机器人关节状态（只读，不发指令）
"""
import sys
import time

try:
    from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)


def low_state_handler(msg):
    """收到关节状态时的回调"""
    print(f"⏱  时间戳: {msg.stamp.sec}.{msg.stamp.nanosec}")
    print(f"📊 关节数: {len(msg.motor_state)}")
    for i in range(min(6, len(msg.motor_state))):
        m = msg.motor_state[i]
        print(f"  关节{i}: 角度={m.q:.3f} rad, 温度={m.temperature}°C")
    print("-" * 40)


def main():
    iface = "lo0"  # 回环，安全
    print(f"🔌 使用网络接口: {iface}")
    ChannelFactoryInitialize(0, iface)
    print("✅ DDS 初始化成功")

    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(low_state_handler, 10)
    print("📡 已订阅 rt/lowstate，等待数据...")
    print("（没有实机/仿真时不会有输出，按 Ctrl+C 退出）")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 退出")


if __name__ == "__main__":
    main()

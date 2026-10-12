"""
Day 8: 模拟节拍（不用音乐文件）
设定 BPM = 120（每 0.5 秒一个节拍）
"""
import numpy as np

BPM = 120
BEAT_INTERVAL = 60.0 / BPM   # 每拍间隔秒数
DURATION = 20.0              # 总时长

num_beats = int(DURATION / BEAT_INTERVAL)
beats = [i * BEAT_INTERVAL for i in range(num_beats)]

print(f"🥁 BPM: {BPM}（每拍 {BEAT_INTERVAL:.2f} 秒）")
print(f"🎼 总时长: {DURATION} 秒")
print(f"🎯 节拍数: {len(beats)}")
print(f"\n前 20 个节拍：")
for i, t in enumerate(beats[:20]):
    print(f"  节拍 {i+1:3d}: {t:.3f} 秒")

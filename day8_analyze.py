"""
Day 8: 音乐节拍分析
输入：music.mp3（或其他音频文件）
输出：BPM + 节拍时间点
"""
import librosa
import sys

MUSIC_FILE = sys.argv[1] if len(sys.argv) > 1 else "music.mp3"

print(f"🎵 分析: {MUSIC_FILE}")
print("=" * 60)

# 加载音频
y, sr = librosa.load(MUSIC_FILE, sr=None)
duration = len(y) / sr
print(f"🎼 时长: {duration:.2f} 秒")
print(f"📻 采样率: {sr} Hz")

# 提取节拍
tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")

print(f"🥁 BPM: {float(tempo):.1f}")
print(f"🎯 节拍数: {len(beats)}")
print(f"\n📋 前 20 个节拍时刻（秒）：")
for i, t in enumerate(beats[:20]):
    print(f"  节拍 {i+1:3d}: {t:.3f} 秒")

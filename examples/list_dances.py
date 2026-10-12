"""列出所有可用舞蹈"""
import os
import json

DANCES_DIR = "dances"

if not os.path.exists(DANCES_DIR):
    print("❌ 没有 dances/ 目录")
    exit(1)

print("=" * 60)
print("📚 可用舞蹈列表")
print("=" * 60)

for fname in sorted(os.listdir(DANCES_DIR)):
    if not fname.endswith(".json"):
        continue
    path = os.path.join(DANCES_DIR, fname)
    with open(path, "r", encoding="utf-8") as f:
        dance = json.load(f)
    total_beats = sum(s["beats"] for s in dance["sequence"])
    total_time = total_beats * 60.0 / dance["bpm"]
    print(f"\n📄 {fname}")
    print(f"   名称: {dance['name']}")
    print(f"   描述: {dance.get('description', '无')}")
    print(f"   BPM:  {dance['bpm']}")
    print(f"   拍数: {total_beats}（{total_time:.1f} 秒）")
    print(f"   动作数: {len(dance['sequence'])}")
    print(f"   音乐: {dance.get('music', '无')}")

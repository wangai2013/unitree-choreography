"""
Day 44: 渲染 HalfCheetah（RL 训练效果）
"""
import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import time

# 加载模型
print("📂 加载模型: models/halfcheetah_ppo.zip")
model = PPO.load("models/halfcheetah_ppo")

# 环境（人类可视）
env = gym.make("HalfCheetah-v4", render_mode="human")

print("=" * 60)
print("🎬 渲染 HalfCheetah（跑 5 局）")
print("=" * 60)

total_all = []
for ep in range(5):
    obs, _ = env.reset()
    total = 0
    step = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, _ = env.step(action)
        total += reward
        step += 1
        done = term or trunc
        time.sleep(0.01)  # 慢一点看
    total_all.append(total)
    print(f"第 {ep+1} 局: {total:.1f} 分（{step} 步）")

print()
print("=" * 60)
print(f"平均分: {np.mean(total_all):.1f}")
print(f"最高分: {np.max(total_all):.1f}")
print(f"最低分: {np.min(total_all):.1f}")
print("=" * 60)

env.close()

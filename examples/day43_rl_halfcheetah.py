"""
Day 43: RL 训练 MuJoCo 机器人（HalfCheetah）
- 6 关节连续控制
- PPO + MPS
"""
import gymnasium as gym
from stable_baselines3 import PPO
import torch
import time
import os

# 1. 环境
env = gym.make("HalfCheetah-v4")

# 2. 设备
device = "cpu"  # ⭐ PPO + MLP 用 CPU 更快
print(f"🎯 训练设备: {device}")
print(f"📊 观测空间: {env.observation_space}")
print(f"🎮 动作空间: {env.action_space}")

# 3. 模型（PPO）
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device=device,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    learning_rate=3e-4,
    gamma=0.99,
)

# 4. 训练
print("=" * 60)
print("🎬 开始训练（100000 步）")
print("=" * 60)

t0 = time.time()
model.learn(total_timesteps=1000000)
t1 = time.time()

print(f"✅ 训练完成！耗时 {t1 - t0:.1f} 秒")

# 5. 保存
os.makedirs("models", exist_ok=True)
model.save("models/halfcheetah_ppo")
print("✅ 模型已保存: models/halfcheetah_ppo.zip")

# 6. 测试
print()
print("=" * 60)
print("🎬 测试（跑 3 局）")
print("=" * 60)

env_test = gym.make("HalfCheetah-v4")
for i in range(3):
    obs, _ = env_test.reset()
    total_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env_test.step(action)
        total_reward += reward
        done = terminated or truncated
    print(f"第 {i+1} 局: 得分 {total_reward:.1f}")

env.close()

"""
Day 42: RL 入门 —— 训练"倒立摆"（CartPole）
用 PPO + MPS
"""
import gymnasium as gym
from stable_baselines3 import PPO
import torch
import time

# 1. 环境
env = gym.make("CartPole-v1")

# 2. 设备（MPS / CPU）
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🎯 训练设备: {device}")

# 3. PPO 模型
model = PPO("MlpPolicy", env, verbose=1, device=device)

# 4. 训练
print("=" * 60)
print("🎬 开始训练（10000 步）")
print("=" * 60)

t0 = time.time()
model.learn(total_timesteps=10000)
t1 = time.time()

print(f"✅ 训练完成！耗时 {t1 - t0:.1f} 秒")

# 5. 保存
model.save("models/cartpole_ppo")
print("✅ 模型已保存: models/cartpole_ppo.zip")

# 6. 测试
print()
print("=" * 60)
print("🎬 测试（跑 3 局）")
print("=" * 60)

env_test = gym.make("CartPole-v1", render_mode=None)
for i in range(3):
    obs, _ = env_test.reset()
    total_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env_test.step(action)
        total_reward += reward
        done = terminated or truncated
    print(f"第 {i+1} 局: 得分 {total_reward:.0f}")

env.close()

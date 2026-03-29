import os
import gymnasium as gym
import ale_py
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# Register Atari environments
gym.register_envs(ale_py)

ENV_ID = "ALE/BankHeist-v5"
N_EPISODES = 10  # increase to 20–30 for better accuracy

#  Create environment (same as training)
def make_env():
    def _init():
        env = gym.make(ENV_ID)
        env = AtariWrapper(env)
        return env
    return _init


# ✅ Evaluation function (VecEnv compatible)
def evaluate(model, env, n_episodes=10):
    rewards = []

    for _ in range(n_episodes):
        obs = env.reset()
        done = [False]  # VecEnv returns list/array
        total_reward = 0

        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            total_reward += reward[0]

        rewards.append(total_reward)

    return np.mean(rewards), np.std(rewards)


# 📦 Models to evaluate
MODELS = [
    ("Limpho", "exp1_standard_cnn", "models/limpho/exp1_standard_cnn/best_model.zip"),
    ("Limpho", "exp2_very_high_lr_small_batch", "models/limpho/exp2_very_high_lr_small_batch/best_model.zip"),
    ("Limpho", "exp3_myopic_gamma", "models/limpho/exp3_myopic_gamma/best_model.zip"),
    ("Limpho", "exp4_farsighted_gamma", "models/limpho/exp4_farsighted_gamma/best_model.zip"),
    ("Limpho", "exp5_low_initial_eps", "models/limpho/exp5_low_initial_eps/best_model.zip"),
    ("Limpho", "exp6_near_zero_final_eps", "models/limpho/exp6_near_zero_final_eps/best_model.zip"),
    ("Limpho", "exp7_balanced", "models/limpho/exp7_balanced/best_model.zip"),
    ("Limpho", "exp8_high_all", "models/limpho/exp8_high_all/best_model.zip"),
    ("Limpho", "exp9_mlp_tuned", "models/limpho/exp9_mlp_tuned/best_model.zip"),
    ("Limpho", "exp10_limpho_best", "models/limpho/exp10_limpho_best/best_model.zip"),
]

print("\n" + "="*70)
print(f"{'MEMBER':<10} {'EXPERIMENT':<35} {'AVG REWARD':>12} {'STD':>10}")
print("="*70)

results = []

for member, name, path in MODELS:
    if not os.path.exists(path):
        print(f"{member:<10} {name:<35} {'NOT FOUND':>12}")
        continue

    try:
        # ✅ Create correct environment
        env = DummyVecEnv([make_env()])
        env = VecFrameStack(env, n_stack=4)

        # ✅ Load model WITH environment (CRITICAL FIX)
        model = DQN.load(path, env=env)

        # ✅ Evaluate model
        mean_reward, std_reward = evaluate(model, env, N_EPISODES)
        results.append((mean_reward, member, name))

        print(f"{member:<10} {name:<35} {mean_reward:>12.2f} {std_reward:>10.2f}")

        env.close()

    except Exception as e:
        print(f"{member:<10} {name:<35} ERROR: {e}")

print("="*70)

# 🏆 Best model
if results:
    results.sort(reverse=True)
    best_reward, best_member, best_name = results[0]

    print(f"\n Best Limpho Model: {best_member} — {best_name}")
    print(f"Average Reward: {best_reward:.2f}")
else:
    print("\nNo valid models evaluated.")
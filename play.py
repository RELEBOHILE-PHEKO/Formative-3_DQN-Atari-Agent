import argparse
import time
import numpy as np
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

ENV_ID = "ALE/BankHeist-v5"
NUM_EPISODES = 5
RENDER_DELAY = 0.02
DEFAULT_MODEL_PATH = "models/rele/exp10_best_combo/best_model.zip"

# Selects the best action based on learned Q-values
def greedy_action(model, obs):
    action, _ = model.predict(obs, deterministic=True)
    return action

# Creates environment with same structure used during training
def make_vec_play_env():
    def _init():
        env = gym.make(ENV_ID, render_mode="human")
        env = AtariWrapper(env)
        return env
    env = DummyVecEnv([_init])
    env = VecFrameStack(env, n_stack=4)
    return env

# Runs evaluation loop over multiple episodes
def play(model_path):
    print("BankHeist-v5 DQN Evaluation")
    print(f"Loading model: {model_path}")

    model = DQN.load(model_path)
    print("Model loaded")

    env = make_vec_play_env()
    episode_rewards = []
    episode_lengths = []

    for episode in range(1, NUM_EPISODES + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        print(f"Episode {episode}/{NUM_EPISODES}")

        while not done:
            action = greedy_action(model, obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1
            time.sleep(RENDER_DELAY)

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"Reward: {total_reward:.1f}, Steps: {steps}")

    # Displays summary statistics after evaluation
    print("Summary")
    print(f"Episodes: {NUM_EPISODES}")
    print(f"Mean reward: {np.mean(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Std reward: {np.std(episode_rewards):.2f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.1f}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play BankHeist-v5 with a trained DQN model")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--delay", type=float, default=RENDER_DELAY)
    args = parser.parse_args()

    NUM_EPISODES = args.episodes
    RENDER_DELAY = args.delay

    play(args.model)

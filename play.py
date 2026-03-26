import argparse
import time
import numpy as np
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv


# Game we trained on
ENV_ID = "ALE/BankHeist-v5"

# How many episodes we want to watch
NUM_EPISODES = 5

# Small delay so the gameplay is visible
RENDER_DELAY = 0.02

# Shared best model after comparing everyone's experiments
DEFAULT_MODEL_PATH = "models/best_model.zip"


# During evaluation we act greedily (no randomness)
def greedy_action(model, obs):
    action, _ = model.predict(obs, deterministic=True)
    return action


# Create environment similar to training setup
def make_vec_play_env():
    def _init():
        env = gym.make(ENV_ID, render_mode="human")
        env = AtariWrapper(env)
        return env

    env = DummyVecEnv([_init])
    env = VecFrameStack(env, n_stack=4)
    return env


def play(model_path):
    print("\nBankHeist-v5 Evaluation")
    print("Group: Fidel, Limpho, Rele")
    print(f"Using model: {model_path}\n")

    # Load trained model
    model = DQN.load(model_path)
    print("Model loaded successfully.\n")

    env = make_vec_play_env()

    episode_rewards = []
    episode_lengths = []

    # Run multiple episodes
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

        print(f"Reward: {total_reward:.1f} | Steps: {steps}\n")

    # Final summary
    print("Summary")
    print(f"Episodes played: {NUM_EPISODES}")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Best reward: {np.max(episode_rewards):.2f}")
    print(f"Worst reward: {np.min(episode_rewards):.2f}")
    print(f"Reward std: {np.std(episode_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f}\n")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play BankHeist with a trained DQN model")

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--delay", type=float, default=RENDER_DELAY)

    args = parser.parse_args()

    NUM_EPISODES = args.episodes
    RENDER_DELAY = args.delay

    play(args.model)
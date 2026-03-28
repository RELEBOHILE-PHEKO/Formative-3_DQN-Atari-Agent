"""
train_rele.py - DQN Agent Training Script
Environment: ALE/BankHeist-v5
Author: Rele
"""

import os
import gymnasium as gym
import ale_py

from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback


# Total number of steps each experiment will train for
TOTAL_TIMESTEPS = 200_000

# If True, runs all experiments one after the other
# If False, runs only the selected experiment index below
RUN_ALL = True
ACTIVE_EXPERIMENT = 0


# Each experiment tweaks one idea so we can observe its effect on performance
EXPERIMENTS = [
    # Baseline setup to compare everything else against
    {
        "name": "exp1_baseline",
        "policy": "CnnPolicy",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch_size": 32,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "buffer_size": 100000,
    },

    # Higher learning rate to see if faster learning helps or destabilizes training
    {
        "name": "exp2_high_lr",
        "policy": "CnnPolicy",
        "lr": 5e-4,
        "gamma": 0.99,
        "batch_size": 32,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "buffer_size": 100000,
    },

    # Lower learning rate for slower but potentially more stable learning
    {
        "name": "exp3_low_lr",
        "policy": "CnnPolicy",
        "lr": 1e-5,
        "gamma": 0.99,
        "batch_size": 32,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "buffer_size": 100000,
    },

    # Lower gamma makes the agent focus more on immediate rewards
    {
        "name": "exp4_low_gamma",
        "policy": "CnnPolicy",
        "lr": 1e-4,
        "gamma": 0.90,
        "batch_size": 32,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "buffer_size": 100000,
    },

    # Larger batch size for more stable gradient updates
    {
        "name": "exp5_large_batch",
        "policy": "CnnPolicy",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch_size": 128,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "buffer_size": 100000,
    },

    # Smaller batch size for faster but noisier updates
    {
        "name": "exp6_small_batch",
        "policy": "CnnPolicy",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch_size": 16,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "buffer_size": 100000,
    },

    # Slower epsilon decay means more exploration for longer
    {
        "name": "exp7_long_exploration",
        "policy": "CnnPolicy",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch_size": 32,
        "exploration_fraction": 0.3,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "buffer_size": 100000,
    },

    # Higher final epsilon keeps some randomness even later in training
    {
        "name": "exp8_high_final_eps",
        "policy": "CnnPolicy",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch_size": 32,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.2,
        "buffer_size": 100000,
    },

    # MLP instead of CNN; reduced buffer to avoid memory crash
    {
        "name": "exp9_mlp_policy",
        "policy": "MlpPolicy",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch_size": 32,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "buffer_size": 10000,
    },

    # A tuned combination of hyperparameters
    {
        "name": "exp10_best_combo",
        "policy": "CnnPolicy",
        "lr": 2.5e-4,
        "gamma": 0.995,
        "batch_size": 64,
        "exploration_fraction": 0.15,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.01,
        "buffer_size": 100000,
    },
]


# Creates a single environment instance
# CNN policies require Atari preprocessing (grayscale, resizing, frame skipping)
def make_env(policy):
    def _init():
        env = gym.make("ALE/BankHeist-v5", render_mode=None)
        if policy == "CnnPolicy":
            env = AtariWrapper(env)
        return Monitor(env)
    return _init


# Runs one experiment from start to finish
def run_experiment(cfg):
    print(f"\nRunning {cfg['name']}")
    print(f"lr={cfg['lr']} gamma={cfg['gamma']} batch={cfg['batch_size']} buffer={cfg['buffer_size']}")

    log_dir = f"logs/rele/{cfg['name']}"
    save_dir = f"models/rele/{cfg['name']}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # Training environment
    vec_env = DummyVecEnv([make_env(cfg["policy"])])

    # Evaluation environment (separate to avoid bias)
    eval_env = DummyVecEnv([make_env(cfg["policy"])])

    # Frame stacking only applies to image-based inputs (CNN)
    if cfg["policy"] == "CnnPolicy":
        vec_env = VecFrameStack(vec_env, n_stack=4)
        eval_env = VecFrameStack(eval_env, n_stack=4)

    # Evaluates the agent periodically and saves the best version
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=20000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    # Saves intermediate checkpoints during training
    checkpoint_cb = CheckpointCallback(
        save_freq=50000,
        save_path=save_dir,
        name_prefix="dqn_ckpt",
    )

    # Initialize the DQN agent with experiment-specific hyperparameters
    model = DQN(
        policy=cfg["policy"],
        env=vec_env,
        learning_rate=cfg["lr"],
        gamma=cfg["gamma"],
        batch_size=cfg["batch_size"],
        exploration_fraction=cfg["exploration_fraction"],
        exploration_initial_eps=cfg["exploration_initial_eps"],
        exploration_final_eps=cfg["exploration_final_eps"],
        buffer_size=cfg["buffer_size"],
        learning_starts=10000,
        train_freq=4,
        target_update_interval=1000,
        verbose=1,
        tensorboard_log=log_dir,
    )

    # Train the agent
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[eval_cb, checkpoint_cb])

    # Save final trained model
    final_path = os.path.join(save_dir, "dqn_model_final")
    model.save(final_path)
    print(f"Saved: {final_path}.zip")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    if RUN_ALL:
        for cfg in EXPERIMENTS:
            run_experiment(cfg)
    else:
        run_experiment(EXPERIMENTS[ACTIVE_EXPERIMENT])
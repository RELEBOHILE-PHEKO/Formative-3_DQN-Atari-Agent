"""
train_limpho.py - DQN Agent Training Script
Environment: ALE/BankHeist-v5
Author: Limpho
"""

import os
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback


# Experiment configurations
# Each dictionary represents one hyperparameter setup
EXPERIMENTS = [
    # Exp 1 – Standard CNN with moderate settings
    {
        "name": "exp1_standard_cnn",
        "policy": "CnnPolicy",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch_size": 64,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
    },
    # Exp 2 – Very high learning rate with small batch size
    {
        "name": "exp2_very_high_lr_small_batch",
        "policy": "CnnPolicy",
        "lr": 1e-3,
        "gamma": 0.99,
        "batch_size": 16,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
    },
    # Exp 3 – Low gamma (focus on short-term rewards)
    {
        "name": "exp3_myopic_gamma",
        "policy": "CnnPolicy",
        "lr": 1e-4,
        "gamma": 0.80,
        "batch_size": 32,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
    },
    # Exp 4 – Very high gamma (focus on long-term rewards)
    {
        "name": "exp4_farsighted_gamma",
        "policy": "CnnPolicy",
        "lr": 1e-4,
        "gamma": 0.999,
        "batch_size": 32,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
    },
    # Exp 5 – Lower initial exploration
    {
        "name": "exp5_low_initial_eps",
        "policy": "CnnPolicy",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch_size": 32,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 0.5,
        "exploration_final_eps": 0.05,
    },
    # Exp 6 – Very low final exploration
    {
        "name": "exp6_near_zero_final_eps",
        "policy": "CnnPolicy",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch_size": 32,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.001,
    },
    # Exp 7 – Balanced configuration
    {
        "name": "exp7_balanced",
        "policy": "CnnPolicy",
        "lr": 3e-4,
        "gamma": 0.99,
        "batch_size": 64,
        "exploration_fraction": 0.15,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.02,
    },
    # Exp 8 – High learning rate, gamma, and batch size
    {
        "name": "exp8_high_all",
        "policy": "CnnPolicy",
        "lr": 5e-4,
        "gamma": 0.995,
        "batch_size": 128,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
    },
    # Exp 9 – MLP policy instead of CNN
    {
        "name": "exp9_mlp_tuned",
        "policy": "MlpPolicy",
        "lr": 3e-4,
        "gamma": 0.99,
        "batch_size": 64,
        "exploration_fraction": 0.2,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
    },
    # Exp 10 – Limpho’s best guess combination
    {
        "name": "exp10_limpho_best",
        "policy": "CnnPolicy",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch_size": 64,
        "exploration_fraction": 0.2,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.01,
    },
]

ACTIVE_EXPERIMENT = 0
RUN_ALL = True
TOTAL_TIMESTEPS = 500_000


# Creates environment instance depending on policy type
def make_env(policy):
    def _init():
        env = gym.make("ALE/BankHeist-v5", render_mode=None)
        if policy == "CnnPolicy":
            env = AtariWrapper(env)
        env = Monitor(env)
        return env
    return _init


# Runs a single training experiment
def run_experiment(cfg):
    print(f"Running {cfg['name']}")
    print(f"lr={cfg['lr']} gamma={cfg['gamma']} batch={cfg['batch_size']}")

    log_dir = f"logs/limpho/{cfg['name']}"
    save_dir = f"models/limpho/{cfg['name']}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    vec_env = DummyVecEnv([make_env(cfg["policy"])])
    if cfg["policy"] == "CnnPolicy":
        vec_env = VecFrameStack(vec_env, n_stack=4)

    eval_env = DummyVecEnv([make_env(cfg["policy"])])
    if cfg["policy"] == "CnnPolicy":
        eval_env = VecFrameStack(eval_env, n_stack=4)

    # Evaluates performance and saves best model
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=20000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    # Saves checkpoints during training
    checkpoint_cb = CheckpointCallback(
        save_freq=50000,
        save_path=save_dir,
        name_prefix="dqn_ckpt",
    )

    model = DQN(
        policy=cfg["policy"],
        env=vec_env,
        learning_rate=cfg["lr"],
        gamma=cfg["gamma"],
        batch_size=cfg["batch_size"],
        exploration_fraction=cfg["exploration_fraction"],
        exploration_initial_eps=cfg["exploration_initial_eps"],
        exploration_final_eps=cfg["exploration_final_eps"],
        buffer_size=100000,
        learning_starts=10000,
        train_freq=4,
        target_update_interval=1000,
        verbose=1,
        tensorboard_log=log_dir,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[eval_cb, checkpoint_cb])

    final_path = os.path.join(save_dir, "dqn_model_final")
    model.save(final_path)
    print(f"Model saved to {final_path}.zip")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    if RUN_ALL:
        for cfg in EXPERIMENTS:
            run_experiment(cfg)
    else:
        run_experiment(EXPERIMENTS[ACTIVE_EXPERIMENT])
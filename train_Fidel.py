"""
train_fidel.py - DQN Agent Training Script
Environment: ALE/BankHeist-v5
Author: Fidel
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
    # Exp 1 – High lr + low gamma
    {
        "name": "exp1_high_lr_low_gamma",
        "policy": "CnnPolicy",
        "lr": 5e-4,
        "gamma": 0.90,
        "batch_size": 32,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
    },
    # Exp 2 – Low lr + high gamma
    {
        "name": "exp2_low_lr_high_gamma",
        "policy": "CnnPolicy",
        "lr": 1e-5,
        "gamma": 0.995,
        "batch_size": 32,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
    },
    # Exp 3 – Large batch + high gamma
    {
        "name": "exp3_large_batch_high_gamma",
        "policy": "CnnPolicy",
        "lr": 1e-4,
        "gamma": 0.995,
        "batch_size": 128,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
    },
    # Exp 4 – Fast epsilon decay
    {
        "name": "exp4_fast_eps_decay",
        "policy": "CnnPolicy",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch_size": 32,
        "exploration_fraction": 0.05,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.01,
    },
    # Exp 5 – Slow epsilon decay
    {
        "name": "exp5_slow_eps_decay",
        "policy": "CnnPolicy",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch_size": 32,
        "exploration_fraction": 0.5,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
    },
    # Exp 6 – Medium learning rate and batch size
    {
        "name": "exp6_mid_lr_mid_batch",
        "policy": "CnnPolicy",
        "lr": 2e-4,
        "gamma": 0.99,
        "batch_size": 64,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
    },
    # Exp 7 – High final epsilon (more exploration retained)
    {
        "name": "exp7_high_final_eps",
        "policy": "CnnPolicy",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch_size": 32,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.3,
    },
    # Exp 8 – Very small batch size
    {
        "name": "exp8_tiny_batch",
        "policy": "CnnPolicy",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch_size": 8,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
    },
    # Exp 9 – Aggressive setup
    {
        "name": "exp9_aggressive",
        "policy": "CnnPolicy",
        "lr": 5e-4,
        "gamma": 0.99,
        "batch_size": 128,
        "exploration_fraction": 0.3,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.01,
    },
    # Exp 10 – Conservative setup
    {
        "name": "exp10_conservative",
        "policy": "CnnPolicy",
        "lr": 5e-5,
        "gamma": 0.99,
        "batch_size": 64,
        "exploration_fraction": 0.2,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
    },
]

ACTIVE_EXPERIMENT = 0
RUN_ALL = True
TOTAL_TIMESTEPS = 500_000


# Creates environment instance based on policy type
def make_env(policy):
    def _init():
        env = gym.make("ALE/BankHeist-v5", render_mode=None)
        if policy == "CnnPolicy":
            env = AtariWrapper(env)
        env = Monitor(env)
        return env
    return _init


# Runs training for one experiment configuration
def run_experiment(cfg):
    print(f"Running {cfg['name']}")
    print(f"lr={cfg['lr']} gamma={cfg['gamma']} batch={cfg['batch_size']}")

    log_dir = f"logs/fidel/{cfg['name']}"
    save_dir = f"models/fidel/{cfg['name']}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    vec_env = DummyVecEnv([make_env(cfg["policy"])])
    if cfg["policy"] == "CnnPolicy":
        vec_env = VecFrameStack(vec_env, n_stack=4)

    eval_env = DummyVecEnv([make_env(cfg["policy"])])
    if cfg["policy"] == "CnnPolicy":
        eval_env = VecFrameStack(eval_env, n_stack=4)

    # Evaluates and saves best model during training
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=20000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    # Saves periodic checkpoints
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

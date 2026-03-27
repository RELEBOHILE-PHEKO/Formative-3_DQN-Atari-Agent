"""
train_rele_last.py - Runs only exp9 and exp10
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


TOTAL_TIMESTEPS = 200_000

EXPERIMENTS = [
    # Exp 9 - MLP policy (reduced buffer_size to avoid memory crash)
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
    # Exp 10 - Tuned combination of parameters
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


def make_env(policy):
    def _init():
        env = gym.make("ALE/BankHeist-v5", render_mode=None)
        if policy == "CnnPolicy":
            env = AtariWrapper(env)
        env = Monitor(env)
        return env
    return _init


def run_experiment(cfg):
    print(f"\nRunning {cfg['name']}")
    print(f"lr={cfg['lr']} gamma={cfg['gamma']} batch={cfg['batch_size']} buffer={cfg['buffer_size']}")

    log_dir  = f"logs/rele/{cfg['name']}"
    save_dir = f"models/rele/{cfg['name']}"
    os.makedirs(log_dir,  exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    vec_env = DummyVecEnv([make_env(cfg["policy"])])
    if cfg["policy"] == "CnnPolicy":
        vec_env = VecFrameStack(vec_env, n_stack=4)

    eval_env = DummyVecEnv([make_env(cfg["policy"])])
    if cfg["policy"] == "CnnPolicy":
        eval_env = VecFrameStack(eval_env, n_stack=4)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=20000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )
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
        buffer_size=cfg["buffer_size"],
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
    for cfg in EXPERIMENTS:
        run_experiment(cfg)
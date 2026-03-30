import os
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback


# ✅ FAST + STRONG CONFIG
TOTAL_TIMESTEPS = 80_000   
MODEL_NAME = "final_model"


# Create environment
def make_env():
    def _init():
        env = gym.make("ALE/BankHeist-v5", render_mode=None)
        env = AtariWrapper(env)
        env = Monitor(env)
        return env
    return _init


def train():
    print("\n🚀 Training FINAL DQN Model (Fidel)\n")

    log_dir = "logs/final"
    save_dir = "models/final"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # ✅ Training environment
    vec_env = DummyVecEnv([make_env()])
    vec_env = VecFrameStack(vec_env, n_stack=4)
    vec_env = VecTransposeImage(vec_env)

    # ✅ Evaluation environment (MATCH EXACTLY)
    eval_env = DummyVecEnv([make_env()])
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_env = VecTransposeImage(eval_env)

    # ✅ Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=25_000,
        n_eval_episodes=3,
        deterministic=True,
        verbose=1,
    )

    # ✅ OPTIMIZED MODEL (FAST + MEMORY SAFE)
    model = DQN(
        policy="CnnPolicy",
        env=vec_env,
        learning_rate=2e-4,
        gamma=0.99,
        batch_size=64,
        buffer_size=20_000,        
        learning_starts=5_000,
        train_freq=8,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=0,
        tensorboard_log=log_dir,
    )

    # Train model
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback
    )

    # Save final model
    model.save(f"{save_dir}/{MODEL_NAME}")
    print(f"\n✅ Model saved at {save_dir}/{MODEL_NAME}.zip\n")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    train()
import os
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback


# These are the different experiments we run
# Each one tweaks hyperparameters to see what works best
EXPERIMENTS = [
    # Exp 1 – high learning rate, low discount (focus more on short-term rewards)
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

    # Exp 2 – slow learning, cares more about long-term rewards
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

    # Exp 3 – bigger batches, still long-term focused
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

    # Exp 4 – exploration drops quickly (agent commits early)
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

    # Exp 5 – exploration stays high for longer
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

    # Exp 6 – balanced setup (nothing extreme)
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

    # Exp 7 – keeps exploring even at the end
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

    # Exp 8 – try MLP instead of CNN (to compare performance)
    {
        "name": "exp8_mlp_policy",
        "policy": "MlpPolicy",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch_size": 32,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
    },

    # Exp 9 – more aggressive setup (faster learning, bigger batches, longer exploration)
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

    # Exp 10 – more conservative setup (slow learning, stable updates)
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


# Controls whether we run all experiments or just one
ACTIVE_EXPERIMENT = 0
RUN_ALL = True

# How long each experiment trains
TOTAL_TIMESTEPS = 200_000


def make_env(policy):
    def _init():
        env = gym.make("ALE/BankHeist-v5", render_mode=None)

        # Only use Atari preprocessing if we're using CNN
        if policy == "CnnPolicy":
            env = AtariWrapper(env)

        # Monitor helps track rewards and episode stats
        env = Monitor(env)
        return env

    return _init


def run_experiment(cfg):
    print(f"\n{'='*60}")
    print(f"Fidel | {cfg['name']}")
    print(f"lr={cfg['lr']}  gamma={cfg['gamma']}  batch={cfg['batch_size']}")
    print(f"eps: {cfg['exploration_initial_eps']} → {cfg['exploration_final_eps']} "
          f"(fraction={cfg['exploration_fraction']})")
    print(f"policy={cfg['policy']}")
    print(f"{'='*60}\n")

    log_dir = f"logs/fidel/{cfg['name']}"
    save_dir = f"models/fidel/{cfg['name']}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # Training environment
    vec_env = DummyVecEnv([make_env(cfg["policy"])])
    if cfg["policy"] == "CnnPolicy":
        vec_env = VecFrameStack(vec_env, n_stack=4)

    # Separate evaluation environment
    eval_env = DummyVecEnv([make_env(cfg["policy"])])
    if cfg["policy"] == "CnnPolicy":
        eval_env = VecFrameStack(eval_env, n_stack=4)

    # Saves best model based on evaluation performance
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=20_000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    # Saves checkpoints during training (just in case)
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
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
        buffer_size=100_000,
        learning_starts=10_000,
        train_freq=4,
        target_update_interval=1_000,
        verbose=1,
        tensorboard_log=log_dir,
    )

    # Train the agent
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_cb, checkpoint_cb]
    )

    # Save final model after training ends
    final_path = os.path.join(save_dir, "dqn_model_final")
    model.save(final_path)
    print(f"\nModel saved → {final_path}.zip\n")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    if RUN_ALL:
        for cfg in EXPERIMENTS:
            run_experiment(cfg)
    else:
        run_experiment(EXPERIMENTS[ACTIVE_EXPERIMENT])
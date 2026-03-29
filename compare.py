"""
compare_models.py - Compare all experiments and find the best model
Authors: Rele, Limpho
"""

import numpy as np
import os

EXPERIMENTS = [
    # Rele's experiments
    ("Rele",   "exp1_baseline",                "logs/rele/exp1_baseline/evaluations.npz"),
    ("Rele",   "exp2_high_lr",                 "logs/rele/exp2_high_lr/evaluations.npz"),
    ("Rele",   "exp3_low_lr",                  "logs/rele/exp3_low_lr/evaluations.npz"),
    ("Rele",   "exp4_low_gamma",               "logs/rele/exp4_low_gamma/evaluations.npz"),
    ("Rele",   "exp5_large_batch",             "logs/rele/exp5_large_batch/evaluations.npz"),
    ("Rele",   "exp6_small_batch",             "logs/rele/exp6_small_batch/evaluations.npz"),
    ("Rele",   "exp7_long_exploration",        "logs/rele/exp7_long_exploration/evaluations.npz"),
    ("Rele",   "exp8_high_final_eps",          "logs/rele/exp8_high_final_eps/evaluations.npz"),
    ("Rele",   "exp9_mlp_policy",              "logs/rele/exp9_mlp_policy/evaluations.npz"),
    ("Rele",   "exp10_best_combo",             "logs/rele/exp10_best_combo/evaluations.npz"),

    # Limpho's experiments
    ("Limpho", "exp1_standard_cnn",            "logs/limpho/exp1_standard_cnn/evaluations.npz"),
    ("Limpho", "exp2_very_high_lr_small_batch","logs/limpho/exp2_very_high_lr_small_batch/evaluations.npz"),
    ("Limpho", "exp3_myopic_gamma",            "logs/limpho/exp3_myopic_gamma/evaluations.npz"),
    ("Limpho", "exp4_farsighted_gamma",        "logs/limpho/exp4_farsighted_gamma/evaluations.npz"),
    ("Limpho", "exp5_low_initial_eps",         "logs/limpho/exp5_low_initial_eps/evaluations.npz"),
    ("Limpho", "exp6_near_zero_final_eps",     "logs/limpho/exp6_near_zero_final_eps/evaluations.npz"),
    ("Limpho", "exp7_balanced",                "logs/limpho/exp7_balanced/evaluations.npz"),
    ("Limpho", "exp8_high_all",                "logs/limpho/exp8_high_all/evaluations.npz"),
    ("Limpho", "exp9_mlp_tuned",               "logs/limpho/exp9_mlp_tuned/evaluations.npz"),
    ("Limpho", "exp10_limpho_best",            "logs/limpho/exp10_limpho_best/evaluations.npz"),
]

print("\n" + "="*65)
print(f"{'MEMBER':<10} {'EXPERIMENT':<35} {'BEST REWARD':>15}")
print("="*65)

results = []

for member, name, path in EXPERIMENTS:
    if not os.path.exists(path):
        print(f"{member:<10} {name:<35} {'NOT FOUND':>15}")
        continue
    try:
        data = np.load(path, allow_pickle=True)
        best_reward = data["results"].mean(axis=1).max()
        results.append((best_reward, member, name, path))
        print(f"{member:<10} {name:<35} {best_reward:>15.2f}")
    except Exception:
        # Fallback if evaluation file is corrupted
        model_path = path.replace("logs", "models").replace("evaluations.npz", "best_model.zip")
        if os.path.exists(model_path):
            print(f"{member:<10} {name:<35} {'LOG CORRUPTED - model exists':>15}")
        else:
            print(f"{member:<10} {name:<35} {'CORRUPTED':>15}")

print("="*65)

if results:
    results.sort(reverse=True)
    best_reward, best_member, best_name, best_path = results[0]
    best_model_path = best_path.replace("logs", "models").replace("evaluations.npz", "best_model.zip")

    print(f"\nBest Experiment: {best_member} — {best_name}")
    print(f"Best Evaluation Reward: {best_reward:.2f}")
    print(f"Model Path: {best_model_path}")
    print("\nRun the game with:")
    print(f"python play.py --model {best_model_path}\n")
else:
    print("\nNo valid results found.")
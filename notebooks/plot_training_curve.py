#!/usr/bin/env python
# coding: utf-8
"""
Plot PPO training curve from EvalCallback evaluations.npz log.
Run this after any training phase to see the reward curve.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ── Find the evaluations log ─────────────────────────────────
log_path = '../models/evaluations.npz'

if not os.path.exists(log_path):
    print(f"No evaluations.npz found at {log_path}")
    print("Make sure you ran phase4_ppo_training.py first.")
    exit()

data      = np.load(log_path)
timesteps = data['timesteps']
results   = data['results']          # shape: (n_evals, n_episodes)
mean_rew  = results.mean(axis=1)
std_rew   = results.std(axis=1)

total_steps = timesteps[-1]
phase       = total_steps // 4_000_000

print(f"Total timesteps logged: {total_steps:,}")
print(f"Training phase:         {phase}/3")
print(f"Eval points:            {len(timesteps)}")
print(f"Final mean reward:      {mean_rew[-1]:,.1f}")
print(f"Peak mean reward:       {mean_rew.max():,.1f} at step {timesteps[mean_rew.argmax()]:,}")

# ── Plot ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(timesteps, mean_rew,
        color='#16a34a', linewidth=2, label='Mean Reward')
ax.fill_between(timesteps,
                mean_rew - std_rew,
                mean_rew + std_rew,
                color='#16a34a', alpha=0.15, label='±1 Std Dev')
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.6)

# Mark phase boundaries
for p in [1, 2, 3]:
    boundary = p * 4_000_000
    if boundary <= total_steps:
        ax.axvline(x=boundary, color='#94a3b8', linestyle='--',
                   linewidth=1, alpha=0.7)
        ax.text(boundary, ax.get_ylim()[0], f' {p*4}M',
                color='#94a3b8', fontsize=9, va='bottom')

ax.set_xlabel('Timesteps', fontsize=12)
ax.set_ylabel('Mean Reward', fontsize=12)
ax.set_title(
    f'PPO Training Curve — Phase {phase}/3  '
    f'({total_steps/1_000_000:.0f}M steps total)\n'
    f'System: 650 kWh Battery | 132.5 kWp Solar | Average Ghana SREP Site',
    fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()

save_path = f'../data/training_curve_phase{phase}.png'
plt.savefig(save_path, dpi=150)
plt.show()
print(f"\nCurve saved: {save_path}")

# ── Trend check ───────────────────────────────────────────────
last_quarter = mean_rew[len(mean_rew)*3//4:]
trend = np.polyfit(range(len(last_quarter)), last_quarter, 1)[0]
print(f"\nTrend (last 25% of training): {trend:+.2f} reward/eval")
if trend > 50:
    print("Still trending UP — run next phase")
elif trend > 0:
    print("Slight upward trend — worth running next phase")
else:
    print("Flat/converged — consider stopping here")

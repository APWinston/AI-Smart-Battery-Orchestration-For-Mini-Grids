#!/usr/bin/env python
# coding: utf-8
"""
Phase 5 — Full Evaluation of PPO Agent
========================================
System: 1 x 650 kWh LiFePO4 | 132.5 kWp solar | ~1,318 people
        Solar/Load ratio: 1.51 | Autonomy: 24.0h

Evaluates the trained PPO agent against a rule-based baseline across
six locations — three seen during training (Tamale, Kumasi, Axim) and
three unseen (Accra, Bolgatanga, Akosombo) — over a fixed 6-year episode
(Jan 1 2020 → Jan 1 2026).

Metrics reported:
  - Load Served (%)         — reliability
  - LOLP (%)                — Loss of Load Probability
  - Total ENS (kWh)         — Energy Not Served
  - Final SOH               — State of Health after 6 years
  - SOH Year 1–6            — SOH at end of each calendar year
  - Battery Degradation/yr  — annualised SOH loss
  - Est. Lifespan (years)   — years to 80% SOH
  - EFC                     — Equivalent Full Cycles
  - Solar Curtailment (%)   — wasted solar
  - Mean SOC                — average state of charge
  - Mean Reward             — episodic reward

Outputs (saved to ../data/):
  - eval_results_summary.csv
  - eval_soc_soh_timeseries.png
  - eval_soh_per_year.png
  - eval_ens_timeseries.png
  - eval_metrics_comparison.png
  - eval_action_distribution.png
  - eval_daily_profiles.png
  - eval_generalisation_gap.png

Run after phase4_ppo_training.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from gymnasium import spaces
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================================
# PATHS  — adjust if your folder layout differs
# ============================================================
MODEL_PATH   = '../models/best_model'               # best eval checkpoint (EvalCallback)
VN_PATH      = '../models/vecnormalize_srep_avg_650kwh_4m.pkl'
LSTM_PATH    = '../models/best_lstm_scaled.pth'
DATA_PATH    = '../data/master_dataset_scaled.csv'
RAW_DATA_DIR = '../data'   # folder containing the unseen location CSVs
OUT_DIR      = '../data'
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# SYSTEM CONSTANTS  — must match training exactly
# ============================================================
BATTERY_KWH   = 650.0
USABLE_KWH    = BATTERY_KWH * 0.70
MEAN_LOAD_KW  = USABLE_KWH / 24              # 18.958 kW
MAX_RATE_KW   = BATTERY_KWH * 0.2            # 130 kW (0.2C)
SOLAR_AREA_M2 = 176.7
SOLAR_PEAK_KW = 132.5
SOLAR_NORM    = (1000.0 / 1000.0) * SOLAR_AREA_M2 * 0.75
LOAD_NORM     = MEAN_LOAD_KW * 2.0

feat_cols = ['ssrd_wm2', 'tp', 'temp_c', 'load_kw',
             'location_code', 'hour', 'month', 'dayofweek']

print("=" * 65)
print("  Phase 5 — PPO Evaluation | Average Ghana SREP Site")
print("=" * 65)
print(f"  Battery:      {BATTERY_KWH:.0f} kWh LiFePO4")
print(f"  Solar:        {SOLAR_PEAK_KW:.0f} kWp ({SOLAR_AREA_M2:.1f} m2)")
print(f"  Mean load:    {MEAN_LOAD_KW:.4f} kW (~1,318 people)")
print(f"  Solar/Load:   {((216.0/1000.0)*SOLAR_AREA_M2*0.75)/MEAN_LOAD_KW:.3f}")
print()

# ============================================================
# LOAD DATA
# ============================================================
print("Loading scaled dataset...")
df = pd.read_csv(DATA_PATH)
df['datetime']      = pd.to_datetime(df['datetime'])
df['location_code'] = df['location'].map({'Tamale': 0, 'Kumasi': 1, 'Axim': 2})
df['hour']          = df['datetime'].dt.hour
df['month']         = df['datetime'].dt.month
df['dayofweek']     = df['datetime'].dt.dayofweek

scaler_X = MinMaxScaler(); scaler_y = MinMaxScaler()
scaler_X.fit(df[feat_cols]); scaler_y.fit(df[['ssrd_wm2', 'load_kw']])

df_tamale = df[df['location'] == 'Tamale'].reset_index(drop=True)
df_kumasi  = df[df['location'] == 'Kumasi'].reset_index(drop=True)
df_axim    = df[df['location'] == 'Axim'].reset_index(drop=True)
print(f"Training data loaded: {df.shape}")

# ============================================================
# BUILD UNSEEN LOCATION DATAFRAMES
# Raw CSVs use:
#   ssrd  → J/m² accumulated → divide by 3600 to get W/m²
#   t2m   → Kelvin           → subtract 273.15 for °C
#   tp    → already in metres (kept as-is, same unit as training)
# Load is reused from the scaled training set (same SREP mean load).
# ============================================================
UNSEEN_LOCS   = ['Accra', 'Bolgatanga', 'Akosombo']
UPLOAD_DIR    = RAW_DATA_DIR   # raw ERA5 CSVs — configure RAW_DATA_DIR at the top of this file

# Use Tamale's load profile as the representative load for unseen sites
# (same scaled SREP load — the model was trained on this distribution)
_load_template = df_tamale[['datetime', 'load_kw']].copy()
_load_template['hour']      = _load_template['datetime'].dt.hour
_load_template['month']     = _load_template['datetime'].dt.month
_load_template['dayofweek'] = _load_template['datetime'].dt.dayofweek

def build_unseen_df(loc_name):
    sol = pd.read_csv(f'{UPLOAD_DIR}/Solar Irradiance {loc_name}.csv',
                      parse_dates=['valid_time'])
    tmp = pd.read_csv(f'{UPLOAD_DIR}/2m temperature {loc_name}.csv',
                      parse_dates=['valid_time'])
    pcp = pd.read_csv(f'{UPLOAD_DIR}/Precipitation {loc_name}.csv',
                      parse_dates=['valid_time'])

    merged = sol[['valid_time', 'ssrd']].merge(
             tmp[['valid_time', 't2m']], on='valid_time').merge(
             pcp[['valid_time', 'tp']],  on='valid_time')

    merged = merged.rename(columns={'valid_time': 'datetime'})
    merged['ssrd_wm2'] = merged['ssrd'] / 3600.0          # J/m² → W/m²
    merged['temp_c']   = merged['t2m']  - 273.15           # K → °C
    merged['location'] = loc_name

    # Align datetime index to load template (inner join on hour/month/dayofweek pattern)
    merged['hour']      = merged['datetime'].dt.hour
    merged['month']     = merged['datetime'].dt.month
    merged['dayofweek'] = merged['datetime'].dt.dayofweek

    # Match rows by position (same ERA5 hourly cadence as training data)
    n = min(len(merged), len(_load_template))
    merged = merged.iloc[:n].copy()
    merged['load_kw']      = _load_template['load_kw'].values[:n]
    merged['location_code'] = 3   # unseen — outside training distribution

    return merged[['datetime', 'location', 'ssrd_wm2', 'tp', 'temp_c',
                   'load_kw', 'location_code', 'hour', 'month', 'dayofweek']].reset_index(drop=True)

print("\nBuilding unseen location datasets...")
unseen_dfs = {}
for loc in UNSEEN_LOCS:
    try:
        unseen_dfs[loc] = build_unseen_df(loc)
        print(f"  {loc}: {unseen_dfs[loc].shape} | "
              f"Mean solar: {unseen_dfs[loc]['ssrd_wm2'].mean():.1f} W/m² | "
              f"Mean temp: {unseen_dfs[loc]['temp_c'].mean():.1f} °C")
    except FileNotFoundError as e:
        print(f"  WARNING: Could not load {loc} — {e}")

# All locations dict: trained + unseen
loc_dfs = {
    'Tamale (train)':     df_tamale,
    'Kumasi (train)':     df_kumasi,
    'Axim (train)':       df_axim,
}
for loc, loc_df in unseen_dfs.items():
    loc_dfs[f'{loc} (unseen)'] = loc_df

print(f"\nTotal locations to evaluate: {len(loc_dfs)}")

# ============================================================
# LSTM
# ============================================================
class MiniGridLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, forecast, output_size):
        super().__init__()
        self.hidden_size = hidden_size; self.num_layers = num_layers
        self.forecast = forecast; self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden_size, forecast * output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :]).view(-1, self.forecast, self.output_size)

lstm_model = MiniGridLSTM(8, 128, 2, 24, 2)
lstm_model.load_state_dict(torch.load(LSTM_PATH, map_location='cpu'))
lstm_model.eval()
print("LSTM loaded!")

# ============================================================
# ENVIRONMENT
# ============================================================
class SingleBatteryEnv(gym.Env):
    def __init__(self, loc_df, lstm, scaler_X, scaler_y):
        super().__init__()
        self.df = loc_df.reset_index(drop=True)
        self.lstm = lstm; self.scaler_X = scaler_X; self.scaler_y = scaler_y
        self.battery_kwh = BATTERY_KWH; self.max_rate_kw = MAX_RATE_KW
        self.charge_efficiency = 0.95; self.discharge_efficiency = 0.95
        self.soc_min = 0.20; self.soc_max = 0.90
        self.usable_range = self.soc_max - self.soc_min
        self.initial_soh = 1.0
        self.solar_area_m2 = SOLAR_AREA_M2
        self.episode_length = 52608  # exact hours Jan 1 2020 → Jan 1 2026 (incl. leap years 2020 & 2024)
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(52,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.soc = 0.5; self.soh = self.initial_soh
        self.efc = 0.0; self.ens = 0.0; self.current_step = 0
        self.daily_dod = 0.0
        self.rf_direction  = 0
        self.rf_half_start = 0.5
        self._prev_soc     = 0.5
        self.step_idx = 24   # always start from Jan 1 2020 00:00 (index 24 leaves 24h lookback)
        return self._get_obs(), {}

    def _get_forecast(self):
        lookback = self.df.iloc[self.step_idx - 24:self.step_idx]
        X = self.scaler_X.transform(lookback[feat_cols].values)
        X_t = torch.FloatTensor(X).unsqueeze(0)
        with torch.no_grad():
            fc = self.lstm(X_t).numpy()[0]
        fc_inv = self.scaler_y.inverse_transform(fc.reshape(-1, 2))
        return np.clip(fc_inv[:, 0], 0, None), np.clip(fc_inv[:, 1], 0, None)

    def _get_obs(self):
        sf, lf = self._get_forecast()
        row = self.df.iloc[self.step_idx]
        return np.concatenate([
            [self.soc, self.soh],
            sf / SOLAR_NORM, lf / LOAD_NORM,
            [row['hour'] / 23.0, row['month'] / 12.0]
        ]).astype(np.float32)

    def _compute_degradation(self, temp_c, action):
        soc_change = self.soc - self._prev_soc
        deg_cycle  = 0.0
        if abs(soc_change) > 1e-4:
            new_dir = 1 if soc_change > 0 else -1
            if new_dir != self.rf_direction and self.rf_direction != 0:
                half_dod = abs(self.soc - self.rf_half_start) / self.usable_range
                if half_dod > 1e-4:
                    self.efc      += half_dod * 0.5
                    cycle_life     = 3500.0 * (1.0 / max(half_dod, 0.01)) ** 1.5
                    deg_cycle      = (0.20 / cycle_life) * 0.5
                    self.daily_dod = half_dod
                self.rf_half_start = self.soc
            self.rf_direction = new_dir
        self._prev_soc = self.soc
        cell_temp  = temp_c + 5.0 + abs(action) * 3.0
        T_kelvin   = max(cell_temp + 273.15, 273.15)
        k_cal      = 14876.0 * np.exp(-24500.0 / (8.314 * T_kelvin))
        if self.soc >= 0.50:
            soc_stress = 0.70 + 0.60 * (self.soc - 0.50) ** 2
        else:
            soc_stress = 0.70 + 0.30 * (self.soc - 0.50) ** 2
        deg_cal     = 1.423e-6 * k_cal * soc_stress
        deg_low_soc = 2e-4 * max(0.0, 0.30 - self.soc) ** 2
        return deg_cycle + deg_cal + deg_low_soc

    def step(self, action):
        action = float(np.clip(action[0], -1.0, 1.0))
        row = self.df.iloc[self.step_idx]
        solar_kw = (row['ssrd_wm2'] / 1000.0) * self.solar_area_m2 * 0.75
        load_kw  = row['load_kw']; soc_before = self.soc
        residual_load = max(0.0, load_kw - solar_kw)
        solar_surplus = max(0.0, solar_kw - load_kw)
        if action > 0:
            ckw = min(action * self.max_rate_kw,
                      (self.soc_max - self.soc) * self.battery_kwh)
            self.soc += (ckw * self.charge_efficiency) / self.battery_kwh
            net_load = residual_load; curtailed_kw = max(0.0, solar_surplus - ckw)
        else:
            dkw = min(abs(action) * self.max_rate_kw,
                      (self.soc - self.soc_min) * self.battery_kwh)
            self.soc -= dkw / self.battery_kwh
            net_load = max(0.0, residual_load - dkw * self.discharge_efficiency)
            curtailed_kw = solar_surplus
        self.soc = float(np.clip(self.soc, self.soc_min, self.soc_max))
        ens_step = min(max(0.0, net_load), load_kw); self.ens += ens_step
        cycle_depth = abs(self.soc - soc_before)
        self.soh = max(0.0, self.soh - self._compute_degradation(row['temp_c'], action))

        ens_ratio    = ens_step / max(load_kw, 1.0)
        curtail_ratio = curtailed_kw / max(solar_kw, 1.0)
        served_ratio  = 1.0 - ens_ratio
        reward  =  served_ratio * 2.5
        reward -= (ens_step > 0) * 3.5
        reward -= ens_ratio * 4.0
        reward += max(0.0, self.soc - 0.50) * served_ratio * 2.5
        reward += max(0.0, self.soh - 0.85) * 1.5
        if self.current_step % 24 == 23 and self.daily_dod > 0.60:
            reward -= (self.daily_dod - 0.60) * 3.0
        reward -= (self.soc > 0.85) * 2.0
        reward -= (self.soc < 0.25) * 2.0
        reward -= (self.soc < 0.20) * 10.0
        reward -= cycle_depth * (1.0 - served_ratio) * 2.0
        reward -= curtail_ratio * 0.5
        reward = np.clip(reward, -15.0, 5.0)

        self.step_idx += 1; self.current_step += 1
        done = self.current_step >= self.episode_length
        next_obs = self._get_obs() if not done else np.zeros(52, dtype=np.float32)
        return next_obs, reward, done, False, {
            'soc': self.soc, 'soh': self.soh, 'efc': self.efc,
            'ens': ens_step, 'solar_kw': solar_kw, 'load_kw': load_kw,
            'action': action, 'curtailed_kw': curtailed_kw,
        }

# ============================================================
# RULE-BASED CONTROLLER
# ============================================================
def rule_based_action(soc, solar_kw, load_kw):
    net = solar_kw - load_kw
    if net > 0:
        return np.array([min(net / MAX_RATE_KW, 1.0)]) if soc < 0.90 else np.array([0.0])
    else:
        return np.array([-1.0]) if soc > 0.20 else np.array([0.0])

# ============================================================
# EPISODE RUNNER — returns dict of per-step lists + summary metrics
# ============================================================
# Precompute actual hourly boundaries for each year-end in the dataset.
# Training data starts at index 0 = 2020-01-01 00:00; step_idx=24 is first step.
# We need the index offset from step 0 (index 24) to end of each year.
_START_DATE = pd.Timestamp('2020-01-01 00:00')
_YEAR_END_OFFSETS = {}   # year -> number of steps from episode start (step_idx=24)
for _yr in range(1, 7):
    _end = pd.Timestamp(f'{2019 + _yr}-12-31 23:00')
    _delta = int((_end - _START_DATE).total_seconds() / 3600) + 1
    _YEAR_END_OFFSETS[_yr] = _delta   # e.g. 2020 has 8784h (leap), 2021 has 8760h

def yearly_soh(soh_array):
    """Return SOH at end of each calendar year (2020–2025), leap-year aware."""
    result = {}
    cumulative = 0
    for y in range(1, 7):
        cumulative = _YEAR_END_OFFSETS[y]
        idx = min(cumulative, len(soh_array)) - 1
        result[f'soh_year{y}'] = float(soh_array[idx])
    return result

def run_episode(env, policy_fn):
    """
    Run one full episode.
    policy_fn(obs, env) -> action array
    Returns per-step logs and a summary dict.
    """
    obs, _ = env.reset()
    done = False
    step = 0
    logs = {'soc': [], 'soh': [], 'ens': [], 'solar_kw': [],
            'load_kw': [], 'action': [], 'curtailed_kw': [], 'reward': [], 'efc': []}

    while not done:
        action = policy_fn(obs, env)
        obs, reward, done, _, info = env.step(action)
        for k in ['soc', 'soh', 'ens', 'solar_kw', 'load_kw', 'action', 'curtailed_kw']:
            logs[k].append(info[k])
        logs['reward'].append(reward)
        logs['efc'].append(env.efc)   # cumulative EFC at this step
        step += 1
        if step in _YEAR_END_OFFSETS.values():
            yr = next(y for y, v in _YEAR_END_OFFSETS.items() if v == step)
            print(f"      [Rule-Based] Year {yr}/6 | SOC:{info['soc']*100:.1f}% | "
                  f"SOH:{info['soh']*100:.2f}% | "
                  f"ENS so far: {sum(logs['ens']):.1f} kWh")

    logs = {k: np.array(v) for k, v in logs.items()}
    total_load  = logs['load_kw'].sum()
    total_solar = logs['solar_kw'].sum()
    total_ens   = logs['ens'].sum()

    summary = {
        'load_served_pct':    (1 - total_ens / max(total_load, 1)) * 100,
        'lolp_pct':           np.mean(logs['ens'] > 0) * 100,
        'total_ens_kwh':      total_ens,
        'final_soh':          logs['soh'][-1],
        'deg_per_year':       (1.0 - logs['soh'][-1]) / 6,
        'lifespan_years':     (0.20 / max((1.0 - logs['soh'][-1]) / 6, 1e-9)) if (1.0 - logs['soh'][-1]) > 0 else 99.9,
        'efc':                env.efc,
        'curtail_pct':        (logs['curtailed_kw'].sum() / max(total_solar, 1)) * 100,
        'mean_soc':           logs['soc'].mean(),
        'mean_reward':        logs['reward'].mean(),
        'total_steps':        len(logs['soc']),
        **yearly_soh(logs['soh']),   # soh_year1 … soh_year6
    }
    return logs, summary

# ============================================================
# LOAD PPO MODEL
# ============================================================
print("\nLoading PPO model...")
_raw_eval = DummyVecEnv([lambda: Monitor(SingleBatteryEnv(df_tamale, lstm_model, scaler_X, scaler_y))])
eval_env_vec = VecNormalize.load(VN_PATH, _raw_eval)
eval_env_vec.training = False
eval_env_vec.norm_reward = False

ppo_model = PPO.load(MODEL_PATH, env=eval_env_vec, device='cpu')
print("PPO model loaded!")

# ============================================================
# EVALUATION LOOP
# ============================================================
print(f"\nRunning 6-year evaluation — {len(loc_dfs)} locations × 2 controllers...")
print(f"  Training locations: Tamale, Kumasi, Axim")
print(f"  Unseen  locations:  {', '.join(UNSEEN_LOCS)}")
print("-" * 65)

all_results = []
last_logs   = {}

for loc_name, loc_df in loc_dfs.items():
    for controller in ['PPO', 'Rule-Based']:

        if controller == 'PPO':
            _raw = DummyVecEnv([lambda loc=loc_df: Monitor(
                SingleBatteryEnv(loc, lstm_model, scaler_X, scaler_y))])
            _vec = VecNormalize.load(VN_PATH, _raw)
            _vec.training = False; _vec.norm_reward = False
            inner_env = _vec.envs[0].unwrapped

            obs_raw, _ = inner_env.reset()
            done = False
            step = 0
            logs = {'soc': [], 'soh': [], 'ens': [], 'solar_kw': [],
                    'load_kw': [], 'action': [], 'curtailed_kw': [], 'reward': [], 'efc': []}

            while not done:
                obs_norm = _vec.normalize_obs(obs_raw.reshape(1, -1))[0]
                action, _ = ppo_model.predict(obs_norm, deterministic=True)
                action = action.flatten()
                obs_raw, reward, done, _, info = inner_env.step(action)
                for k in ['soc', 'soh', 'ens', 'solar_kw', 'load_kw', 'action', 'curtailed_kw']:
                    logs[k].append(info[k])
                logs['reward'].append(reward)
                logs['efc'].append(inner_env.efc)   # cumulative EFC at this step
                step += 1
                if step in _YEAR_END_OFFSETS.values():
                    yr = next(y for y, v in _YEAR_END_OFFSETS.items() if v == step)
                    print(f"      [PPO | {loc_name}] Year {yr}/6 | "
                          f"SOC:{info['soc']*100:.1f}% | SOH:{info['soh']*100:.2f}% | "
                          f"ENS so far: {sum(logs['ens']):.1f} kWh")

            logs = {k: np.array(v) for k, v in logs.items()}
            total_load  = logs['load_kw'].sum()
            total_solar = logs['solar_kw'].sum()
            total_ens   = logs['ens'].sum()
            summary = {
                'load_served_pct':  (1 - total_ens / max(total_load, 1)) * 100,
                'lolp_pct':         np.mean(logs['ens'] > 0) * 100,
                'total_ens_kwh':    total_ens,
                'final_soh':        logs['soh'][-1],
                'deg_per_year':     (1.0 - logs['soh'][-1]) / 6,
                'lifespan_years':   (0.20 / max((1.0 - logs['soh'][-1]) / 6, 1e-9)),
                'efc':              inner_env.efc,
                'curtail_pct':      (logs['curtailed_kw'].sum() / max(total_solar, 1)) * 100,
                'mean_soc':         logs['soc'].mean(),
                'mean_reward':      logs['reward'].mean(),
                'total_steps':      len(logs['soc']),
                **yearly_soh(logs['soh']),
            }

        else:  # Rule-Based
            env = SingleBatteryEnv(loc_df, lstm_model, scaler_X, scaler_y)
            def rb_policy(obs, env):
                row   = env.df.iloc[env.step_idx]
                solar = (row['ssrd_wm2'] / 1000.0) * SOLAR_AREA_M2 * 0.75
                return rule_based_action(env.soc, solar, row['load_kw'])
            logs, summary = run_episode(env, rb_policy)

        last_logs[(loc_name, controller)] = logs
        avg = dict(summary)
        avg['location']   = loc_name
        avg['controller'] = controller
        all_results.append(avg)

        print(f"  {loc_name} | {controller:10s} | "
              f"Served: {avg['load_served_pct']:.1f}% | "
              f"LOLP: {avg['lolp_pct']:.1f}% | "
              f"SOH: {avg['final_soh']:.4f} | "
              f"Lifespan: {avg['lifespan_years']:.1f} yr")

# ============================================================
# RESULTS TABLE
# ============================================================
results_df = pd.DataFrame(all_results)
results_df['split'] = results_df['location'].apply(
    lambda x: 'unseen' if '(unseen)' in x else 'train')
soh_year_cols = [f'soh_year{y}' for y in range(1, 7)]
cols_order = ['location', 'split', 'controller', 'load_served_pct', 'lolp_pct',
              'total_ens_kwh', 'final_soh', 'deg_per_year', 'lifespan_years',
              'efc', 'curtail_pct', 'mean_soc', 'mean_reward', 'total_steps'] + soh_year_cols
results_df = results_df[cols_order]

csv_path = os.path.join(OUT_DIR, 'eval_results_summary.csv')
results_df.to_csv(csv_path, index=False)
print(f"\nResults saved: {csv_path}")

# ============================================================
# PER-YEAR KPI TABLE  — eval_results_by_year.csv
# ============================================================
# Calendar year boundaries (leap-year aware), same as _YEAR_END_OFFSETS.
# Each year slice is [prev_end, year_end) in step indices.
_YEAR_BOUNDARIES = {}  # year -> (start_step, end_step) inclusive, 0-indexed into logs
_prev = 0
for _yr in range(1, 7):
    _end = _YEAR_END_OFFSETS[_yr]          # cumulative steps from episode start
    _YEAR_BOUNDARIES[_yr] = (_prev, _end)  # [start, end)
    _prev = _end

YEAR_LABELS = {1: '2020', 2: '2021', 3: '2022', 4: '2023', 5: '2024', 6: '2025'}

def compute_yearly_kpis(logs, efc_start, soh_start):
    """
    Given full-episode logs (numpy arrays) and the cumulative EFC / starting SOH
    at the beginning of the episode, return a list of one dict per year with all
    KPIs the app needs to display.

    KPIs per year:
        year, year_label,
        load_served_pct, lolp_pct, total_ens_kwh,
        final_soh, soh_start (SOH at start of that year),
        deg_this_year (SOH loss within the year),
        lifespan_years (projected from this year's degradation rate),
        efc_this_year (EFCs accumulated this year),
        curtail_pct, mean_soc, soc_std,
        mean_reward, total_hours
    """
    yearly_rows = []
    # Track cumulative EFC so we can compute per-year EFC delta
    # efc is stored per-step in logs['efc'] if present, else we can't compute it
    # We use logs['efc'] which is the cumulative total at each step
    has_efc = 'efc' in logs and len(logs.get('efc', [])) == len(logs['soc'])

    for yr in range(1, 7):
        s, e = _YEAR_BOUNDARIES[yr]
        e    = min(e, len(logs['soc']))   # clamp to actual episode length
        if s >= e:
            continue

        sl_ens   = logs['ens'][s:e]
        sl_soc   = logs['soc'][s:e]
        sl_soh   = logs['soh'][s:e]
        sl_solar = logs['solar_kw'][s:e]
        sl_load  = logs['load_kw'][s:e]
        sl_curt  = logs['curtailed_kw'][s:e]
        sl_rew   = logs['reward'][s:e]

        total_load  = sl_load.sum()
        total_solar = sl_solar.sum()
        total_ens   = sl_ens.sum()
        total_curt  = sl_curt.sum()

        soh_at_start = float(logs['soh'][s - 1]) if s > 0 else soh_start
        soh_at_end   = float(sl_soh[-1])
        deg_this_yr  = soh_at_start - soh_at_end

        # Projected lifespan from this year's degradation rate
        if deg_this_yr > 1e-9:
            lifespan = round(0.20 / deg_this_yr, 1)
        else:
            lifespan = 99.9

        # EFC accumulated this year
        if has_efc:
            efc_end   = float(logs['efc'][e - 1])
            efc_begin = float(logs['efc'][s - 1]) if s > 0 else 0.0
            efc_yr    = efc_end - efc_begin
        else:
            efc_yr = float('nan')

        # SCR ≈ inverse of curtailment rate
        curtail_pct = (total_curt / max(total_solar, 1)) * 100
        scr         = min((1.0 - curtail_pct / 100.0) * 100.0, 100.0)

        yearly_rows.append({
            'year':            yr,
            'year_label':      YEAR_LABELS[yr],
            'total_hours':     int(e - s),
            'load_served_pct': round((1 - total_ens / max(total_load, 1)) * 100, 4),
            'lolp_pct':        round(float(np.mean(sl_ens > 0)) * 100, 4),
            'total_ens_kwh':   round(float(total_ens), 2),
            'soh_start':       round(soh_at_start, 6),
            'final_soh':       round(soh_at_end, 6),
            'deg_this_year':   round(deg_this_yr, 6),
            'lifespan_years':  lifespan,
            'efc_this_year':   round(float(efc_yr), 2),
            'curtail_pct':     round(float(curtail_pct), 4),
            'scr_pct':         round(float(scr), 4),
            'mean_soc':        round(float(sl_soc.mean()), 6),
            'soc_std':         round(float(sl_soc.std()), 6),
            'mean_reward':     round(float(sl_rew.mean()), 4),
        })
    return yearly_rows


print("\nComputing per-year KPIs...")
yearly_results = []

for (loc_name, controller), logs in last_logs.items():
    split = 'unseen' if '(unseen)' in loc_name else 'train'
    rows  = compute_yearly_kpis(logs, efc_start=0.0, soh_start=1.0)
    for row in rows:
        row['location']   = loc_name
        row['split']      = split
        row['controller'] = controller
        yearly_results.append(row)

yearly_df = pd.DataFrame(yearly_results)

# Column order
yr_cols = ['location', 'split', 'controller', 'year', 'year_label', 'total_hours',
           'load_served_pct', 'lolp_pct', 'total_ens_kwh',
           'soh_start', 'final_soh', 'deg_this_year', 'lifespan_years',
           'efc_this_year', 'curtail_pct', 'scr_pct',
           'mean_soc', 'soc_std', 'mean_reward']
yearly_df = yearly_df[[c for c in yr_cols if c in yearly_df.columns]]

yearly_csv_path = os.path.join(OUT_DIR, 'eval_results_by_year.csv')
yearly_df.to_csv(yearly_csv_path, index=False)
print(f"Per-year results saved: {yearly_csv_path}")
print(f"  Shape: {yearly_df.shape}  ({len(last_logs)} location×controller combos × up to 6 years)")

# Pretty-print per-year table for Tamale as a sanity check
print("\n" + "=" * 110)
print("  PER-YEAR KPI SAMPLE — Tamale")
print("=" * 110)
tamale_yr = yearly_df[yearly_df['location'] == 'Tamale (train)']
yr_hdr2 = (f"{'Ctrl':<12} {'Yr':>4} {'Served%':>8} {'LOLP%':>7} {'ENS kWh':>10} "
           f"{'SOH_end':>8} {'Deg/yr':>8} {'Life':>6} {'EFC':>7} {'Curt%':>7} {'MeanSOC':>8} {'SOC_std':>8}")
print(yr_hdr2); print("-" * 110)
for _, row in tamale_yr.iterrows():
    print(f"{row['controller']:<12} {row['year']:>4} {row['load_served_pct']:>8.2f} "
          f"{row['lolp_pct']:>7.2f} {row['total_ens_kwh']:>10.1f} "
          f"{row['final_soh']:>8.4f} {row['deg_this_year']:>8.4f} "
          f"{row['lifespan_years']:>6.1f} {row['efc_this_year']:>7.1f} "
          f"{row['curtail_pct']:>7.2f} {row['mean_soc']:>8.4f} {row['soc_std']:>8.4f}")
print("=" * 110)

# Pretty-print summary — training then unseen
print("\n" + "=" * 88)
print("  EVALUATION SUMMARY")
print("=" * 88)
hdr = (f"{'Location':<22} {'Split':<8} {'Controller':<12} {'Served%':>8} {'LOLP%':>7} "
       f"{'SOH':>7} {'Life(yr)':>9} {'EFC':>6} {'Curtail%':>9}")
print(hdr); print("-" * 88)
for split_label in ['train', 'unseen']:
    sub = results_df[results_df['split'] == split_label]
    for _, row in sub.iterrows():
        print(f"{row['location']:<22} {row['split']:<8} {row['controller']:<12} "
              f"{row['load_served_pct']:>8.2f} {row['lolp_pct']:>7.2f} "
              f"{row['final_soh']:>7.4f} {row['lifespan_years']:>9.1f} "
              f"{row['efc']:>6.1f} {row['curtail_pct']:>9.2f}")
    print("-" * 88)
print("=" * 88)

# Per-year SOH breakdown
print("\n" + "=" * 88)
print("  STATE OF HEALTH — END OF EACH YEAR")
print("=" * 88)
yr_hdr = f"{'Location':<22} {'Controller':<12}" + "".join(f"{'Yr'+str(y):>9}" for y in range(1, 7))
print(yr_hdr); print("-" * 88)
for split_label in ['train', 'unseen']:
    sub = results_df[results_df['split'] == split_label]
    for _, row in sub.iterrows():
        yr_vals = "".join(f"{row[f'soh_year{y}']:>9.4f}" for y in range(1, 7))
        print(f"{row['location']:<22} {row['controller']:<12}{yr_vals}")
    print("-" * 88)
print("=" * 88)

# ============================================================
# PLOT 1 — SOC & SOH Time-Series (Tamale train vs Accra unseen)
# ============================================================
SHOW = 24 * 30
fig, axes = plt.subplots(2, 2, figsize=(16, 8))
fig.suptitle('SOC & SOH — Tamale (train) vs Accra (unseen), PPO, First 30 Days', fontsize=13)

for col, loc_key in enumerate(['Tamale (train)', 'Accra (unseen)']):
    key = (loc_key, 'PPO')
    if key not in last_logs:
        axes[0, col].set_visible(False); axes[1, col].set_visible(False); continue
    lg = last_logs[key]
    axes[0, col].plot(lg['soc'][:SHOW], color='#2563eb', linewidth=0.8)
    axes[0, col].axhline(0.50, color='green',  linestyle='--', linewidth=1, alpha=0.7, label='SOC 50%')
    axes[0, col].axhline(0.20, color='red',    linestyle='--', linewidth=1, alpha=0.7, label='Min 20%')
    axes[0, col].axhline(0.90, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Max 90%')
    axes[0, col].set_title(f'{loc_key} — SOC', fontsize=11)
    axes[0, col].set_ylabel('SOC'); axes[0, col].set_ylim(0, 1)
    axes[0, col].legend(fontsize=8); axes[0, col].grid(True, alpha=0.3)

    axes[1, col].plot(lg['soh'][:SHOW], color='#16a34a', linewidth=0.8)
    axes[1, col].axhline(0.85, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='SOH 85%')
    axes[1, col].set_title(f'{loc_key} — SOH', fontsize=11)
    axes[1, col].set_ylabel('SOH')
    soh_min = max(0.0, lg['soh'][:SHOW].min() - 0.002)
    axes[1, col].set_ylim(soh_min, 1.001)
    axes[1, col].legend(fontsize=8); axes[1, col].grid(True, alpha=0.3)

plt.tight_layout()
p1 = os.path.join(OUT_DIR, 'eval_soc_soh_timeseries.png')
plt.savefig(p1, dpi=150); plt.show(); print(f"Saved: {p1}")

# ============================================================
# PLOT 1b — SOH at End of Each Year, all locations & controllers
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('State of Health (SOH) at End of Each Year — All Locations', fontsize=13)
years = list(range(1, 7))
year_labels = ['End 2020', 'End 2021', 'End 2022', 'End 2023', 'End 2024', 'End 2025']

colors_ctrl = {'PPO': '#2563eb', 'Rule-Based': '#ef4444'}
styles_split = {'train': '-', 'unseen': '--'}
markers_split = {'train': 'o', 'unseen': 's'}

for ax, split_label in zip(axes, ['train', 'unseen']):
    sub = results_df[results_df['split'] == split_label]
    for _, row in sub.iterrows():
        soh_vals = [row[f'soh_year{y}'] for y in years]
        ax.plot(year_labels, soh_vals,
                color=colors_ctrl[row['controller']],
                linestyle=styles_split[split_label],
                marker=markers_split[split_label],
                linewidth=2, markersize=6,
                label=f"{row['location']} — {row['controller']}")
        ax.annotate(f"{soh_vals[-1]:.4f}",
                    xy=(year_labels[-1], soh_vals[-1]),
                    xytext=(5, 0), textcoords='offset points',
                    fontsize=7, color=colors_ctrl[row['controller']])
    ax.axhline(0.80, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='EOL (80% SOH)')
    ax.set_title(f"{'Training' if split_label == 'train' else 'Unseen'} Locations", fontsize=12)
    ax.set_xlabel('Year End'); ax.set_ylabel('SOH')
    ax.set_ylim(bottom=min(0.75, results_df[[f'soh_year{y}' for y in years]].min().min() - 0.01))
    ax.legend(fontsize=7, loc='lower left'); ax.grid(True, alpha=0.3)

plt.tight_layout()
p1b = os.path.join(OUT_DIR, 'eval_soh_per_year.png')
plt.savefig(p1b, dpi=150); plt.show(); print(f"Saved: {p1b}")

# ============================================================
# PLOT 2 — ENS Time-Series (all 6 locations, first 30 days)
# ============================================================
all_loc_keys = list(loc_dfs.keys())
n_locs = len(all_loc_keys)
fig, axes = plt.subplots(n_locs, 1, figsize=(15, 3 * n_locs))
fig.suptitle('Energy Not Served (ENS) — First 30 Days per Location', fontsize=14)

for row_i, loc_key in enumerate(all_loc_keys):
    ax = axes[row_i] if n_locs > 1 else axes
    for ctrl, color in [('Rule-Based', '#ef4444'), ('PPO', '#2563eb')]:
        key = (loc_key, ctrl)
        if key in last_logs:
            ax.plot(last_logs[key]['ens'][:SHOW],
                    color=color, linewidth=0.7, alpha=0.85, label=ctrl)
    split_tag = '★ unseen' if 'unseen' in loc_key else 'train'
    ax.set_title(f'{loc_key}  [{split_tag}]', fontsize=10)
    ax.set_ylabel('ENS (kWh)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Hour') if n_locs > 1 else axes.set_xlabel('Hour')
plt.tight_layout()
p2 = os.path.join(OUT_DIR, 'eval_ens_timeseries.png')
plt.savefig(p2, dpi=150); plt.show(); print(f"Saved: {p2}")

# ============================================================
# PLOT 3 — Metrics Bar Comparison (train avg vs unseen avg, PPO only)
# ============================================================
metrics_plot = {
    'Load Served (%)':    'load_served_pct',
    'LOLP (%)':           'lolp_pct',
    'Final SOH':          'final_soh',
    'Lifespan (years)':   'lifespan_years',
    'Solar Curtailed (%)':'curtail_pct',
    'Mean SOC':           'mean_soc',
}
ppo_df     = results_df[results_df['controller'] == 'PPO']
ppo_train  = ppo_df[ppo_df['split'] == 'train'][list(metrics_plot.values())].mean()
ppo_unseen = ppo_df[ppo_df['split'] == 'unseen'][list(metrics_plot.values())].mean()
rb_df      = results_df[results_df['controller'] == 'Rule-Based']
rb_all     = rb_df[list(metrics_plot.values())].mean()

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('PPO: Train vs Unseen Locations | vs Rule-Based Baseline', fontsize=13)
axes = axes.flatten()
bar_labels = ['PPO (train)', 'PPO (unseen)', 'Rule-Based']
bar_colors = ['#2563eb', '#7c3aed', '#ef4444']

for i, (label, col) in enumerate(metrics_plot.items()):
    ax = axes[i]
    vals = [ppo_train[col], ppo_unseen[col], rb_all[col]]
    bars = ax.bar(bar_labels, vals, color=bar_colors, alpha=0.85, edgecolor='white')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title(label, fontsize=11)
    ax.tick_params(axis='x', labelsize=8)
    ax.grid(True, alpha=0.3, axis='y'); ax.set_ylim(bottom=0)

plt.tight_layout()
p3 = os.path.join(OUT_DIR, 'eval_metrics_comparison.png')
plt.savefig(p3, dpi=150); plt.show(); print(f"Saved: {p3}")

# ============================================================
# PLOT 4 — Action Distribution (Tamale train vs Accra unseen, PPO)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('PPO Action Distribution — Tamale (train) vs Accra (unseen)', fontsize=13)

for col, loc_key in enumerate(['Tamale (train)', 'Accra (unseen)']):
    key = (loc_key, 'PPO')
    if key not in last_logs:
        axes[col].set_visible(False); continue
    acts = last_logs[key]['action']
    axes[col].hist(acts, bins=60, color='#2563eb' if col == 0 else '#7c3aed',
                   alpha=0.8, edgecolor='white')
    axes[col].axvline(0, color='black', linewidth=1.5, linestyle='--')
    axes[col].set_title(f'PPO — {loc_key}', fontsize=11)
    axes[col].set_xlabel('Action (charge +, discharge -)'); axes[col].set_ylabel('Count')
    axes[col].grid(True, alpha=0.3)
    charge_pct = np.mean(acts > 0.05) * 100
    idle_pct   = np.mean(np.abs(acts) <= 0.05) * 100
    disch_pct  = np.mean(acts < -0.05) * 100
    axes[col].text(0.02, 0.97,
                   f'Charging: {charge_pct:.1f}%\nIdle: {idle_pct:.1f}%\nDischarging: {disch_pct:.1f}%',
                   transform=axes[col].transAxes, va='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
p4 = os.path.join(OUT_DIR, 'eval_action_distribution.png')
plt.savefig(p4, dpi=150); plt.show(); print(f"Saved: {p4}")

# ============================================================
# PLOT 5 — Average Daily Profiles (SOC & ENS by hour, Tamale)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Average Hourly Profiles — Tamale (train, Full Episode)', fontsize=13)

for ctrl, color in [('Rule-Based', '#ef4444'), ('PPO', '#2563eb')]:
    key = ('Tamale (train)', ctrl)
    if key not in last_logs: continue
    lg = last_logs[key]
    hours = np.arange(len(lg['soc'])) % 24
    soc_by_hour = [lg['soc'][hours == h].mean() for h in range(24)]
    ens_by_hour = [lg['ens'][hours == h].mean() for h in range(24)]
    axes[0].plot(range(24), soc_by_hour, color=color, linewidth=2, label=ctrl, marker='o', markersize=3)
    axes[1].plot(range(24), ens_by_hour, color=color, linewidth=2, label=ctrl, marker='o', markersize=3)

axes[0].set_title('Mean SOC by Hour of Day', fontsize=12)
axes[0].set_xlabel('Hour'); axes[0].set_ylabel('Mean SOC')
axes[0].axhline(0.50, color='green', linestyle='--', alpha=0.6, linewidth=1)
axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].set_title('Mean ENS by Hour of Day', fontsize=12)
axes[1].set_xlabel('Hour'); axes[1].set_ylabel('Mean ENS (kWh)')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
p5 = os.path.join(OUT_DIR, 'eval_daily_profiles.png')
plt.savefig(p5, dpi=150); plt.show(); print(f"Saved: {p5}")

# ============================================================
# PLOT 6 — Generalisation Gap: per-location Load Served % (PPO)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 5))
ppo_locs = results_df[results_df['controller'] == 'PPO'].copy()
colors_bar = ['#2563eb' if s == 'train' else '#7c3aed' for s in ppo_locs['split']]
bars = ax.bar(ppo_locs['location'], ppo_locs['load_served_pct'],
              color=colors_bar, alpha=0.85, edgecolor='white')
for bar, v in zip(bars, ppo_locs['load_served_pct']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.axhline(ppo_locs[ppo_locs['split'] == 'train']['load_served_pct'].mean(),
           color='#2563eb', linestyle='--', linewidth=1.5, alpha=0.7, label='Train avg')
ax.axhline(ppo_locs[ppo_locs['split'] == 'unseen']['load_served_pct'].mean(),
           color='#7c3aed', linestyle='--', linewidth=1.5, alpha=0.7, label='Unseen avg')
legend_els = [mpatches.Patch(facecolor='#2563eb', label='Training location'),
              mpatches.Patch(facecolor='#7c3aed', label='Unseen location')]
ax.legend(handles=legend_els, fontsize=10)
ax.set_title('PPO Load Served (%) — Training vs Unseen Locations (Generalisation Test)', fontsize=13)
ax.set_ylabel('Load Served (%)'); ax.set_ylim(bottom=0)
ax.tick_params(axis='x', rotation=15); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
p6 = os.path.join(OUT_DIR, 'eval_generalisation_gap.png')
plt.savefig(p6, dpi=150); plt.show(); print(f"Saved: {p6}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 65)
print("  EVALUATION COMPLETE")
print("=" * 65)
print(f"  CSVs saved to {OUT_DIR}/:")
print(f"    {os.path.basename(csv_path)}          (6-year totals per location)")
print(f"    {os.path.basename(yearly_csv_path)}    (all KPIs per location × year)")
print(f"  Plots saved to {OUT_DIR}/:")
for p in [p1, p1b, p2, p3, p4, p5, p6]:
    print(f"    {os.path.basename(p)}")

ppo_train_served  = ppo_df[ppo_df['split'] == 'train']['load_served_pct'].mean()
ppo_unseen_served = ppo_df[ppo_df['split'] == 'unseen']['load_served_pct'].mean()
rb_served         = results_df[results_df['controller'] == 'Rule-Based']['load_served_pct'].mean()
ppo_train_life    = ppo_df[ppo_df['split'] == 'train']['lifespan_years'].mean()
ppo_unseen_life   = ppo_df[ppo_df['split'] == 'unseen']['lifespan_years'].mean()

print(f"\n  PPO (train locs)  — Load Served: {ppo_train_served:.2f}% | Lifespan: {ppo_train_life:.1f} yr")
print(f"  PPO (unseen locs) — Load Served: {ppo_unseen_served:.2f}% | Lifespan: {ppo_unseen_life:.1f} yr")
print(f"  Rule-Based (all)  — Load Served: {rb_served:.2f}%")
print(f"\n  Generalisation gap (train → unseen): {ppo_unseen_served - ppo_train_served:+.2f}% Load Served")
print(f"  PPO vs Rule-Based (unseen only):     {ppo_unseen_served - rb_served:+.2f}% Load Served")
print("=" * 65)

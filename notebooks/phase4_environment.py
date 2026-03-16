#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

print("✅ All libraries imported successfully!")

df = pd.read_csv('../data/master_dataset.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

from sklearn.preprocessing import MinMaxScaler

input_features = ['ssrd_wm2', 'tp', 'temp_c', 'load_kw',
                  'location_code', 'hour', 'month', 'dayofweek']

df['location_code'] = df['location'].map({'Tamale': 0, 'Kumasi': 1, 'Axim': 2})
df['hour']          = df['datetime'].dt.hour
df['month']         = df['datetime'].dt.month
df['dayofweek']     = df['datetime'].dt.dayofweek

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
scaler_X.fit(df[input_features])
scaler_y.fit(df[['ssrd_wm2', 'load_kw']])

class MiniGridLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, forecast, output_size):
        super(MiniGridLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.forecast    = forecast
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden_size, forecast * output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :]).view(-1, self.forecast, self.output_size)

lstm_model = MiniGridLSTM(8, 128, 2, 24, 2)
lstm_model.load_state_dict(torch.load('../models/best_lstm.pth', map_location='cpu'))
lstm_model.eval()
print("✅ Data and LSTM model loaded!")
print(f"Dataset shape: {df.shape}")

class MiniGridEnv(gym.Env):
    def __init__(self, df, lstm_model, scaler_X, scaler_y):
        super(MiniGridEnv, self).__init__()
        self.df       = df.reset_index(drop=True)
        self.lstm     = lstm_model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.battery_capacity    = 2000.0  # kWh — properly sized (2-day autonomy)
        self.max_charge_rate     = 400.0   # kW
        self.max_discharge_rate  = 400.0   # kW
        self.charge_efficiency   = 0.95
        self.discharge_efficiency= 0.95
        self.soc_min             = 0.20
        self.soc_max             = 0.90
        self.initial_soh         = 1.0
        # Realistic LiFePO4: 3000-6000 cycles to 80% SOH
        # At 1 deep cycle/day rule-based hits 80% in ~10 yrs → deg = 0.2/3000 = 0.0000667
        # We use 0.00005 so rule-based lands ~12 yrs, AI target ~22 yrs
        self.soh_deg_per_cycle   = 0.00005
        self.calendar_deg_rate   = 0.000002  # per hour at SOC > 0.85 (electrolyte oxidation)
        self.episode_length      = 24 * 365
        self.current_step        = 0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(52,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.soc          = 0.5
        self.soh          = self.initial_soh
        self.rul          = 1.0
        self.efc          = 0.0
        self.ens          = 0.0
        self.current_step = 0
        self.step_idx     = np.random.randint(24, len(self.df) - self.episode_length - 24)
        return self._get_obs(), {}

    def _get_forecast(self):
        lookback = self.df.iloc[self.step_idx-24:self.step_idx]
        X = self.scaler_X.transform(lookback[['ssrd_wm2','tp','temp_c','load_kw',
             'location_code','hour','month','dayofweek']])
        X_t = torch.FloatTensor(X).unsqueeze(0)
        with torch.no_grad():
            fc = self.lstm(X_t).numpy()[0]
        fc_inv = self.scaler_y.inverse_transform(fc.reshape(-1, 2))
        return np.clip(fc_inv[:, 0], 0, None), np.clip(fc_inv[:, 1], 0, None)

    def _get_obs(self):
        sf, lf = self._get_forecast()
        row = self.df.iloc[self.step_idx]
        return np.concatenate([[self.soc, self.soh], sf/1000.0, lf/50.0,
            [row['hour']/23.0, row['month']/12.0]]).astype(np.float32)

    def step(self, action):
        action     = float(np.clip(action[0], -1.0, 1.0))
        row        = self.df.iloc[self.step_idx]
        # ssrd_wm2 is in W/m² → convert to kW: (W/m² / 1000) × 750 m² × 0.75 efficiency
        solar_kw   = (row['ssrd_wm2'] / 1000.0) * 750.0 * 0.75  # ≈ 112 kW peak
        load_kw    = row['load_kw']
        soc_before = self.soc   # track SOC change for cycle depth

        # Solar first serves load directly
        residual_load = max(0.0, load_kw - solar_kw)   # load not covered by solar
        solar_surplus = max(0.0, solar_kw - load_kw)   # excess solar after load

        if action > 0:
            # Charging: use solar surplus first, then grid if needed
            charge_kw = min(action * self.max_charge_rate,
                            (self.soc_max - self.soc) * self.battery_capacity)
            self.soc += (charge_kw * self.charge_efficiency) / self.battery_capacity
            # ENS = residual consumer load not covered by solar or battery discharge
            net_load  = residual_load   # battery is charging, not helping load
        else:
            # Discharging: battery helps cover residual load
            discharge_kw = min(abs(action) * self.max_discharge_rate,
                               (self.soc - self.soc_min) * self.battery_capacity)
            self.soc    -= discharge_kw / self.battery_capacity
            net_load     = max(0.0, residual_load - discharge_kw * self.discharge_efficiency)

        self.soc  = np.clip(self.soc, self.soc_min, self.soc_max)
        # ENS = unserved *load* only — capped at load_kw so it can't exceed demand
        ens_step  = min(max(0.0, net_load), load_kw)
        self.ens += ens_step

        # ── Realistic two-factor degradation ─────────────────
        # Factor 1: Cycle degradation — deeper swings wear more
        cycle_depth = abs(self.soc - soc_before)
        self.efc   += cycle_depth               # depth-weighted EFC (1 full swing = 1 EFC)

        # Factor 2: Calendar aging — high SOC oxidises electrolyte
        calendar_stress = max(0.0, self.soc - 0.85) * self.calendar_deg_rate

        deg_this_step = (cycle_depth * self.soh_deg_per_cycle) + calendar_stress
        self.soh      = max(0.0, self.soh - deg_this_step)
        self.rul      = self.soh / self.initial_soh

        # ── Reward: ENS priority > battery longevity ─────────
        # ENS penalties (max 8.0) >> cycling penalty (max 2.0)
        # Agent always serves load first, avoids waste second
        ens_ratio = ens_step / max(load_kw, 1.0)

        reward  =  1.0
        reward -= (ens_step > 0)   * 4.0   # LOLP: flat penalty every blackout hour
        reward -= ens_ratio        * 4.0   # ENS: how much load unserved
        reward -= cycle_depth      * 2.0   # cycling: avoid unnecessary deep swings
        reward -= (self.soc > 0.85)* 1.5   # high SOC: calendar aging stress
        reward -= (self.soc < 0.30)* 3.0   # low SOC: approaching deep discharge
        reward -= (self.soc < 0.20)* 10.0  # hard floor guard
        reward  = np.clip(reward, -15.0, 2.0)

        self.step_idx     += 1
        self.current_step += 1
        done = self.current_step >= self.episode_length

        return self._get_obs(), reward, done, False, {
            'soc': self.soc, 'soh': self.soh, 'rul': self.rul,
            'ens': ens_step, 'solar_kw': solar_kw, 'load_kw': load_kw, 'action': action}

print("✅ MiniGridEnv — depth-aware degradation + LOLP-aware reward!")

env = MiniGridEnv(df, lstm_model, scaler_X, scaler_y)
obs, _ = env.reset()
print(f"\nObservation space : {env.observation_space.shape}")
print(f"Action space      : {env.action_space.shape}")
print(f"Battery capacity  : {env.battery_capacity} kWh")
print(f"SOC limits        : [{env.soc_min}, {env.soc_max}]")
print(f"Degradation model : depth-aware (|ΔSOC| × 0.5)")
print(f"Reward            : LOLP flat + ENS mag + depth penalty")

print("\n--- Testing 5 random steps ---")
for i in range(5):
    action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)
    print(f"Step {i+1} | Action: {info['action']:+.2f} | SOC: {info['soc']:.3f} | "
          f"SOH: {info['soh']:.4f} | ENS: {info['ens']:.3f} | Reward: {reward:.3f}")

def rule_based_controller(soc, solar_kw, load_kw, soc_min=0.20, soc_max=0.90):
    net = solar_kw - load_kw
    if net > 0:
        return np.array([min(net / 100.0, 1.0)]) if soc < soc_max else np.array([0.0])
    else:
        return np.array([-1.0]) if soc > soc_min else np.array([0.0])

env_rb = MiniGridEnv(df, lstm_model, scaler_X, scaler_y)
obs, _ = env_rb.reset()
rb_soc, rb_soh, rb_ens, rb_solar, rb_load, rb_actions = [], [], [], [], [], []
done = False; total_reward = 0

while not done:
    row    = env_rb.df.iloc[env_rb.step_idx]
    solar_kw_rb = (row['ssrd_wm2'] / 1000.0) * 750.0 * 0.75
    action = rule_based_controller(env_rb.soc, solar_kw_rb, row['load_kw'])
    obs, reward, done, _, info = env_rb.step(action)
    total_reward += reward
    rb_soc.append(info['soc']); rb_soh.append(info['soh'])
    rb_ens.append(info['ens']); rb_solar.append(info['solar_kw'])
    rb_load.append(info['load_kw']); rb_actions.append(info['action'])

print(f"\n✅ Rule-Based tested!")
print(f"Total Steps  : {len(rb_soc):,}")
print(f"Total ENS    : {sum(rb_ens):,.2f} kWh")
print(f"LOLP         : {np.mean(np.array(rb_ens)>0)*100:.1f}%")
print(f"Final SOH    : {rb_soh[-1]:.4f}")

fig, axes = plt.subplots(3, 1, figsize=(15, 10))
show = 24 * 30
axes[0].plot(rb_soc[:show], color='#2d8a45', linewidth=0.8)
axes[0].axhline(y=0.20, color='red',  linestyle='--', linewidth=1, label='Min SOC (20%)')
axes[0].axhline(y=0.90, color='blue', linestyle='--', linewidth=1, label='Max SOC (90%)')
axes[0].set_title('Rule-Based: Battery SOC (First 30 Days)', fontsize=12)
axes[0].set_ylabel('SOC'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(rb_soh[:show], color='orange', linewidth=0.8)
axes[1].set_title('Rule-Based: Battery SOH (First 30 Days)', fontsize=12)
axes[1].set_ylabel('SOH'); axes[1].grid(True, alpha=0.3)
axes[2].plot(rb_solar[:show], color='gold', linewidth=0.8, label='Solar (kW)')
axes[2].plot(rb_load[:show],  color='red',  linewidth=0.8, label='Load (kW)')
axes[2].set_title('Rule-Based: Solar vs Load (First 30 Days)', fontsize=12)
axes[2].set_ylabel('kW'); axes[2].legend(); axes[2].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../data/rule_based_performance.png', dpi=150)
plt.show()
print("✅ Plot saved!")

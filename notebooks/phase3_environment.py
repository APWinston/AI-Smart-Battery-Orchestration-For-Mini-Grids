#!/usr/bin/env python
# coding: utf-8
"""
Phase 3 — Single Battery Environment for Average Ghana SREP Mini-Grid
======================================================================
System: 1 x 650 kWh LiFePO4 battery
  Solar:        132.5 kWp (176.7 m2) — average SREP site
  Mean load:    18.96 kW (~1,318 people)
  Max rate:     130 kW (0.2C)
  Autonomy:     24.0 hours
  Solar/Load:   1.51

Reference: Ghana SREP programme (AfDB/World Bank/Swiss Gov)
  35 mini-grids x 4.525 MWp = 132.5 kWp average per site

Reward improvements over original system:
  - SOC health threshold: 50% (was 40%)
  - SOC health weight:    2.5  (was 1.5)
  - Deep cycle penalty:   NEW — fires when daily DOD > 60%
  - SOH health bonus:     NEW — rewards SOH above 85%
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported!")

# ============================================================
# SYSTEM CONSTANTS — must match phase4_ppo_training.py exactly
# ============================================================
BATTERY_KWH    = 650.0
USABLE_KWH     = BATTERY_KWH * 0.70           # 455 kWh
MEAN_LOAD_KW   = USABLE_KWH / 24              # 18.958 kW
MAX_RATE_KW    = BATTERY_KWH * 0.2            # 130 kW (0.2C)
SOLAR_PEAK_KW    = 132.5                         # kWp at Ghana mean ssrd
SOLAR_AREA_M2    = 176.7                         # m2 — Solar/Load = 1.51
SOLAR_NORM     = (1000.0/1000.0) * SOLAR_AREA_M2 * 0.75  # 132.5 kW peak
LOAD_NORM      = MEAN_LOAD_KW * 2.0           # 37.92 kW

# Verify solar/load ratio at module load
_solar_mean = (216.0/1000.0) * SOLAR_AREA_M2 * 0.75
_ratio      = _solar_mean / MEAN_LOAD_KW
print(f"System: {BATTERY_KWH:.0f} kWh battery | {SOLAR_PEAK_KW:.0f} kWp solar | "
      f"Mean load: {MEAN_LOAD_KW:.4f} kW | Solar/Load: {_ratio:.3f}")
assert abs(_ratio - 1.51) < 0.02, f"Solar/Load ratio wrong: {_ratio:.3f}"
print("Solar/Load ratio: PASS")

input_features = ['ssrd_wm2','tp','temp_c','load_kw',
                  'location_code','hour','month','dayofweek']

# ============================================================
# LOAD SCALED DATA
# ============================================================
df = pd.read_csv('../data/master_dataset_scaled.csv')
df['datetime']      = pd.to_datetime(df['datetime'])
df['location_code'] = df['location'].map({'Tamale':0,'Kumasi':1,'Axim':2})
df['hour']          = df['datetime'].dt.hour
df['month']         = df['datetime'].dt.month
df['dayofweek']     = df['datetime'].dt.dayofweek

scaler_X = MinMaxScaler(); scaler_y = MinMaxScaler()
scaler_X.fit(df[input_features])
scaler_y.fit(df[['ssrd_wm2','load_kw']])
print(f"Data: {df.shape} | Mean load: {df['load_kw'].mean():.4f} kW")

# ============================================================
# LSTM
# ============================================================
class MiniGridLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, forecast, output_size):
        super().__init__()
        self.hidden_size=hidden_size; self.num_layers=num_layers
        self.forecast=forecast; self.output_size=output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden_size, forecast*output_size)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:,-1,:]).view(-1, self.forecast, self.output_size)

lstm_model = MiniGridLSTM(8, 128, 2, 24, 2)
lstm_model.load_state_dict(
    torch.load('../models/best_lstm_scaled.pth', map_location='cpu'))
lstm_model.eval()
print("LSTM (scaled) loaded!")


# ============================================================
# SINGLE BATTERY ENVIRONMENT
# ============================================================
class SingleBatteryEnv(gym.Env):
    """
    650 kWh LiFePO4 battery environment — average Ghana SREP site.

    Observation (52,): [soc, soh, solar_fc x24, load_fc x24, hour, month]
    Action (1,):       charge/discharge in [-1.0, +1.0]

    Reward — stronger battery lifespan signal than original:
      + served_ratio x 2.5        reliability
      + soc_health x 2.5          SOC above 50% (raised from 40%)
      + soh_health x 1.5          SOH above 85% (new)
      - LOLP_penalty x 3.5        flat blackout
      - ens_ratio x 4.0           proportional ENS
      - deep_cycle_pen x 3.0      fires when rainflow half-DOD > 60%
      - calendar_aging (Arrhenius) temperature + SOC stress [Wang et al. 2014]
      - floor_guards              SOC below 25%/20%
      - curtailment x 0.5
    """

    def __init__(self, df, lstm, scaler_X, scaler_y):
        super().__init__()
        self.df                   = df.reset_index(drop=True)
        self.lstm                 = lstm
        self.scaler_X             = scaler_X
        self.scaler_y             = scaler_y
        self.battery_kwh          = BATTERY_KWH
        self.max_rate_kw          = MAX_RATE_KW
        self.charge_efficiency    = 0.95
        self.discharge_efficiency = 0.95
        self.soc_min              = 0.20
        self.soc_max              = 0.90
        self.usable_range         = self.soc_max - self.soc_min
        self.initial_soh          = 1.0
        self.solar_area_m2        = SOLAR_AREA_M2
        self.episode_length       = 24 * 365
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(52,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.soc           = 0.5
        self.soh           = self.initial_soh
        self.efc           = 0.0
        self.ens           = 0.0
        self.current_step  = 0
        self.daily_dod      = 0.0
        self.rf_direction   = 0     # rainflow: +1 charging, -1 discharging
        self.rf_half_start  = 0.5   # rainflow: SOC at start of current half-cycle
        self._prev_soc      = 0.5   # rainflow: previous SOC for direction detection
        self.step_idx      = np.random.randint(24, len(self.df) - self.episode_length - 24)
        return self._get_obs(), {}

    def _get_forecast(self):
        lookback = self.df.iloc[self.step_idx-24:self.step_idx]
        X = self.scaler_X.transform(lookback[input_features])
        X_t = torch.FloatTensor(X).unsqueeze(0)
        with torch.no_grad():
            fc = self.lstm(X_t).numpy()[0]
        fc_inv = self.scaler_y.inverse_transform(fc.reshape(-1,2))
        return np.clip(fc_inv[:,0],0,None), np.clip(fc_inv[:,1],0,None)

    def _get_obs(self):
        sf, lf = self._get_forecast()
        row = self.df.iloc[self.step_idx]
        return np.concatenate([
            [self.soc, self.soh],
            sf / SOLAR_NORM,
            lf / LOAD_NORM,
            [row['hour']/23.0, row['month']/12.0]
        ]).astype(np.float32)

    def _compute_degradation(self, temp_c, action):
        """
        Realistic LiFePO4 degradation — 3 mechanisms:

        1. Cycle aging via rainflow half-cycle counting [Xu et al. 2016]
           L(DOD) = 3500 × (1/DOD)^1.5
           Each direction reversal closes a half-cycle and applies 0.5 × deg

        2. Calendar aging — Arrhenius + cell temperature + asymmetric SOC stress
           [Wang et al. 2014]
           Cell temp = ambient + 5°C (enclosure) + |action|×3°C (self-heating)
           SOC stress asymmetric: high SOC punished harder than low SOC
           (cathode oxidation above 50% SOC more severe than anode stress below)

        3. Low-SOC lithium plating below 30% SOC [Schmalstieg et al. 2014]

        Parameters
        ----------
        temp_c : float — ambient temperature from ERA5 dataset (°C)
        action : float — normalised charge/discharge action [-1, +1]
        """
        # ── 1. Rainflow cycle aging ───────────────────────────────
        soc_change = self.soc - self._prev_soc
        deg_cycle  = 0.0
        if abs(soc_change) > 1e-4:
            new_dir = 1 if soc_change > 0 else -1
            if new_dir != self.rf_direction and self.rf_direction != 0:
                # Direction reversal — close half-cycle
                half_dod = abs(self.soc - self.rf_half_start) / self.usable_range
                if half_dod > 1e-4:
                    self.efc      += half_dod * 0.5
                    cycle_life     = 3500.0 * (1.0 / max(half_dod, 0.01)) ** 1.5
                    deg_cycle      = (0.20 / cycle_life) * 0.5
                    self.daily_dod = half_dod
                self.rf_half_start = self.soc
            self.rf_direction = new_dir
        self._prev_soc = self.soc

        # ── 2. Calendar aging (Arrhenius + cell temp + asymmetric SOC) ──
        cell_temp  = temp_c + 5.0 + abs(action) * 3.0
        T_kelvin   = max(cell_temp + 273.15, 273.15)
        k_cal      = 14876.0 * np.exp(-24500.0 / (8.314 * T_kelvin))
        if self.soc >= 0.50:
            soc_stress = 0.70 + 0.60 * (self.soc - 0.50) ** 2
        else:
            soc_stress = 0.70 + 0.30 * (self.soc - 0.50) ** 2
        deg_cal = 1.423e-6 * k_cal * soc_stress

        # ── 3. Low-SOC lithium plating ────────────────────────────
        deg_low_soc = 2e-4 * max(0.0, 0.30 - self.soc) ** 2

        return deg_cycle + deg_cal + deg_low_soc


    def step(self, action):
        action        = float(np.clip(action[0], -1.0, 1.0))
        row           = self.df.iloc[self.step_idx]
        solar_kw      = (row['ssrd_wm2']/1000.0) * self.solar_area_m2 * 0.75
        load_kw       = row['load_kw']
        soc_before    = self.soc
        residual_load = max(0.0, load_kw - solar_kw)
        solar_surplus = max(0.0, solar_kw - load_kw)

        if action > 0:
            charge_kw    = min(action*self.max_rate_kw,
                               (self.soc_max-self.soc)*self.battery_kwh)
            self.soc    += (charge_kw*self.charge_efficiency)/self.battery_kwh
            net_load     = residual_load
            curtailed_kw = max(0.0, solar_surplus - charge_kw)
        else:
            discharge_kw = min(abs(action)*self.max_rate_kw,
                               (self.soc-self.soc_min)*self.battery_kwh)
            self.soc    -= discharge_kw/self.battery_kwh
            net_load     = max(0.0, residual_load - discharge_kw*self.discharge_efficiency)
            curtailed_kw = solar_surplus

        self.soc  = float(np.clip(self.soc, self.soc_min, self.soc_max))
        ens_step  = min(max(0.0, net_load), load_kw)
        self.ens += ens_step

        cycle_depth   = abs(self.soc - soc_before)
        deg_this_step = self._compute_degradation(row['temp_c'], action)
        self.soh      = max(0.0, self.soh - deg_this_step)

        # ── Reward ──────────────────────────────────────────
        ens_ratio     = ens_step / max(load_kw, 1.0)
        curtail_ratio = curtailed_kw / max(solar_kw, 1.0)
        served_ratio  = 1.0 - ens_ratio

        reward  =  served_ratio * 2.5
        reward -= (ens_step > 0) * 3.5
        reward -= ens_ratio      * 4.0
        soc_health = max(0.0, self.soc - 0.50) * served_ratio * 2.5
        reward += soc_health
        soh_health = max(0.0, self.soh - 0.85) * 1.5
        reward += soh_health
        if self.current_step % 24 == 23 and self.daily_dod > 0.60:
            reward -= (self.daily_dod - 0.60) * 3.0
        reward -= (self.soc > 0.85) * 2.0
        reward -= (self.soc < 0.25) * 2.0
        reward -= (self.soc < 0.20) * 10.0
        reward -= cycle_depth * (1.0 - served_ratio) * 2.0
        reward -= curtail_ratio * 0.5
        reward  = np.clip(reward, -15.0, 5.0)

        self.step_idx += 1; self.current_step += 1
        done = self.current_step >= self.episode_length
        return self._get_obs(), reward, done, False, {
            'soc': self.soc, 'soh': self.soh, 'efc': self.efc,
            'ens': ens_step, 'solar_kw': solar_kw, 'load_kw': load_kw,
            'action': action, 'curtailed_kw': curtailed_kw,
        }


print(f"\nSingleBatteryEnv ready!")
print(f"  Battery:     {BATTERY_KWH:.0f} kWh | Max rate: {MAX_RATE_KW:.0f} kW (0.2C)")
print(f"  Solar:       {SOLAR_PEAK_KW:.0f} kWp ({SOLAR_AREA_M2:.1f} m2)")
print(f"  Obs: (52,)   Action: (1,)")
print(f"  Reward:      soc_health x2.5 (>50%) + deep_cycle_pen (DOD>60%) + soh_bonus")


# ============================================================
# VERIFICATION
# ============================================================
env = SingleBatteryEnv(df, lstm_model, scaler_X, scaler_y)
obs, _ = env.reset()
print(f"\nObs space:   {env.observation_space.shape}")
print(f"Action space:{env.action_space.shape}")
print(f"Battery:     {env.battery_kwh:.0f} kWh")
print(f"Solar area:  {env.solar_area_m2} m2")
print(f"SOC limits:  [{env.soc_min}, {env.soc_max}]")
print(f"Max rate:    {env.max_rate_kw:.0f} kW")

print("\n--- Testing 5 random steps ---")
for i in range(5):
    action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)
    print(f"Step {i+1} | Act:{info['action']:+.2f} | SOC:{info['soc']:.3f} | "
          f"SOH:{info['soh']:.4f} | ENS:{info['ens']:.4f} | R:{reward:.3f}")


# ============================================================
# RULE-BASED CONTROLLER
# ============================================================
def rule_based(soc, solar_kw, load_kw, soc_min=0.20, soc_max=0.90):
    net = solar_kw - load_kw
    if net > 0:
        return np.array([min(net/MAX_RATE_KW, 1.0)]) if soc < soc_max else np.array([0.0])
    else:
        return np.array([-1.0]) if soc > soc_min else np.array([0.0])


# ============================================================
# RULE-BASED BASELINE (1 year)
# ============================================================
print("\n--- Rule-Based Baseline (1 year) ---")
env_rb = SingleBatteryEnv(df, lstm_model, scaler_X, scaler_y)
obs, _ = env_rb.reset()
rb_ens=[]; rb_soh=[]; done=False

while not done:
    row    = env_rb.df.iloc[env_rb.step_idx]
    solar  = (row['ssrd_wm2']/1000.0) * SOLAR_AREA_M2 * 0.75
    action = rule_based(env_rb.soc, solar, row['load_kw'])
    obs, _, done, _, info = env_rb.step(action)
    rb_ens.append(info['ens']); rb_soh.append(info['soh'])

rb_ens_arr = np.array(rb_ens)
deg_per_yr  = 1.0 - rb_soh[-1]
print(f"Total Steps:  {len(rb_ens):,}")
print(f"Total ENS:    {sum(rb_ens):,.4f} kWh")
print(f"LOLP:         {np.mean(rb_ens_arr>0)*100:.1f}%")
# Sum load over the actual episode steps (step_idx ran from start_idx to start_idx+8760)
_ep_start = env_rb.step_idx - len(rb_ens)
_total_load = env_rb.df['load_kw'].iloc[_ep_start:_ep_start+len(rb_ens)].sum()
print(f"Load Served:  {(1-sum(rb_ens)/_total_load)*100:.1f}%")
print(f"Final SOH:    {rb_soh[-1]:.4f}")
print(f"Deg/year:     {deg_per_yr:.4f}")
print(f"Lifespan:     {0.20/deg_per_yr:.1f} years (to 80% SOH)" if deg_per_yr>0 else "Lifespan: >99yr")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(14, 9))
show = 24*30
axes[0].plot(rb_soh[:show], color='orange', linewidth=0.8)
axes[0].set_title(f'Rule-Based: SOH — {BATTERY_KWH:.0f} kWh Battery (First 30 Days)', fontsize=12)
axes[0].set_ylabel('SOH'); axes[0].grid(True, alpha=0.3)

axes[1].plot(rb_ens[:show], color='red', linewidth=0.6, alpha=0.8)
axes[1].set_title('Rule-Based: ENS per Hour (First 30 Days)', fontsize=12)
axes[1].set_ylabel('ENS (kWh)'); axes[1].grid(True, alpha=0.3)

tamale = df[df['location']=='Tamale'].reset_index(drop=True)
sol_kw = (tamale['ssrd_wm2']/1000.0) * SOLAR_AREA_M2 * 0.75
axes[2].plot(sol_kw.values[:show], color='#f59e0b', linewidth=0.8, label='Solar (kW)')
axes[2].plot(tamale['load_kw'].values[:show], color='#ef4444', linewidth=0.8, label='Load (kW)')
axes[2].set_title('Solar vs Load (Tamale, First 30 Days)', fontsize=12)
axes[2].set_ylabel('kW'); axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../data/singlebattery_rb_baseline.png', dpi=150)
plt.show()
print("\nPlot saved: ../data/singlebattery_rb_baseline.png")
print("Run phase4_ppo_training.py next.")

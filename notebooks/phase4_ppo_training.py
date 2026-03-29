#!/usr/bin/env python
# coding: utf-8
"""
Phase 4 — PPO Training for Average Ghana SREP Mini-Grid
========================================================
System: 1 x 650 kWh LiFePO4 | 132.5 kWp solar | ~1,318 people
        Solar/Load ratio: 1.51 | Autonomy: 24.0h
        Obs: (52,) | Action: (1,)

Reference: Ghana SREP (AfDB/World Bank) — 35 mini-grids,
  4.525 MWp total = 132.5 kWp average per site

Run order:
  CHECKPOINT=1 -> python phase4_ppo_training.py   (0M-4M)
  CHECKPOINT=2 -> python phase4_ppo_training.py   (4M-8M)
  CHECKPOINT=3 -> python phase4_ppo_training.py   (8M-12M)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import gymnasium as gym
from gymnasium import spaces
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
CHECKPOINT    = 1          # <- SET THIS: 1, 2, or 3
STEPS_PER_RUN = 4_000_000
FRESH_START   = (CHECKPOINT == 1)

# System constants — must match phase3_environment.py exactly
BATTERY_KWH   = 650.0
USABLE_KWH    = BATTERY_KWH * 0.70
MEAN_LOAD_KW  = USABLE_KWH / 24              # 18.958 kW
MAX_RATE_KW   = BATTERY_KWH * 0.2            # 130 kW (0.2C)
SOLAR_PEAK_KW    = 132.5                         # kWp at Ghana mean ssrd
SOLAR_AREA_M2    = 176.7                         # m2 — Solar/Load = 1.51
SOLAR_NORM    = (1000.0/1000.0) * SOLAR_AREA_M2 * 0.75  # 132.5 kW
LOAD_NORM     = MEAN_LOAD_KW * 2.0           # 37.92 kW

TAG      = f"{CHECKPOINT * 4}m"
SAVE_TAG = f"srep_avg_650kwh_{TAG}"
MODEL_SAVE = f"../models/{SAVE_TAG}"
VN_SAVE    = f"../models/vecnormalize_{SAVE_TAG}.pkl"

feat_cols = ['ssrd_wm2','tp','temp_c','load_kw',
             'location_code','hour','month','dayofweek']

# ============================================================
# LOAD DATA
# ============================================================
print("Loading scaled data...")
df = pd.read_csv('../data/master_dataset_scaled.csv')
df['datetime']      = pd.to_datetime(df['datetime'])
df['location_code'] = df['location'].map({'Tamale':0,'Kumasi':1,'Axim':2})
df['hour']          = df['datetime'].dt.hour
df['month']         = df['datetime'].dt.month
df['dayofweek']     = df['datetime'].dt.dayofweek

scaler_X = MinMaxScaler(); scaler_y = MinMaxScaler()
scaler_X.fit(df[feat_cols]); scaler_y.fit(df[['ssrd_wm2','load_kw']])

df_tamale = df[df['location']=='Tamale'].reset_index(drop=True)
df_kumasi  = df[df['location']=='Kumasi'].reset_index(drop=True)
df_axim    = df[df['location']=='Axim'].reset_index(drop=True)

_solar = (216.0/1000.0) * SOLAR_AREA_M2 * 0.75
print(f"Data: {df.shape} | Mean load: {df['load_kw'].mean():.4f} kW")
print(f"Solar/Load ratio: {_solar/MEAN_LOAD_KW:.3f} (target: 1.51)")

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
# ENVIRONMENT
# ============================================================
class SingleBatteryEnv(gym.Env):
    def __init__(self, df, lstm, scaler_X, scaler_y):
        super().__init__()
        self.df=df.reset_index(drop=True)
        self.lstm=lstm; self.scaler_X=scaler_X; self.scaler_y=scaler_y
        self.battery_kwh=BATTERY_KWH; self.max_rate_kw=MAX_RATE_KW
        self.charge_efficiency=0.95; self.discharge_efficiency=0.95
        self.soc_min=0.20; self.soc_max=0.90
        self.usable_range=self.soc_max-self.soc_min
        self.initial_soh=1.0
        self.solar_area_m2=SOLAR_AREA_M2; self.episode_length=24*365
        self.action_space      = spaces.Box(-1.0,1.0,shape=(1,),dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf,np.inf,shape=(52,),dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.soc=0.5; self.soh=self.initial_soh
        self.efc=0.0; self.ens=0.0; self.current_step=0
        self.daily_dod=0.0
        self.rf_direction   = 0
        self.rf_half_start  = 0.5
        self._prev_soc      = 0.5
        self.step_idx=np.random.randint(24, len(self.df)-self.episode_length-24)
        return self._get_obs(), {}

    def _get_forecast(self):
        lookback = self.df.iloc[self.step_idx-24:self.step_idx]
        X = self.scaler_X.transform(lookback[feat_cols].values)
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
            sf/SOLAR_NORM, lf/LOAD_NORM,
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
        action=float(np.clip(action[0],-1.0,1.0))
        row=self.df.iloc[self.step_idx]
        solar_kw=(row['ssrd_wm2']/1000.0)*self.solar_area_m2*0.75
        load_kw=row['load_kw']; soc_before=self.soc
        residual_load=max(0.0,load_kw-solar_kw)
        solar_surplus=max(0.0,solar_kw-load_kw)
        if action>0:
            ckw=min(action*self.max_rate_kw,(self.soc_max-self.soc)*self.battery_kwh)
            self.soc+=(ckw*self.charge_efficiency)/self.battery_kwh
            net_load=residual_load; curtailed_kw=max(0.0,solar_surplus-ckw)
        else:
            dkw=min(abs(action)*self.max_rate_kw,
                    (self.soc-self.soc_min)*self.battery_kwh)
            self.soc-=dkw/self.battery_kwh
            net_load=max(0.0,residual_load-dkw*self.discharge_efficiency)
            curtailed_kw=solar_surplus
        self.soc=float(np.clip(self.soc,self.soc_min,self.soc_max))
        ens_step=min(max(0.0,net_load),load_kw); self.ens+=ens_step
        cycle_depth=abs(self.soc-soc_before)
        self.soh = max(0.0, self.soh - self._compute_degradation(row['temp_c'], action))

        ens_ratio=ens_step/max(load_kw,1.0)
        curtail_ratio=curtailed_kw/max(solar_kw,1.0)
        served_ratio=1.0-ens_ratio

        reward  =  served_ratio * 2.5
        reward -= (ens_step > 0) * 3.5
        reward -= ens_ratio      * 4.0
        reward += max(0.0, self.soc-0.50) * served_ratio * 2.5
        reward += max(0.0, self.soh-0.85) * 1.5
        if self.current_step%24==23 and self.daily_dod>0.60:
            reward -= (self.daily_dod-0.60)*3.0
        reward -= (self.soc > 0.85) * 2.0
        reward -= (self.soc < 0.25) * 2.0
        reward -= (self.soc < 0.20) * 10.0
        reward -= cycle_depth*(1.0-served_ratio)*2.0
        reward -= curtail_ratio*0.5
        reward  = np.clip(reward,-15.0,5.0)

        self.step_idx+=1; self.current_step+=1
        done=self.current_step>=self.episode_length
        return self._get_obs(), reward, done, False, {
            'soc':self.soc,'soh':self.soh,'ens':ens_step,
            'solar_kw':solar_kw,'load_kw':load_kw,'action':action,'curtailed_kw':curtailed_kw,
        }

print(f"SingleBatteryEnv ready! Obs(52,) Act(1,)")


# ============================================================
# MULTI-LOCATION WRAPPER
# ============================================================
class MultiLocationEnv(SingleBatteryEnv):
    def __init__(self):
        self._locs=[df_tamale,df_kumasi,df_axim]; self._ep=0
        super().__init__(df_tamale, lstm_model, scaler_X, scaler_y)
    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)  # skip SingleBatteryEnv.reset() to avoid wasted LSTM call
        loc_idx=(self._ep//6)%3
        self.df=self._locs[loc_idx].reset_index(drop=True)
        n_chunks=(len(self.df)-self.episode_length-24)//self.episode_length
        chunk_idx=self._ep%max(n_chunks,1)
        self.step_idx=min(24+chunk_idx*self.episode_length,
                          len(self.df)-self.episode_length-24)
        self._ep+=1
        self.soc=0.5; self.soh=self.initial_soh
        self.efc=0.0; self.ens=0.0; self.current_step=0
        self.daily_dod=0.0
        self.rf_direction   = 0
        self.rf_half_start  = 0.5
        self._prev_soc      = 0.5
        return self._get_obs(), {}


# ============================================================
# AUTO-SAVE CALLBACK
# ============================================================
class AutoSaveCallback(BaseCallback):
    def __init__(self, freq, save_tag, verbose=1):
        super().__init__(verbose); self.freq=freq; self.save_tag=save_tag
    def _on_step(self):
        if self.n_calls%self.freq==0:
            self.model.save(f'../models/{self.save_tag}_autosave')
            self.training_env.save(f'../models/vecnormalize_{self.save_tag}_autosave.pkl')
            if self.verbose: print(f'\nAuto-saved at step {self.n_calls:,}')
        return True


# ============================================================
# TRAINING
# ============================================================
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    N_ENVS = 4
    env = DummyVecEnv([lambda: Monitor(MultiLocationEnv()) for _ in range(N_ENVS)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    eval_env = DummyVecEnv([lambda: Monitor(MultiLocationEnv())])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    policy_kwargs = dict(net_arch=[256,256])

    if FRESH_START:
        print(f'\nFresh start Phase 1 (0M to 4M)...')
        model = PPO(
            'MlpPolicy', env, policy_kwargs=policy_kwargs,
            learning_rate=3e-4, n_steps=1024, batch_size=256, n_epochs=10,
            gamma=0.995, gae_lambda=0.95, clip_range=0.2, ent_coef=0.005,
            vf_coef=0.5, max_grad_norm=0.5, verbose=1, device='auto'
        )
    else:
        prev=f"srep_avg_650kwh_{(CHECKPOINT-1)*4}m"
        print(f'\nLoading checkpoint {prev}...')
        _raw_env = DummyVecEnv([lambda: Monitor(MultiLocationEnv()) for _ in range(N_ENVS)])
        env = VecNormalize.load(f'../models/vecnormalize_{prev}.pkl', _raw_env)
        env.norm_reward=True
        model = PPO.load(f'../models/{prev}', env=env, device='auto')
        eval_env.obs_rms = env.obs_rms   # sync normalisation stats

    eval_cb = EvalCallback(
        eval_env, best_model_save_path='../models/',
        log_path='../models/', eval_freq=10000,
        n_eval_episodes=5, deterministic=True, verbose=1)
    save_cb = AutoSaveCallback(freq=500_000, save_tag=SAVE_TAG)

    print(f'\n{"="*60}')
    print(f'  PPO Training Phase {CHECKPOINT}/3 — Average Ghana SREP Site')
    print(f'  Battery: {BATTERY_KWH:.0f} kWh | Solar: {SOLAR_PEAK_KW:.0f} kWp | Load: {MEAN_LOAD_KW:.2f} kW')
    print(f'  Solar/Load: {_solar/MEAN_LOAD_KW:.3f} | Population: ~{MEAN_LOAD_KW*69.5:.0f} people')
    print(f'  Obs: (52,) | Action: (1,)')
    print(f'  Reward: soc_health x2.5 (>50%) + deep_cycle_pen (DOD>60%) + soh_bonus')
    print(f'{"="*60}')

    model.learn(
        total_timesteps=STEPS_PER_RUN,
        callback=[eval_cb, save_cb],
        progress_bar=True,
        reset_num_timesteps=FRESH_START
    )

    eval_env.obs_rms = env.obs_rms   # sync normalisation before saving
    model.save(MODEL_SAVE); env.save(VN_SAVE)
    print(f'\nSaved: {MODEL_SAVE}.zip')
    print(f'Saved: {VN_SAVE}')
    print(f'\nPhase {CHECKPOINT} complete!')
    if CHECKPOINT < 3:
        print(f'Next: set CHECKPOINT={CHECKPOINT+1} and run again')
    else:
        print('All phases complete! Run phase5_evaluation.py next.')

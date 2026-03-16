import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from sklearn.preprocessing import MinMaxScaler
import gymnasium as gym
from gymnasium import spaces

print("Loading data...")
df = pd.read_csv('../data/master_dataset.csv')
df['datetime']      = pd.to_datetime(df['datetime'])
df['location_code'] = df['location'].map({'Tamale':0,'Kumasi':1,'Axim':2})
df['hour']          = df['datetime'].dt.hour
df['month']         = df['datetime'].dt.month
df['dayofweek']     = df['datetime'].dt.dayofweek

input_features = ['ssrd_wm2','tp','temp_c','load_kw',
                  'location_code','hour','month','dayofweek']
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
scaler_X.fit(df[input_features])
scaler_y.fit(df[['ssrd_wm2','load_kw']])
print(f"✅ Data loaded! Shape: {df.shape}")

class MiniGridLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, forecast, output_size):
        super().__init__()
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
        return self.fc(out[:,-1,:]).view(-1, self.forecast, self.output_size)

lstm_model = MiniGridLSTM(8, 128, 2, 24, 2)
lstm_model.load_state_dict(torch.load('../models/best_lstm.pth', map_location='cpu'))
lstm_model.eval()
print("✅ LSTM loaded!")

class MiniGridEnv(gym.Env):
    def __init__(self, df, lstm, scaler_X, scaler_y):
        super().__init__()
        self.df                  = df.reset_index(drop=True)
        self.lstm                = lstm
        self.scaler_X            = scaler_X
        self.scaler_y            = scaler_y
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
        fc_inv = self.scaler_y.inverse_transform(fc.reshape(-1,2))
        return np.clip(fc_inv[:,0],0,None), np.clip(fc_inv[:,1],0,None)

    def _get_obs(self):
        sf, lf = self._get_forecast()
        row    = self.df.iloc[self.step_idx]
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
            'soc':self.soc,'soh':self.soh,'rul':self.rul,
            'ens':ens_step,'solar_kw':solar_kw,'load_kw':load_kw,'action':action}

print("✅ Environment — depth-aware degradation + LOLP-aware reward!")

df_tamale = df[df['location']=='Tamale'].reset_index(drop=True)
df_kumasi  = df[df['location']=='Kumasi'].reset_index(drop=True)
df_axim    = df[df['location']=='Axim'].reset_index(drop=True)

class MultiLocationEnv(MiniGridEnv):
    def __init__(self):
        self.location_dfs   = [df_tamale, df_kumasi, df_axim]
        self.location_names = ['Tamale','Kumasi','Axim']
        self.episode_count  = 0
        super().__init__(df_tamale, lstm_model, scaler_X, scaler_y)

    def reset(self, seed=None, options=None):
        loc_idx  = (self.episode_count // 6) % 3
        self.df  = self.location_dfs[loc_idx].reset_index(drop=True)
        self.current_location = self.location_names[loc_idx]
        n_chunks  = (len(self.df) - self.episode_length - 24) // self.episode_length
        chunk_idx = self.episode_count % max(n_chunks, 1)
        self.step_idx = 24 + (chunk_idx * self.episode_length)
        self.step_idx = min(self.step_idx, len(self.df) - self.episode_length - 24)
        self.episode_count += 1
        self.soc          = 0.5
        self.soh          = self.initial_soh
        self.rul          = 1.0
        self.efc          = 0.0
        self.ens          = 0.0
        self.current_step = 0
        return self._get_obs(), {}

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()   # required on Windows

    env      = SubprocVecEnv([lambda: Monitor(MultiLocationEnv()) for _ in range(2)])  # 2 envs saves RAM
    env      = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # eval_env must NOT use VecNormalize with norm_reward during eval
    # but must share the same obs normalisation stats → sync with env
    eval_env = SubprocVecEnv([lambda: Monitor(MultiLocationEnv()) for _ in range(1)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    TOTAL_STEPS   = 3_000_000

    print("Creating fresh PPO model...")
    policy_kwargs = dict(net_arch=[256, 256])

    model_ppo = PPO(
        policy        = 'MlpPolicy',
        env           = env,
        policy_kwargs = policy_kwargs,
        learning_rate = 3e-4,
        n_steps       = 1024,   # reduced from 2048 to lower RAM usage
        batch_size    = 256,
        n_epochs      = 10,
        gamma         = 0.995,
        gae_lambda    = 0.95,
        clip_range    = 0.2,
        ent_coef      = 0.005,
        vf_coef       = 0.5,
        max_grad_norm = 0.5,
        verbose       = 1
    )
    print("✅ Fresh model created!")

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = '../models/',
        log_path             = '../models/',
        eval_freq            = 10000,
        n_eval_episodes      = 5,
        deterministic        = True,
        render               = False,
        verbose              = 1
    )

    print("\n🚀 Starting Fresh PPO Training...")
    print("Episode length : 8,760 steps (1 year)")
    print("Envs           : 2 parallel (SubprocVecEnv) — memory optimised")
    print("Obs/Reward     : VecNormalize active")
    print("Policy network : [256, 256] (upgraded)")
    print(f"Total timesteps: {TOTAL_STEPS:,} (fresh start)")
    print("Reward         : LOLP flat + ENS magnitude + depth-aware degradation")
    print("Locations      : Tamale → Kumasi → Axim (sequential)")
    print("-" * 50)

    model_ppo.learn(
        total_timesteps     = TOTAL_STEPS,
        callback            = eval_callback,
        progress_bar        = True,
        reset_num_timesteps = True
    )

    # Save model AND normalisation stats — both needed for inference
    model_ppo.save('../models/ppo_final')
    env.save('../models/vecnormalize_stats.pkl')
    print("-" * 50)
    print("✅ Training Complete!")
    print("✅ Best model      → ../models/best_model.zip")
    print("✅ Final model     → ../models/ppo_final.zip")
    print("✅ Norm stats      → ../models/vecnormalize_stats.pkl")

    try:
        evals     = np.load('../models/evaluations.npz')
        timesteps = evals['timesteps']
        rewards   = evals['results'].mean(axis=1)
        plt.figure(figsize=(10,4))
        plt.plot(timesteps, rewards, color='green', linewidth=2)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
        plt.title('PPO Training — LOLP + Depth-Aware Degradation Reward')
        plt.xlabel('Timesteps'); plt.ylabel('Mean Reward')
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig('../data/ppo_training_curve.png', dpi=150)
        print("✅ Training curve saved!")
    except Exception as e:
        print(f"Could not plot: {e}")

# ══════════════════════════════════════════════════════════════════
# Phase 6 — Evaluation on UNSEEN locations (Accra, Bolgatanga, Akosombo)
# These locations were NOT used in training → unbiased generalisation test
# ══════════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sklearn.preprocessing import MinMaxScaler
import gymnasium as gym
from gymnasium import spaces

# ── Step 1: Build evaluation dataset from new CDS files ──────────
# Place your downloaded CDS files in ../data/ with these exact names:
#   Solar Irradiance Accra.csv
#   Solar Irradiance Bolgatanga.csv
#   Solar Irradiance Akosombo.csv
#   Precipitation Accra.csv
#   Precipitation Bolgatanga.csv
#   Precipitation Akosombo.csv
#   2m temperature Accra.csv
#   2m temperature Bolgatanga.csv
#   2m temperature Akosombo.csv

def build_eval_dataset(data_path='../data/'):
    """Build evaluation dataset for new locations using same load profile as training."""

    new_locations = ['Accra', 'Bolgatanga', 'Akosombo']

    # Load Nigeria demand data — same as training (load is reused)
    nigeria = pd.read_excel(data_path + 'Nigeria Load data.xlsx', header=3)
    nigeria['date time'] = pd.to_datetime(nigeria['date time'])
    nigeria_clean = nigeria[['date time', 'National Suppressed Demand']].copy()
    nigeria_clean.columns = ['datetime', 'demand_mw']
    max_demand = nigeria_clean['demand_mw'].max()
    nigeria_clean['load_kw'] = (nigeria_clean['demand_mw'] / max_demand) * 50

    # Tile load to cover 2020–2026
    era5_start   = pd.Timestamp('2020-01-01')
    era5_end     = pd.Timestamp('2026-02-05 23:00:00')
    hourly_index = pd.date_range(start=era5_start, end=era5_end, freq='h')
    n_repeats    = int(np.ceil(len(hourly_index) / len(nigeria_clean)))
    load_repeated = pd.DataFrame({
        'datetime': hourly_index,
        'load_kw':  np.tile(nigeria_clean['load_kw'].values, n_repeats)[:len(hourly_index)]
    })
    load_repeated['datetime'] = pd.to_datetime(load_repeated['datetime']).dt.tz_localize(None)

    dfs = []
    for loc in new_locations:
        print(f'  Loading {loc}...')
        try:
            ssrd  = pd.read_csv(data_path + f'Solar Irradiance {loc}.csv')
            precip= pd.read_csv(data_path + f'Precipitation {loc}.csv')
            temp  = pd.read_csv(data_path + f'2m temperature {loc}.csv')
        except FileNotFoundError as e:
            print(f'  ❌ File not found: {e}')
            print(f'  ⚠️  Skipping {loc} — download the CDS files first')
            continue

        # Parse and strip timezone
        for df in [ssrd, precip, temp]:
            df['valid_time'] = pd.to_datetime(df['valid_time']).dt.tz_localize(None)

        # Merge
        df = ssrd[['valid_time','ssrd']].rename(columns={'valid_time':'datetime'})
        df = df.merge(precip[['valid_time','tp']].rename(columns={'valid_time':'datetime'}), on='datetime')
        df = df.merge(temp[['valid_time','t2m']].rename(columns={'valid_time':'datetime'}),   on='datetime')
        df = df.merge(load_repeated, on='datetime')

        # Convert units — same as training
        df['ssrd_wm2'] = df['ssrd'] / 3600
        df['temp_c']   = df['t2m'] - 273.15
        df['location'] = loc
        df = df[['datetime','location','ssrd_wm2','tp','temp_c','load_kw']]
        print(f'  ✅ {loc}: {df.shape} | {df["datetime"].min()} → {df["datetime"].max()}')
        dfs.append(df)

    if not dfs:
        raise ValueError('No location data found. Download CDS files first.')

    eval_df = pd.concat(dfs, ignore_index=True)
    print(f'\n✅ Evaluation dataset: {eval_df.shape}')
    return eval_df

# ── Step 2: Load models + fit scalers on TRAINING data ───────────
print('Loading training data to fit scalers...')
train_df = pd.read_csv('../data/master_dataset.csv')
train_df['datetime']      = pd.to_datetime(train_df['datetime'])
train_df['location_code'] = train_df['location'].map({'Tamale':0,'Kumasi':1,'Axim':2})
train_df['hour']          = train_df['datetime'].dt.hour
train_df['month']         = train_df['datetime'].dt.month
train_df['dayofweek']     = train_df['datetime'].dt.dayofweek

# CRITICAL: scalers must be fit on training data only
input_features = ['ssrd_wm2','tp','temp_c','load_kw','location_code','hour','month','dayofweek']
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
scaler_X.fit(train_df[input_features])
scaler_y.fit(train_df[['ssrd_wm2','load_kw']])
print('✅ Scalers fitted on training data')

# ── LSTM definition ───────────────────────────────────────────────
class MiniGridLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, forecast, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.forecast    = forecast
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden_size, forecast * output_size)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:,-1,:]).view(-1, self.forecast, self.output_size)

lstm_model = MiniGridLSTM(8, 128, 2, 24, 2)
lstm_model.load_state_dict(torch.load('../models/best_lstm.pth', map_location='cpu'))
lstm_model.eval()
print('✅ LSTM loaded')

# ── Environment (must match phase5 exactly) ───────────────────────
class MiniGridEnv(gym.Env):
    def __init__(self, df, lstm, scaler_X, scaler_y, location_code=3):
        super().__init__()
        self.df                  = df.reset_index(drop=True)
        self.lstm                = lstm
        self.scaler_X            = scaler_X
        self.scaler_y            = scaler_y
        self.location_code       = location_code  # 3=Accra, 4=Bolgatanga, 5=Akosombo (unseen)
        self.battery_capacity    = 2000.0
        self.max_charge_rate     = 400.0
        self.max_discharge_rate  = 400.0
        self.charge_efficiency   = 0.95
        self.discharge_efficiency= 0.95
        self.soc_min             = 0.20
        self.soc_max             = 0.90
        self.initial_soh         = 1.0
        self.soh_deg_per_cycle   = 0.00005
        self.calendar_deg_rate   = 0.000002
        self.episode_length      = 24 * 365 * 6
        self.action_space        = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space   = spaces.Box(low=-np.inf, high=np.inf, shape=(52,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.soc = 0.5; self.soh = self.initial_soh; self.rul = 1.0
        self.efc = 0.0; self.ens = 0.0; self.current_step = 0; self.step_idx = 24
        return self._get_obs(), {}

    def _get_forecast(self):
        lookback = self.df.iloc[self.step_idx-24:self.step_idx].copy()
        # Use location_code 0 (Tamale-like) for scaler compatibility with new locations
        lookback['location_code'] = self.location_code
        lookback['hour']          = pd.to_datetime(lookback['datetime']).dt.hour if 'datetime' in lookback else 0
        lookback['month']         = pd.to_datetime(lookback['datetime']).dt.month if 'datetime' in lookback else 1
        lookback['dayofweek']     = pd.to_datetime(lookback['datetime']).dt.dayofweek if 'datetime' in lookback else 0
        X = self.scaler_X.transform(lookback[input_features])
        X_t = torch.FloatTensor(X).unsqueeze(0)
        with torch.no_grad():
            fc = self.lstm(X_t).numpy()[0]
        fc_inv = self.scaler_y.inverse_transform(fc.reshape(-1,2))
        return np.clip(fc_inv[:,0],0,None), np.clip(fc_inv[:,1],0,None)

    def _get_obs(self):
        sf, lf = self._get_forecast()
        row = self.df.iloc[self.step_idx]
        hour  = pd.to_datetime(row['datetime']).hour  if 'datetime' in row else 0
        month = pd.to_datetime(row['datetime']).month if 'datetime' in row else 1
        return np.concatenate([[self.soc, self.soh], sf/1000.0, lf/50.0,
            [hour/23.0, month/12.0]]).astype(np.float32)

    def step(self, action):
        action     = float(np.clip(action[0], -1.0, 1.0))
        row        = self.df.iloc[self.step_idx]
        solar_kw   = (row['ssrd_wm2'] / 1000.0) * 750.0 * 0.75
        load_kw    = row['load_kw']
        soc_before = self.soc
        residual_load = max(0.0, load_kw - solar_kw)
        if action > 0:
            charge_kw = min(action * self.max_charge_rate,
                            (self.soc_max - self.soc) * self.battery_capacity)
            self.soc += (charge_kw * self.charge_efficiency) / self.battery_capacity
            net_load  = residual_load
        else:
            discharge_kw = min(abs(action) * self.max_discharge_rate,
                               (self.soc - self.soc_min) * self.battery_capacity)
            self.soc    -= discharge_kw / self.battery_capacity
            net_load     = max(0.0, residual_load - discharge_kw * self.discharge_efficiency)
        self.soc  = np.clip(self.soc, self.soc_min, self.soc_max)
        ens_step  = min(max(0.0, net_load), load_kw)
        self.ens += ens_step
        cycle_depth      = abs(self.soc - soc_before)
        self.efc        += cycle_depth
        calendar_stress  = max(0.0, self.soc - 0.85) * self.calendar_deg_rate
        self.soh         = max(0.0, self.soh - (cycle_depth * self.soh_deg_per_cycle + calendar_stress))
        self.rul         = self.soh / self.initial_soh
        ens_ratio = ens_step / max(load_kw, 1.0)
        reward  =  1.0
        reward -= (ens_step > 0)   * 4.0
        reward -= ens_ratio        * 4.0
        reward -= cycle_depth      * 2.0
        reward -= (self.soc > 0.85)* 1.5
        reward -= (self.soc < 0.30)* 3.0
        reward -= (self.soc < 0.20)* 10.0
        reward  = np.clip(reward, -15.0, 2.0)
        self.step_idx += 1; self.current_step += 1
        done = self.current_step >= self.episode_length
        return self._get_obs(), reward, done, False, {
            'soc':self.soc,'soh':self.soh,'rul':self.rul,
            'ens':ens_step,'solar_kw':solar_kw,'load_kw':load_kw,'action':action}

def rule_based_action(soc, solar_kw, load_kw, soc_min=0.20, soc_max=0.90):
    net = solar_kw - load_kw
    if net > 0:
        return np.array([min(net/100.0, 1.0)]) if soc < soc_max else np.array([0.0])
    else:
        return np.array([-1.0]) if soc > soc_min else np.array([0.0])

def evaluate(df_loc, controller, location_code=3, ppo_model=None, vec_norm=None):
    print(f'    Running 6-year simulation ({len(df_loc)} steps max)...')
    env    = MiniGridEnv(df_loc, lstm_model, scaler_X, scaler_y, location_code)
    obs, _ = env.reset()
    history = {'soc':[],'soh':[],'ens':[],'solar_kw':[],'load_kw':[],'action':[]}
    done = False; step = 0
    while not done:
        if controller == 'AI':
            obs_norm = vec_norm.normalize_obs(obs)
            action, _ = ppo_model.predict(obs_norm, deterministic=True)
        else:
            action = rule_based_action(
                env.soc,
                (env.df.iloc[env.step_idx]['ssrd_wm2'] / 1000.0) * 750.0 * 0.75,
                env.df.iloc[env.step_idx]['load_kw'])
        obs, _, done, _, info = env.step(action)
        for k in history: history[k].append(info[k])
        step += 1
        if step % 8760 == 0:
            year = step // 8760
            print(f'      ✅ Year {year}/6 | SOC:{info["soc"]*100:.1f}% | SOH:{info["soh"]*100:.1f}% | ENS:{sum(history["ens"]):.0f} kWh')

    ens_arr  = np.array(history['ens'])
    soc_arr  = np.array(history['soc'])
    soh_arr  = np.array(history['soh'])
    sol_arr  = np.array(history['solar_kw'])
    load_arr = np.array(history['load_kw'])
    act_arr  = np.array(history['action'])

    solar_used   = np.sum(np.minimum(sol_arr, load_arr))
    scr          = round(float(min(solar_used / max(np.sum(sol_arr),1)*100, 100)), 2)
    total_load   = np.sum(load_arr)
    total_ens    = np.sum(ens_arr)
    served_pct   = round(float(np.clip((1-total_ens/max(total_load,1))*100, 0, 100)), 2)
    eol_idx      = next((i for i,s in enumerate(soh_arr) if s <= 0.80), None)
    if eol_idx is not None:
        years_to_eol = round(eol_idx/8760, 1)
    else:
        total_deg    = 1.0 - float(soh_arr[-1])
        deg_per_yr   = total_deg / 6.0
        years_to_eol = round(0.20/deg_per_yr, 1) if deg_per_yr > 0 else 99.0

    return {
        'ENS': round(float(np.sum(ens_arr)),2), 'LOLP': round(float(np.mean(ens_arr>0)*100),2),
        'SOH': round(float(soh_arr[-1]*100),2),  'EFC':  round(float(np.sum(np.abs(act_arr))*0.5),2),
        'SCR': scr, 'SOC_STD': round(float(np.std(soc_arr)),4),
        'SERVED_PCT': served_pct, 'LIFESPAN_YRS': years_to_eol,
        'soc_arr':soc_arr,'soh_arr':soh_arr,'ens_arr':ens_arr,'sol_arr':sol_arr,'load_arr':load_arr
    }

# ── Load PPO ──────────────────────────────────────────────────────
print('Loading PPO model...')
ppo_model = PPO.load('../models/best_model.zip')
_dummy_env = DummyVecEnv([lambda: MiniGridEnv(
    train_df[train_df['location']=='Tamale'].reset_index(drop=True),
    lstm_model, scaler_X, scaler_y)])
vec_norm = VecNormalize.load('../models/vecnormalize_stats.pkl', _dummy_env)
vec_norm.training = False; vec_norm.norm_reward = False
print('✅ PPO loaded')

# ── Build eval dataset ────────────────────────────────────────────
print('\nBuilding evaluation dataset for new locations...')
eval_df = build_eval_dataset('../data/')

# Add time features
eval_df['datetime']      = pd.to_datetime(eval_df['datetime'])
eval_df['location_code'] = eval_df['location'].map({'Accra':3,'Bolgatanga':4,'Akosombo':5})
eval_df['hour']          = eval_df['datetime'].dt.hour
eval_df['month']         = eval_df['datetime'].dt.month
eval_df['dayofweek']     = eval_df['datetime'].dt.dayofweek

# ── Run evaluation ────────────────────────────────────────────────
new_locations  = [l for l in ['Accra','Bolgatanga','Akosombo'] if l in eval_df['location'].unique()]
loc_codes      = {'Accra':3,'Bolgatanga':4,'Akosombo':5}
results        = {}

print(f'\nEvaluating {len(new_locations)} unseen locations...')
for loc in new_locations:
    print(f'\n{"="*50}\nEvaluating {loc} (UNSEEN)\n{"="*50}')
    df_loc = eval_df[eval_df['location']==loc].reset_index(drop=True)
    code   = loc_codes[loc]
    print(f'  ▶ Rule-Based:')
    results[f'{loc}_RB'] = evaluate(df_loc, 'RB', code)
    print(f'  ▶ AI Agent:')
    results[f'{loc}_AI'] = evaluate(df_loc, 'AI', code, ppo_model, vec_norm)
    print(f'  ✅ {loc} done!')

# ── Print results ─────────────────────────────────────────────────
print('\n' + '='*90)
print(f'{"KPI COMPARISON — UNSEEN LOCATIONS (Generalisation Test)":^90}')
print('='*90)
print(f'{"Location":<14} {"Controller":<16} {"ENS(kWh)":<12} {"LOLP(%)":<10} '
      f'{"SOH(%)":<10} {"LIFESPAN":<12} {"EFC":<8} {"SERVED%":<10}')
print('-'*90)

kpi_rows = []
for loc in new_locations:
    rb = results[f'{loc}_RB']
    ai = results[f'{loc}_AI']
    print(f'{loc:<14} {"Rule-Based":<16} {rb["ENS"]:<12} {rb["LOLP"]:<10} '
          f'{rb["SOH"]:<10} {str(rb["LIFESPAN_YRS"])+"yr":<12} {rb["EFC"]:<8} {rb["SERVED_PCT"]:<10}')
    print(f'{"":14} {"AI (LSTM+PPO)":<16} {ai["ENS"]:<12} {ai["LOLP"]:<10} '
          f'{ai["SOH"]:<10} {str(ai["LIFESPAN_YRS"])+"yr":<12} {ai["EFC"]:<8} {ai["SERVED_PCT"]:<10}')
    ens_imp  = round((rb["ENS"]-ai["ENS"])/max(rb["ENS"],1)*100,1)
    lolp_imp = round(rb["LOLP"]-ai["LOLP"],1)
    soh_imp  = round(ai["SOH"]-rb["SOH"],1)
    life_imp = round(ai["LIFESPAN_YRS"]-rb["LIFESPAN_YRS"],1)
    print(f'{"":14} {"IMPROVEMENT":<16} {"▼"+str(ens_imp)+"%":<12} {"▼"+str(lolp_imp)+"%":<10} '
          f'{"▲"+str(soh_imp)+"%":<10} {"▲"+str(life_imp)+"yr":<12}')
    print('-'*90)
    for ctrl in ['RB','AI']:
        r = results[f'{loc}_{ctrl}']
        kpi_rows.append({'Location':loc,'Controller':'Rule-Based' if ctrl=='RB' else 'AI (LSTM+PPO)',
            'ENS_kWh':r['ENS'],'LOLP_%':r['LOLP'],'SOH_%':r['SOH'],
            'Lifespan_yrs':r['LIFESPAN_YRS'],'EFC':r['EFC'],
            'SCR_%':r['SCR'],'SERVED_%':r['SERVED_PCT'],'SOC_STD':r['SOC_STD']})

kpi_df = pd.DataFrame(kpi_rows)
kpi_df.to_csv('../data/kpi_results_new_locations.csv', index=False)
print('\n✅ KPI results saved → ../data/kpi_results_new_locations.csv')

# ── Plots ─────────────────────────────────────────────────────────
if new_locations:
    fig, axes = plt.subplots(len(new_locations), 3, figsize=(18, 5*len(new_locations)))
    if len(new_locations) == 1: axes = [axes]
    fig.suptitle('Generalisation Test — Unseen Locations (6-Year)', fontsize=16, fontweight='bold')
    for i, loc in enumerate(new_locations):
        rb    = results[f'{loc}_RB']
        ai    = results[f'{loc}_AI']
        steps = np.arange(len(rb['soc_arr']))
        year_ticks  = np.arange(0, len(steps)+1, 8760)
        year_labels = [f'Y{y+1}' for y in range(len(year_ticks))]
        axes[i][0].plot(steps, rb['soc_arr'], 'r', lw=0.5, label='Rule-Based', alpha=0.8)
        axes[i][0].plot(steps, ai['soc_arr'], 'g', lw=0.5, label='AI', alpha=0.8)
        axes[i][0].set_title(f'{loc} — SOC'); axes[i][0].set_xticks(year_ticks); axes[i][0].set_xticklabels(year_labels)
        axes[i][0].legend(fontsize=8); axes[i][0].grid(True, alpha=0.3)
        axes[i][1].plot(steps, rb['soh_arr'], 'r', lw=0.8, label='Rule-Based', alpha=0.8)
        axes[i][1].plot(steps, ai['soh_arr'], 'g', lw=0.8, label='AI', alpha=0.8)
        axes[i][1].set_title(f'{loc} — SOH'); axes[i][1].set_xticks(year_ticks); axes[i][1].set_xticklabels(year_labels)
        axes[i][1].legend(fontsize=8); axes[i][1].grid(True, alpha=0.3)
        axes[i][2].plot(steps, np.cumsum(rb['ens_arr']), 'r', lw=0.8, label='Rule-Based', alpha=0.8)
        axes[i][2].plot(steps, np.cumsum(ai['ens_arr']), 'g', lw=0.8, label='AI', alpha=0.8)
        axes[i][2].set_title(f'{loc} — Cumulative ENS'); axes[i][2].set_xticks(year_ticks); axes[i][2].set_xticklabels(year_labels)
        axes[i][2].legend(fontsize=8); axes[i][2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../data/evaluation_new_locations.png', dpi=150)
    print('✅ Plots saved → ../data/evaluation_new_locations.png')

print('\n🎉 Generalisation Evaluation Complete!')

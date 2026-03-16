import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
from gymnasium import spaces
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI Mini-Grid Simulation", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Space+Mono:wght@400;700&display=swap');
  .stApp, .main, .block-container { background-color: #0a0a0a !important; }
  section[data-testid="stSidebar"] { background: linear-gradient(180deg,#050f05 0%,#0a1a0a 100%) !important; }
  section[data-testid="stSidebar"] * { color: #c0e0c0 !important; }
  section[data-testid="stSidebar"] hr { border-color: #1a4a1a !important; }
  h1,h2,h3,p,div,span,label { color: #c0e0c0 !important; }
  .stButton > button { background: linear-gradient(135deg,#003a1a,#006633) !important; border: 1px solid #00cc66 !important; color: #00ff88 !important; font-family: 'Rajdhani',sans-serif !important; font-weight: 700 !important; font-size: 16px !important; letter-spacing: 2px !important; border-radius: 8px !important; width: 100% !important; padding: 12px !important; }
  .stButton > button:hover { background: linear-gradient(135deg,#005522,#00994d) !important; }
  .app-header { background: linear-gradient(135deg,#050f05,#0a2a0a,#003a1a); border: 1px solid #00cc66; border-radius: 12px; padding: 20px 28px; margin-bottom: 24px; box-shadow: 0 0 30px rgba(0,255,136,0.1); }
  .app-title { font-family:'Rajdhani',sans-serif; font-size:28px; font-weight:700; color:#00ff88 !important; letter-spacing:3px; margin:0; }
  .app-sub { font-family:'Space Mono',monospace; font-size:10px; color:rgba(0,255,136,0.5) !important; margin:6px 0 0 0; }
  .sec-label { font-family:'Space Mono',monospace; font-size:10px; font-weight:700; letter-spacing:2px; color:#00cc66 !important; text-transform:uppercase; border-left:3px solid #00cc66; padding-left:10px; margin:20px 0 12px 0; display:block; }
  .kpi-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-bottom:4px; }
  .kpi-card { background:#050f05; border-radius:10px; padding:14px 12px; border-left:4px solid #00cc66; }
  .kpi-card.red  { border-left-color:#ff4444; }
  .kpi-card.yel  { border-left-color:#ffc107; }
  .kpi-card.blue { border-left-color:#00aaff; }
  .kpi-lbl { font-family:'Space Mono',monospace; font-size:8px; color:#4a8a4a !important; letter-spacing:1px; text-transform:uppercase; margin-bottom:4px; }
  .kpi-val { font-family:'Rajdhani',sans-serif; font-size:26px; font-weight:700; color:#00ff88 !important; line-height:1; }
  .kpi-val.red  { color:#ff4444 !important; }
  .kpi-val.yel  { color:#ffc107 !important; }
  .kpi-val.blue { color:#00aaff !important; }
  .kpi-unit { font-size:12px; font-weight:400; opacity:0.6; }
  .cmp-wrap { background:#050f05; border:1px solid #1a4a1a; border-radius:12px; padding:20px; margin-top:20px; box-shadow:0 0 20px rgba(0,255,136,0.05); }
  .cmp-title { font-family:'Rajdhani',sans-serif; font-size:18px; font-weight:700; color:#00ff88 !important; letter-spacing:2px; margin-bottom:16px; }
  .cmp-table { width:100%; border-collapse:collapse; }
  .cmp-table th { font-family:'Space Mono',monospace; font-size:9px; letter-spacing:1px; text-transform:uppercase; padding:10px 14px; border-bottom:2px solid #1a4a1a; text-align:left; }
  .cmp-table th.h-metric { color:#4a8a4a !important; } .cmp-table th.h-rb  { color:#ff6666 !important; } .cmp-table th.h-ai  { color:#00ff88 !important; } .cmp-table th.h-imp { color:#ffc107 !important; }
  .cmp-table td { padding:10px 14px; font-size:13px; border-bottom:1px solid #0a1a0a; color:#a0c0a0 !important; }
  .cmp-table tr:last-child td { border-bottom:none; }
  .cmp-table td.metric { color:#7aaa7a !important; font-size:11px; font-family:'Space Mono',monospace; }
  .cmp-table td.rb-v { color:#ff6666 !important; font-family:'Rajdhani',sans-serif; font-size:16px; font-weight:700; }
  .cmp-table td.ai-v { color:#00ff88 !important; font-family:'Rajdhani',sans-serif; font-size:16px; font-weight:700; }
  .badge { display:inline-block; padding:2px 10px; border-radius:4px; font-family:'Space Mono',monospace; font-size:9px; font-weight:700; }
  .badge.good { background:#0a2a0a; border:1px solid #00cc66; color:#00ff88 !important; }
  .badge.bad  { background:#2a0a0a; border:1px solid #cc3333; color:#ff6666 !important; }
  .ctrl-ai { background:#0a2a0a; border:1px solid #00cc66; border-radius:20px; padding:4px 16px; font-family:'Space Mono',monospace; font-size:11px; color:#00ff88 !important; display:inline-block; font-weight:700; }
  .ctrl-rb { background:#2a0a0a; border:1px solid #cc3333; border-radius:20px; padding:4px 16px; font-family:'Space Mono',monospace; font-size:11px; color:#ff6666 !important; display:inline-block; font-weight:700; }
  .vsep { text-align:center; font-family:'Rajdhani',sans-serif; font-size:24px; font-weight:700; color:#1a4a1a !important; padding-top:40px; }
  .ready-box { text-align:center; padding:80px 20px; }
  .ready-icon { font-size:48px; color:#1a4a1a; }
  .ready-title { font-family:'Rajdhani',sans-serif; font-size:22px; color:#00cc66 !important; letter-spacing:3px; margin-top:16px; }
  .ready-sub { font-family:'Space Mono',monospace; font-size:11px; color:#2a6a2a !important; margin-top:10px; }
</style>
""", unsafe_allow_html=True)

class MiniGridLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, forecast, output_size):
        super().__init__()
        self.hidden_size = hidden_size; self.num_layers = num_layers
        self.forecast = forecast; self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden_size, forecast * output_size)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:,-1,:]).view(-1, self.forecast, self.output_size)

class MiniGridEnv(gym.Env):
    def __init__(self, df, lstm_model, scaler_X, scaler_y):
        super().__init__()
        self.df = df.reset_index(drop=True); self.lstm = lstm_model
        self.scaler_X = scaler_X; self.scaler_y = scaler_y
        self.battery_capacity = 2000.0; self.max_charge_rate = 400.0
        self.max_discharge_rate = 400.0; self.charge_efficiency = 0.95
        self.discharge_efficiency = 0.95; self.soc_min = 0.20; self.soc_max = 0.90
        self.initial_soh = 1.0; self.soh_deg_per_cycle = 0.00005
        self.calendar_deg_rate = 0.000002
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(52,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None, start_idx=24):
        self.soc = 0.5; self.soh = self.initial_soh; self.rul = 1.0
        self.efc = 0.0; self.ens = 0.0; self.step_idx = start_idx
        return self._get_obs(), {}

    def _get_forecast(self):
        lookback = self.df.iloc[self.step_idx-24:self.step_idx]
        X = self.scaler_X.transform(lookback[["ssrd_wm2","tp","temp_c","load_kw",
            "location_code","hour","month","dayofweek"]])
        X_t = torch.FloatTensor(X).unsqueeze(0)
        with torch.no_grad(): fc = self.lstm(X_t).numpy()[0]
        fc_inv = self.scaler_y.inverse_transform(fc.reshape(-1,2))
        return np.clip(fc_inv[:,0],0,None), np.clip(fc_inv[:,1],0,None)

    def _get_obs(self):
        sf, lf = self._get_forecast(); row = self.df.iloc[self.step_idx]
        return np.concatenate([[self.soc, self.soh], sf/1000.0, lf/50.0,
            [row["hour"]/23.0, row["month"]/12.0]]).astype(np.float32)

    def step(self, action):
        action = float(np.clip(action[0], -1.0, 1.0))
        row = self.df.iloc[self.step_idx]
        solar_kw = row["ssrd_wm2"] * 0.75; load_kw = row["load_kw"]
        soc_before = self.soc   # track SOC change for cycle depth
        if action > 0:
            charge_kw = min(action*self.max_charge_rate,
                            (self.soc_max-self.soc)*self.battery_capacity)
            self.soc += (charge_kw*self.charge_efficiency)/self.battery_capacity
            net_load = load_kw + charge_kw - solar_kw
        else:
            discharge_kw = min(abs(action)*self.max_discharge_rate,
                               (self.soc-self.soc_min)*self.battery_capacity)
            self.soc -= discharge_kw/self.battery_capacity
            net_load = load_kw - solar_kw - (discharge_kw*self.discharge_efficiency)
        self.soc  = np.clip(self.soc, self.soc_min, self.soc_max)
        ens_step  = max(0.0, net_load); self.ens += ens_step
        # ── Realistic two-factor degradation ─────────────────
        cycle_depth     = abs(self.soc - soc_before)
        self.efc       += cycle_depth
        calendar_stress = max(0.0, self.soc - 0.85) * self.calendar_deg_rate
        deg_this_step   = (cycle_depth * self.soh_deg_per_cycle) + calendar_stress
        self.soh        = max(0.0, self.soh - deg_this_step)
        self.rul        = self.soh / self.initial_soh
        # ── Reward: ENS priority > battery longevity ─────────
        ens_ratio = ens_step / max(load_kw, 1.0)
        reward  =  1.0
        reward -= (ens_step > 0)   * 4.0
        reward -= ens_ratio        * 4.0
        reward -= cycle_depth      * 2.0
        reward -= (self.soc > 0.85)* 1.5
        reward -= (self.soc < 0.30)* 3.0
        reward -= (self.soc < 0.20)* 10.0
        reward  = np.clip(reward, -15.0, 2.0)
        self.step_idx += 1
        done = self.step_idx >= len(self.df) - 25
        return self._get_obs(), reward, done, False, {
            "soc": self.soc, "soh": self.soh, "rul": self.rul,
            "ens": ens_step, "solar_kw": solar_kw, "load_kw": load_kw,
            "action": action, "datetime": str(row["datetime"])}

@st.cache_resource
def load_resources():
    df = pd.read_csv("../data/master_dataset.csv")
    df["datetime"]      = pd.to_datetime(df["datetime"])
    df["location_code"] = df["location"].map({"Tamale":0,"Kumasi":1,"Axim":2})
    df["hour"]          = df["datetime"].dt.hour
    df["month"]         = df["datetime"].dt.month
    df["dayofweek"]     = df["datetime"].dt.dayofweek
    scaler_X = MinMaxScaler(); scaler_y = MinMaxScaler()
    scaler_X.fit(df[["ssrd_wm2","tp","temp_c","load_kw","location_code","hour","month","dayofweek"]])
    scaler_y.fit(df[["ssrd_wm2","load_kw"]])
    lstm = MiniGridLSTM(8, 128, 2, 24, 2)
    lstm.load_state_dict(torch.load("../models/best_lstm.pth", map_location="cpu"))
    lstm.eval()
    ppo = PPO.load("../models/best_model.zip")
    _dummy = DummyVecEnv([lambda: MiniGridEnv(
        df[df["location"]=="Tamale"].reset_index(drop=True),
        lstm, scaler_X, scaler_y)])
    vec_norm = VecNormalize.load("../models/vecnormalize_stats.pkl", _dummy)
    vec_norm.training    = False
    vec_norm.norm_reward = False
    return df, scaler_X, scaler_y, lstm, ppo, vec_norm

def rule_based_action(soc, solar_kw, load_kw, soc_min=0.20, soc_max=0.90):
    net = solar_kw - load_kw
    if net > 0:
        return np.array([min(net/100.0, 1.0)]) if soc < soc_max else np.array([0.0])
    else:
        return np.array([-1.0]) if soc > soc_min else np.array([0.0])

def _precompute_forecasts(loc_df, lstm, scaler_X, scaler_y, start_idx, steps):
    """Batch all LSTM forward passes upfront — major speedup."""
    feat_cols = ["ssrd_wm2","tp","temp_c","load_kw","location_code","hour","month","dayofweek"]
    X_all = scaler_X.transform(loc_df[feat_cols].values)
    indices = range(start_idx, min(start_idx + steps + 24, len(loc_df) - 25))
    windows = np.stack([X_all[i-24:i] for i in indices])
    all_fc = []
    with torch.no_grad():
        for i in range(0, len(windows), 256):
            batch = torch.FloatTensor(windows[i:i+256])
            all_fc.append(lstm(batch).numpy())
    all_fc = np.concatenate(all_fc, axis=0)
    forecasts = {}
    for i, idx in enumerate(indices):
        fc_inv = scaler_y.inverse_transform(all_fc[i].reshape(-1,2))
        forecasts[idx] = (np.clip(fc_inv[:,0],0,None), np.clip(fc_inv[:,1],0,None))
    return forecasts

def run_simulation(df, lstm, scaler_X, scaler_y, ppo, vec_norm, location, controller, years):
    loc_df = df[df["location"]==location].reset_index(drop=True)
    steps  = min(years * 365 * 24, len(loc_df) - 50)
    start_idx = 24
    forecasts    = _precompute_forecasts(loc_df, lstm, scaler_X, scaler_y, start_idx, steps)
    solar_arr    = (loc_df["ssrd_wm2"].values / 1000.0) * 750.0 * 0.75
    load_arr_all = loc_df["load_kw"].values
    hour_arr     = loc_df["hour"].values
    month_arr    = loc_df["month"].values
    dt_arr       = loc_df["datetime"].astype(str).values
    soc_out  = np.zeros(steps); soh_out  = np.zeros(steps)
    ens_out  = np.zeros(steps); act_out  = np.zeros(steps)
    sol_out  = np.zeros(steps); load_out = np.zeros(steps)
    dt_out   = np.empty(steps, dtype=object)
    soc = 0.5; soh = 1.0
    battery_capacity = 2000.0; max_rate = 400.0
    charge_eff = 0.95; discharge_eff = 0.95
    soc_min = 0.20; soc_max = 0.90
    soh_deg = 0.00005; cal_deg = 0.000002
    for i in range(steps):
        idx      = start_idx + i
        solar_kw = solar_arr[idx]
        load_kw  = load_arr_all[idx]
        sf, lf   = forecasts.get(idx, (np.zeros(24), np.zeros(24)))
        obs = np.concatenate([[soc, soh], sf/1000.0, lf/50.0,
            [hour_arr[idx]/23.0, month_arr[idx]/12.0]]).astype(np.float32)
        if controller == "AI":
            obs_norm = vec_norm.normalize_obs(obs)
            action, _ = ppo.predict(obs_norm, deterministic=True)
            action = float(np.clip(action[0], -1.0, 1.0))
        else:
            action = float(rule_based_action(soc, solar_kw, load_kw)[0])
        soc_before    = soc
        residual_load = max(0.0, load_kw - solar_kw)
        if action > 0:
            charge_kw = min(action*max_rate, (soc_max-soc)*battery_capacity)
            soc      += (charge_kw*charge_eff)/battery_capacity
            net_load  = residual_load
        else:
            discharge_kw = min(abs(action)*max_rate, (soc-soc_min)*battery_capacity)
            soc         -= discharge_kw/battery_capacity
            net_load     = max(0.0, residual_load - discharge_kw*discharge_eff)
        soc         = float(np.clip(soc, soc_min, soc_max))
        ens_step    = max(0.0, net_load)
        cycle_depth = abs(soc - soc_before)
        cal_stress  = max(0.0, soc - 0.85) * cal_deg
        soh         = max(0.0, soh - cycle_depth*soh_deg - cal_stress)
        soc_out[i]=soc; soh_out[i]=soh; ens_out[i]=ens_step
        act_out[i]=action; sol_out[i]=solar_kw; load_out[i]=load_kw; dt_out[i]=dt_arr[idx]
    return pd.DataFrame({"datetime":dt_out,"soc":soc_out,"soh":soh_out,
        "ens":ens_out,"solar_kw":sol_out,"load_kw":load_out,"action":act_out})

def compute_kpis(res):
    ens_arr  = res["ens"].values
    sol_arr  = res["solar_kw"].values
    load_arr = res["load_kw"].values
    act_arr  = res["action"].values
    soh_arr  = res["soh"].values
    soc_arr  = res["soc"].values
    solar_used = np.sum(np.minimum(sol_arr, load_arr))
    scr = min(solar_used / max(np.sum(sol_arr), 1) * 100, 100)
    total_load = np.sum(load_arr)
    served_pct = round(float((1 - np.sum(ens_arr)/max(total_load,1)) * 100), 1)
    # Projected lifespan
    eol_idx = next((i for i, s in enumerate(soh_arr) if s <= 0.80), None)
    if eol_idx is not None:
        lifespan = round(eol_idx / 8760, 1)
    else:
        total_deg  = 1.0 - float(soh_arr[-1])
        deg_per_yr = total_deg / max(len(soh_arr)/8760, 0.01)
        lifespan   = round(0.20 / deg_per_yr, 1) if deg_per_yr > 0 else 99.0

    return {
        "ENS":        round(float(np.sum(ens_arr)), 1),
        "LOLP":       round(float(np.mean(ens_arr > 0) * 100), 1),
        "SOH":        round(float(soh_arr[-1] * 100), 1),
        "EFC":        round(float(np.sum(np.abs(act_arr)) * 0.5), 0),
        "SCR":        round(scr, 1),
        "SOC_STD":    round(float(np.std(soc_arr)), 4),
        "SERVED_PCT": served_pct,
        "LIFESPAN":   lifespan,
    }

def dark_layout(title):
    return dict(
        title=dict(text=title, font=dict(color="#00cc66", family="Rajdhani", size=13), x=0),
        paper_bgcolor="#050f05", plot_bgcolor="#050f05",
        font=dict(color="#6a9a6a", family="Space Mono", size=9),
        margin=dict(l=40, r=10, t=35, b=30),
        xaxis=dict(gridcolor="#0a2a0a", showgrid=True, linecolor="#1a4a1a", color="#4a8a4a"),
        yaxis=dict(gridcolor="#0a2a0a", showgrid=True, linecolor="#1a4a1a", color="#4a8a4a"),
        legend=dict(bgcolor="rgba(5,15,5,0.8)", bordercolor="#1a4a1a", borderwidth=1, font=dict(color="#6a9a6a", size=9)),
        height=240,
    )

def render_kpis(kpis):
    served = kpis.get("SERVED_PCT", "N/A")
    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card red"><div class="kpi-lbl">Energy Not Served</div><div class="kpi-val red">{kpis["ENS"]:,.0f}<span class="kpi-unit"> kWh</span></div></div>
      <div class="kpi-card red"><div class="kpi-lbl">Loss of Load Prob</div><div class="kpi-val red">{kpis["LOLP"]}<span class="kpi-unit"> %</span></div></div>
      <div class="kpi-card"><div class="kpi-lbl">State of Health</div><div class="kpi-val">{kpis["SOH"]}<span class="kpi-unit"> %</span></div></div>
      <div class="kpi-card yel"><div class="kpi-lbl">Equiv. Full Cycles</div><div class="kpi-val yel">{int(kpis["EFC"]):,}</div></div>
      <div class="kpi-card blue"><div class="kpi-lbl">Solar Self-Consumption</div><div class="kpi-val blue">{kpis["SCR"]}<span class="kpi-unit"> %</span></div></div>
      <div class="kpi-card"><div class="kpi-lbl">Load Served</div><div class="kpi-val">{served}<span class="kpi-unit"> %</span></div></div>
      <div class="kpi-card blue"><div class="kpi-lbl">Projected Lifespan</div><div class="kpi-val blue">{kpis["LIFESPAN"]}<span class="kpi-unit"> yrs</span></div></div>
    </div>
    """, unsafe_allow_html=True)

def render_comparison_table(rb_kpis, ai_kpis):
    def badge(val, good):
        return f'<span class="badge {"good" if good else "bad"}">{val}</span>'
    metrics = [
        ("Energy Not Served (kWh)", "ENS",        False, "{:,.0f}"),
        ("Loss of Load Prob (%)",   "LOLP",       False, "{:.1f}%"),
        ("State of Health (%)",     "SOH",        True,  "{:.1f}%"),
        ("Projected Lifespan (yrs)","LIFESPAN",   True,  "{:.1f} yrs"),
        ("Equiv. Full Cycles",      "EFC",        False, "{:,.0f}"),
        ("Solar Self-Consumption",  "SCR",        True,  "{:.1f}%"),
        ("Load Served (%)",         "SERVED_PCT", True,  "{:.1f}%"),
        ("SOC Std Deviation",       "SOC_STD",    False, "{:.4f}"),
    ]
    rows = []
    for label, key, higher_is_better, fmt in metrics:
        rb_v = rb_kpis.get(key, 0); ai_v = ai_kpis.get(key, 0)
        if higher_is_better:
            pct  = (ai_v - rb_v) / max(abs(rb_v), 0.001) * 100
            good = pct > 0
            imp  = f"▲ {abs(pct):.1f}%" if good else f"▼ {abs(pct):.1f}%"
        else:
            pct  = (rb_v - ai_v) / max(abs(rb_v), 0.001) * 100
            good = pct > 0
            imp  = f"▼ {abs(pct):.1f}%" if good else f"▲ {abs(pct):.1f}%"
        rows.append(f'''<tr><td class="metric">{label}</td><td class="rb-v">{fmt.format(rb_v)}</td><td class="ai-v">{fmt.format(ai_v)}</td><td>{badge(imp, good)}</td></tr>''')
    st.markdown(f"""
    <div class="cmp-wrap">
      <div class="cmp-title">⚡ FINAL KPI COMPARISON</div>
      <table class="cmp-table">
        <thead><tr><th class="h-metric">Metric</th><th class="h-rb">⚙ Rule-Based</th><th class="h-ai">🤖 AI (LSTM+PPO)</th><th class="h-imp">AI Improvement</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

# ── Scene component (animation) ────────────────────────────────
def _build_scene_component(rb_json, ai_json):
    return f"""<!DOCTYPE html><html><head>
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Rajdhani:wght@600;700&display=swap');
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:#0a0a0a;font-family:'Space Mono',monospace;}}
.outer{{display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:6px;background:#0a0a0a;}}
.panel{{background:#050f05;border:1px solid #1a4a1a;border-radius:12px;overflow:hidden;}}
.hdr{{display:flex;align-items:center;justify-content:space-between;padding:6px 12px;background:rgba(0,0,0,0.5);border-bottom:1px solid #0f2a0f;}}
.hdr-t{{font-size:8px;font-weight:700;letter-spacing:2px;color:#4a8a4a;text-transform:uppercase;}}
.badge{{font-size:8px;font-weight:700;padding:2px 8px;border-radius:8px;}}
.badge-rb{{background:#2a0a0a;border:1px solid #cc3333;color:#ff6666;}}
.badge-ai{{background:#0a2a0a;border:1px solid #00cc66;color:#00ff88;}}
.sky{{width:100%;height:160px;position:relative;overflow:hidden;transition:background 2s;}}
.star{{position:absolute;background:white;border-radius:50%;animation:twinkle 3s infinite alternate;}}
@keyframes twinkle{{0%{{opacity:0.1}}100%{{opacity:0.9}}}}
.sun{{position:absolute;border-radius:50%;background:radial-gradient(circle,#fff7aa,#ffd700 40%,#ff8c00);transition:all 1.5s ease;}}
.moon{{position:absolute;border-radius:50%;background:radial-gradient(circle,#e8e8d0,#c0c0a0);transition:all 1.5s ease;}}
.horiz{{position:absolute;bottom:0;left:0;right:0;height:30px;background:linear-gradient(transparent,#050f05);}}
.cloud{{position:absolute;border-radius:20px;background:rgba(150,180,255,0.06);animation:drift linear infinite;}}
@keyframes drift{{0%{{left:-120px}}100%{{left:110%}}}}
.gnd{{background:linear-gradient(#0a1a0a,#050f05);border-top:1px solid #0d2a0d;padding:8px 10px;display:grid;grid-template-columns:1fr auto 1fr;gap:8px;align-items:center;}}
.solar-arr{{display:flex;gap:2px;align-items:flex-end;}}
.sp{{width:15px;height:22px;background:linear-gradient(135deg,#0a1050,#1a2590);border:1px solid #2a3aaa;border-radius:2px;transition:all 0.8s;}}
.sp.on{{background:linear-gradient(135deg,#1a20a0,#3040dd);box-shadow:0 0 8px rgba(80,120,255,0.7);}}
.skw{{font-family:'Rajdhani',sans-serif;font-size:12px;font-weight:700;color:#ffc107;text-align:center;margin-top:2px;}}
.slbl{{font-size:7px;color:#2a6a2a;text-align:center;letter-spacing:1px;}}
.bc{{display:flex;flex-direction:column;align-items:center;gap:1px;}}
.btp{{width:10px;height:4px;background:#00cc66;border-radius:2px 2px 0 0;margin:0 auto;}}
.bb{{width:36px;height:60px;border:2px solid #00cc66;border-radius:0 0 4px 4px;background:#020a02;position:relative;overflow:hidden;}}
.bf{{position:absolute;bottom:0;left:0;right:0;transition:height 1s,background 1s;border-radius:0 0 2px 2px;}}
.bpct{{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:8px;font-weight:700;color:white;z-index:2;text-shadow:0 0 4px rgba(0,0,0,0.9);}}
.blbl{{font-size:7px;color:#2a6a2a;letter-spacing:1px;margin-top:2px;text-align:center;}}
.bsoh{{font-family:'Rajdhani',sans-serif;font-size:11px;font-weight:700;color:#00cc66;text-align:center;}}
.houses{{display:flex;gap:5px;align-items:flex-end;justify-content:flex-end;}}
.house{{display:flex;flex-direction:column;align-items:center;}}
.roof{{width:0;height:0;border-left:12px solid transparent;border-right:12px solid transparent;border-bottom:10px solid #1a4a1a;}}
.big .roof{{border-left-width:16px;border-right-width:16px;border-bottom-width:13px;}}
.wall{{background:#0a1a0a;border:1px solid #1a3a1a;display:flex;align-items:center;justify-content:center;}}
.win{{width:6px;height:6px;border-radius:1px;transition:all 1s;}}
.win.on{{background:#ffe066;box-shadow:0 0 5px #ffe066;}}
.win.off{{background:#0a1a0a;}}
.lkw{{font-family:'Rajdhani',sans-serif;font-size:12px;font-weight:700;color:#ff6666;text-align:center;margin-top:2px;}}
.llbl{{font-size:7px;color:#2a6a2a;text-align:center;letter-spacing:1px;}}
.ticker{{display:flex;border-top:1px solid #0f2a0f;}}
.ti{{flex:1;padding:5px 6px;text-align:center;border-right:1px solid #0f2a0f;}}
.ti:last-child{{border-right:none;}}
.tl{{font-size:6px;color:#2a6a2a;letter-spacing:1px;text-transform:uppercase;}}
.tv{{font-family:'Rajdhani',sans-serif;font-size:13px;font-weight:700;margin-top:1px;}}
.green{{color:#00cc66;}}.yellow{{color:#ffc107;}}.red{{color:#ff4444;}}.blue{{color:#00aaff;}}
.ctrl{{display:flex;align-items:center;gap:8px;padding:5px 12px;border-top:1px solid #0f2a0f;background:rgba(0,0,0,0.4);}}
.ctrl-lbl{{font-size:7px;color:#2a6a2a;white-space:nowrap;}}
.sbtn{{padding:2px 8px;border-radius:3px;border:1px solid #1a4a1a;cursor:pointer;font-size:7px;font-family:'Space Mono',monospace;background:#050f05;color:#4a8a4a;transition:all 0.2s;}}
.sbtn.active{{background:#0a2a0a;border-color:#00cc66;color:#00ff88;}}
.sinfo{{font-size:7px;color:#2a6a2a;margin-left:auto;white-space:nowrap;}}
</style></head><body>
<div class="outer">
  <div class="panel">
    <div class="hdr"><span class="hdr-t">RULE-BASED CONTROLLER</span><span class="badge badge-rb">⚙ RULE-BASED</span></div>
    <div class="sky" id="sky_rb"><div id="stars_rb"></div><div class="sun" id="sun_rb" style="width:26px;height:26px;top:18px;left:20%;"></div><div class="moon" id="moon_rb" style="width:18px;height:18px;top:14px;left:80%;opacity:0;"></div><div class="cloud" style="width:65px;height:16px;top:28px;animation-duration:26s;animation-delay:-6s;"></div><div class="horiz"></div></div>
    <div class="gnd">
      <div><div class="solar-arr" id="sarr_rb"><div class="sp" id="sp_rb_0"></div><div class="sp" id="sp_rb_1"></div><div class="sp" id="sp_rb_2"></div><div class="sp" id="sp_rb_3"></div><div class="sp" id="sp_rb_4"></div><div class="sp" id="sp_rb_5"></div></div><div class="skw" id="skw_rb">0.0 kW</div><div class="slbl">SOLAR PV</div></div>
      <div class="bc"><div class="btp"></div><div class="bb"><div class="bf" id="bf_rb" style="height:50%;background:#00cc66;"></div><div class="bpct" id="bpct_rb">50%</div></div><div class="blbl">BATTERY</div><div class="bsoh" id="bsoh_rb">100.0%</div></div>
      <div><div class="houses"><div class="house"><div class="roof"></div><div class="wall" style="width:22px;height:16px;"><div class="win on" id="w_rb_0"></div></div></div><div class="house big"><div class="roof"></div><div class="wall" style="width:28px;height:20px;"><div class="win on" id="w_rb_1"></div></div></div><div class="house"><div class="roof"></div><div class="wall" style="width:22px;height:16px;"><div class="win on" id="w_rb_2"></div></div></div></div><div class="lkw" id="lkw_rb">0.0 kW</div><div class="llbl">LOAD</div></div>
    </div>
    <div class="ticker"><div class="ti"><div class="tl">CUM ENS</div><div class="tv red" id="tens_rb">0 kWh</div></div><div class="ti"><div class="tl">ACTION</div><div class="tv blue" id="tact_rb">0.00</div></div><div class="ti"><div class="tl">HOUR</div><div class="tv green" id="thr_rb">H00</div></div><div class="ti"><div class="tl">DAY</div><div class="tv yellow" id="tday_rb">D1</div></div></div>
    <div class="ctrl"><span class="ctrl-lbl">SPEED:</span><button class="sbtn active" onclick="setSpeed(60)">1×</button><button class="sbtn" onclick="setSpeed(25)">2×</button><button class="sbtn" onclick="setSpeed(8)">5×</button><button class="sbtn" onclick="setSpeed(1)">MAX</button><span class="sinfo" id="sinfo">Loading...</span></div>
  </div>
  <div class="panel">
    <div class="hdr"><span class="hdr-t">AI AGENT · LSTM + PPO</span><span class="badge badge-ai">🤖 AI AGENT</span></div>
    <div class="sky" id="sky_ai"><div id="stars_ai"></div><div class="sun" id="sun_ai" style="width:26px;height:26px;top:18px;left:20%;"></div><div class="moon" id="moon_ai" style="width:18px;height:18px;top:14px;left:80%;opacity:0;"></div><div class="cloud" style="width:55px;height:14px;top:32px;animation-duration:29s;animation-delay:-9s;"></div><div class="horiz"></div></div>
    <div class="gnd">
      <div><div class="solar-arr" id="sarr_ai"><div class="sp" id="sp_ai_0"></div><div class="sp" id="sp_ai_1"></div><div class="sp" id="sp_ai_2"></div><div class="sp" id="sp_ai_3"></div><div class="sp" id="sp_ai_4"></div><div class="sp" id="sp_ai_5"></div></div><div class="skw" id="skw_ai">0.0 kW</div><div class="slbl">SOLAR PV</div></div>
      <div class="bc"><div class="btp"></div><div class="bb"><div class="bf" id="bf_ai" style="height:50%;background:#00cc66;"></div><div class="bpct" id="bpct_ai">50%</div></div><div class="blbl">BATTERY</div><div class="bsoh" id="bsoh_ai">100.0%</div></div>
      <div><div class="houses"><div class="house"><div class="roof"></div><div class="wall" style="width:22px;height:16px;"><div class="win on" id="w_ai_0"></div></div></div><div class="house big"><div class="roof"></div><div class="wall" style="width:28px;height:20px;"><div class="win on" id="w_ai_1"></div></div></div><div class="house"><div class="roof"></div><div class="wall" style="width:22px;height:16px;"><div class="win on" id="w_ai_2"></div></div></div></div><div class="lkw" id="lkw_ai">0.0 kW</div><div class="llbl">LOAD</div></div>
    </div>
    <div class="ticker"><div class="ti"><div class="tl">CUM ENS</div><div class="tv red" id="tens_ai">0 kWh</div></div><div class="ti"><div class="tl">ACTION</div><div class="tv blue" id="tact_ai">0.00</div></div><div class="ti"><div class="tl">HOUR</div><div class="tv green" id="thr_ai">H00</div></div><div class="ti"><div class="tl">DAY</div><div class="tv yellow" id="tday_ai">D1</div></div></div>
    <div class="ctrl" style="border-top:1px solid #0f2a0f;background:rgba(0,0,0,0.4);display:flex;align-items:center;padding:5px 12px;"><span class="ctrl-lbl">STATUS:</span><span id="aistat" style="font-size:7px;color:#00cc66;margin-left:8px;">SYNCED</span></div>
  </div>
</div>
<script>
const rbD={rb_json};const aiD={ai_json};
let step=0,ivl=null,spd=60,cumRb=0,cumAi=0;
const N=rbD?rbD.length:0;
['rb','ai'].forEach(id=>{{const c=document.getElementById('stars_'+id);if(!c)return;for(let i=0;i<50;i++){{const s=document.createElement('div');s.className='star';const sz=Math.random()*1.5+0.5;s.style.cssText=`width:${{sz}}px;height:${{sz}}px;left:${{Math.random()*100}}%;top:${{Math.random()*65}}%;animation-delay:${{Math.random()*3}}s;animation-duration:${{2+Math.random()*2}}s`;c.appendChild(s);}}}});
function battCol(s){{return s>0.55?'#00cc66':s>0.3?'#ffc107':'#ff4444';}}
function skyBg(hr){{if(hr>=6&&hr<=8)return 'linear-gradient(180deg,#ff6a00,#ee9c2a 40%,#1a3a1a)';if(hr>=9&&hr<=16)return 'linear-gradient(180deg,#1a6ebc,#4ca3e0 40%,#1a3a1a)';if(hr>=17&&hr<=19)return 'linear-gradient(180deg,#c2410c,#f97316 40%,#1a3a1a)';return 'linear-gradient(180deg,#000008,#000515 40%,#050f05)';}}
function sunPos(hr){{if(hr<6||hr>18)return null;const t=(hr-6)/12;return{{x:t*80+5,y:Math.max(6,80-Math.sin(t*Math.PI)*110)}};}}
function moonPos(hr){{let t=null;if(hr>=18)t=(hr-18)/12;else if(hr<6)t=(hr+6)/12;else return null;return{{x:t*80+5,y:Math.max(6,55-Math.sin(t*Math.PI)*60)}};}}
function upd(id,d,cumEns){{
  if(!d)return;
  const hr=(N>0)?(((step*4)%24+6)%24):(step%24);
  const soc=d.soc,solar=d.solar_kw,load=d.load_kw,act=d.action,soh=d.soh,ens=d.ens;
  const sky=document.getElementById('sky_'+id);if(sky)sky.style.background=skyBg(hr);
  const stars=document.getElementById('stars_'+id);if(stars)stars.style.opacity=(hr<6||hr>19)?'1':(hr<8||hr>17)?'0.4':'0';
  const sun=document.getElementById('sun_'+id);const sp=sunPos(hr);
  if(sun){{if(sp){{sun.style.left=sp.x+'%';sun.style.top=sp.y+'px';sun.style.opacity='1';const sz=22+Math.sin((hr-6)/12*Math.PI)*12;sun.style.width=sz+'px';sun.style.height=sz+'px';const glo=solar>0?Math.min(solar/20,1):0;sun.style.boxShadow=`0 0 ${{16+glo*35}}px #ffd700,0 0 ${{30+glo*50}}px rgba(255,200,0,${{0.15+glo*0.35}})`;}}else sun.style.opacity='0';}}
  const moon=document.getElementById('moon_'+id);const mp=moonPos(hr);
  if(moon){{if(mp){{moon.style.left=mp.x+'%';moon.style.top=mp.y+'px';moon.style.opacity='0.8';}}else moon.style.opacity='0';}}
  const nOn=solar>0?Math.ceil((solar/22)*6):0;for(let i=0;i<6;i++){{const p=document.getElementById('sp_'+id+'_'+i);if(p)p.classList.toggle('on',i<nOn);}}
  const sk=document.getElementById('skw_'+id);if(sk)sk.textContent=solar.toFixed(1)+' kW';
  const pct=Math.round(soc*100);const col=battCol(soc);
  const bf=document.getElementById('bf_'+id);const bp=document.getElementById('bpct_'+id);const bs=document.getElementById('bsoh_'+id);
  if(bf){{bf.style.height=pct+'%';bf.style.background=col;}}if(bp)bp.textContent=pct+'%';if(bs){{bs.textContent=(soh*100).toFixed(1)+'%';bs.style.color=col;}}
  const night=hr<6||hr>20;const lit=ens<0.5&&night;
  for(let i=0;i<3;i++){{const w=document.getElementById('w_'+id+'_'+i);if(w){{w.classList.toggle('on',lit);w.classList.toggle('off',!lit);}}}}
  const lk=document.getElementById('lkw_'+id);if(lk)lk.textContent=load.toFixed(1)+' kW';
  const te=document.getElementById('tens_'+id);const ta=document.getElementById('tact_'+id);const th=document.getElementById('thr_'+id);const td=document.getElementById('tday_'+id);
  if(te)te.textContent=cumEns.toFixed(0)+' kWh';
  if(ta){{ta.textContent=(act>=0?'+':'')+act.toFixed(2);ta.style.color=act>0.05?'#00cc66':act<-0.05?'#ff4444':'#888';}}
  if(th)th.textContent='H'+String(hr).padStart(2,'0');
  if(td)td.textContent='D'+Math.floor(step*4/24+1);
}}
function tick(){{
  if(!rbD||!aiD||step>=N){{if(step>=N&&N>0){{clearInterval(ivl);document.getElementById('sinfo').textContent='✅ Complete — '+N*4+' steps played';}}return;}}
  const rd=rbD[step],ad=aiD[step];cumRb+=(rd.ens||0);cumAi+=(ad.ens||0);
  upd('rb',rd,cumRb);upd('ai',ad,cumAi);
  document.getElementById('sinfo').textContent='Step '+(step*4)+'/'+N*4+' · Yr '+(Math.floor(step*4/8760)+1);
  step++;
}}
function setSpeed(ms){{spd=ms;document.querySelectorAll('.sbtn').forEach(b=>b.classList.remove('active'));event.target.classList.add('active');if(ivl)clearInterval(ivl);ivl=setInterval(tick,ms);}}
if(rbD&&aiD){{
  ivl=setInterval(tick,spd);document.getElementById('aistat').textContent='SYNCED WITH RULE-BASED';
}}else{{
  let dHr=6,dSocRb=0.5,dSocAi=0.5,dSohRb=1.0,dSohAi=1.0,dCumRb=0,dCumAi=0;
  document.getElementById('sinfo').textContent='⏳ Simulation running — demo animation...';
  document.getElementById('aistat').textContent='DEMO MODE';
  setInterval(()=>{{
    dHr=(dHr+1)%24;const day=dHr>=6&&dHr<=18;
    const solar=day?Math.max(0,Math.sin((dHr-6)/12*Math.PI)*22):0;const load=14+Math.sin(dHr/24*Math.PI*2+1)*7;
    if(day&&solar>load){{dSocRb=Math.min(0.90,dSocRb+0.03);dSocAi=Math.min(0.90,dSocAi+0.025);}}
    else{{dSocRb=Math.max(0.20,dSocRb-0.045);dSocAi=Math.max(0.25,dSocAi-0.02);}}
    dSohRb=Math.max(0.70,dSohRb-0.0001);dSohAi=Math.max(0.73,dSohAi-0.00007);
    const rbEns=dSocRb<=0.21?load*0.4:0;dCumRb+=rbEns;
    const fRb={{soc:dSocRb,soh:dSohRb,ens:rbEns,solar_kw:solar,load_kw:load,action:day?0.3:-0.9}};
    const fAi={{soc:dSocAi,soh:dSohAi,ens:0,solar_kw:solar,load_kw:load,action:day?0.2:-0.25}};
    const sv=step;step=dHr;upd('rb',fRb,dCumRb);upd('ai',fAi,dCumAi);step=sv;
  }},800);
}}
</script></body></html>"""

def build_scene_html(rb_res=None, ai_res=None):
    rb_json = 'null'; ai_json = 'null'
    if rb_res is not None:
        cols = ['soc','soh','ens','solar_kw','load_kw','action']
        rb_json = rb_res[cols].iloc[::4].reset_index(drop=True).to_json(orient='records')
    if ai_res is not None:
        cols = ['soc','soh','ens','solar_kw','load_kw','action']
        ai_json = ai_res[cols].iloc[::4].reset_index(drop=True).to_json(orient='records')
    return rb_json, ai_json

# ══ LOAD ══════════════════════════════════════════════════════
with st.spinner('Loading models...'):
    df, scaler_X, scaler_y, lstm, ppo, vec_norm = load_resources()

# ══ HEADER ════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
  <p class="app-title">⚡ AI SMART BATTERY ORCHESTRATION</p>
  <p class="app-sub">SIDE-BY-SIDE SIMULATION · RULE-BASED vs AI (LSTM+PPO) · GHANA MINI-GRID</p>
</div>
""", unsafe_allow_html=True)

# ══ SIDEBAR ═══════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙ SIMULATION SETTINGS")
    st.markdown("---")
    location = st.selectbox("📍 Location", ['Tamale', 'Kumasi', 'Axim'])
    years    = st.selectbox("📅 Simulation Period", [1,2,3,4,5,6],
                            format_func=lambda x: f"{x} Year{'s' if x>1 else ''}")
    st.markdown("---")
    run_btn  = st.button("▶  RUN SIMULATION")
    st.markdown("---")
    loc_info = {
        'Tamale': {'region':'Northern', 'climate':'Hot & Semi-Arid', 'lat':'9.40°N', 'lon':'0.85°W'},
        'Kumasi': {'region':'Ashanti',  'climate':'Humid Tropical',  'lat':'6.67°N', 'lon':'1.62°W'},
        'Axim':   {'region':'Western',  'climate':'Coastal Wet',     'lat':'4.87°N', 'lon':'2.24°W'},
    }
    info = loc_info[location]
    st.markdown(f"""
    **📍 {location}**
    - Region: {info['region']}
    - Climate: {info['climate']}
    - Coordinates: {info['lat']}, {info['lon']}
    - Battery: 400 kWh LiFePO₄
    - Solar PV: 150 m²
    - Steps: {years*365*24:,} hours
    """)

# ══ ANIMATION (always visible — demo before run, real after) ══
st.markdown('<span class="sec-label">LIVE MINI-GRID ANIMATION</span>', unsafe_allow_html=True)
if 'rb_res' in st.session_state:
    rb_json, ai_json = build_scene_html(st.session_state['rb_res'], st.session_state['ai_res'])
else:
    rb_json, ai_json = 'null', 'null'
scene_slot = st.empty()
with scene_slot:
    st.components.v1.html(_build_scene_component(rb_json, ai_json), height=430, scrolling=False)

# ══ RUN ═══════════════════════════════════════════════════════
if run_btn:
    prog = st.progress(0, text="Starting...")
    prog.progress(5,  text=f"⚙ Running Rule-Based for {location} ({years} yr)...")
    rb_res = run_simulation(df, lstm, scaler_X, scaler_y, ppo, vec_norm, location, 'RB', years)
    prog.progress(55, text=f"🤖 Running AI (LSTM+PPO) for {location} ({years} yr)...")
    ai_res = run_simulation(df, lstm, scaler_X, scaler_y, ppo, vec_norm, location, 'AI', years)
    prog.progress(95, text="Computing KPIs...")
    if years == 6:
        try:
            kpi_df  = pd.read_csv("../data/kpi_results.csv")
            rb_row  = kpi_df[(kpi_df["Location"]==location) & (kpi_df["Controller"]=="Rule-Based")].iloc[0]
            ai_row  = kpi_df[(kpi_df["Location"]==location) & (kpi_df["Controller"]=="AI (LSTM+PPO)")].iloc[0]
            rb_kpis = {"ENS": rb_row["ENS_kWh"], "LOLP": rb_row["LOLP_%"],
                       "SOH": rb_row["SOH_%"], "LIFESPAN": rb_row["Lifespan_yrs"],
                       "EFC": rb_row["EFC"],   "SCR": rb_row["SCR_%"],
                       "SERVED_PCT": rb_row["SERVED_%"], "SOC_STD": rb_row["SOC_STD"]}
            ai_kpis = {"ENS": ai_row["ENS_kWh"], "LOLP": ai_row["LOLP_%"],
                       "SOH": ai_row["SOH_%"], "LIFESPAN": ai_row["Lifespan_yrs"],
                       "EFC": ai_row["EFC"],   "SCR": ai_row["SCR_%"],
                       "SERVED_PCT": ai_row["SERVED_%"], "SOC_STD": ai_row["SOC_STD"]}
        except Exception as e:
            st.warning(f"⚠️ Could not load KPI CSV ({e}) — computing from simulation")
            rb_kpis = compute_kpis(rb_res); ai_kpis = compute_kpis(ai_res)
    else:
        rb_kpis = compute_kpis(rb_res); ai_kpis = compute_kpis(ai_res)
    prog.progress(100, text="Complete!"); prog.empty()
    st.session_state.update({'rb_res':rb_res,'ai_res':ai_res,
        'rb_kpis':rb_kpis,'ai_kpis':ai_kpis,'location':location,'years':years})
    rb_json, ai_json = build_scene_html(rb_res, ai_res)
    with scene_slot:
        st.components.v1.html(_build_scene_component(rb_json, ai_json), height=430, scrolling=False)

# ══ RESULTS ═══════════════════════════════════════════════════
if 'rb_res' in st.session_state:
    rb_res  = st.session_state['rb_res']; ai_res  = st.session_state['ai_res']
    rb_kpis = st.session_state['rb_kpis']; ai_kpis = st.session_state['ai_kpis']
    sim_loc = st.session_state['location']; sim_yrs = st.session_state['years']

    st.markdown(f'<span class="sec-label">RESULTS — {sim_loc} · {sim_yrs} Year{"s" if sim_yrs>1 else ""}</span>', unsafe_allow_html=True)

    col_rb, col_vs, col_ai = st.columns([10, 1, 10])
    with col_rb:
        st.markdown('<span class="ctrl-rb">⚙ RULE-BASED CONTROLLER</span>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True); render_kpis(rb_kpis)
    with col_vs:
        st.markdown('<div class="vsep">VS</div>', unsafe_allow_html=True)
    with col_ai:
        st.markdown('<span class="ctrl-ai">🤖 AI · LSTM + PPO</span>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True); render_kpis(ai_kpis)

    st.markdown('<span class="sec-label">BATTERY STATE OF CHARGE (%)</span>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=rb_res['soc']*100, line=dict(color='#ff6666',width=0.8), fill='tozeroy', fillcolor='rgba(255,100,100,0.05)', name='SOC %'))
        fig.add_hline(y=20, line_dash="dash", line_color="#ff4444", line_width=1)
        fig.add_hline(y=90, line_dash="dash", line_color="#333333", line_width=1)
        fig.update_layout(**dark_layout('⚙ Rule-Based — SOC (%)')); st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=ai_res['soc']*100, line=dict(color='#00cc66',width=0.8), fill='tozeroy', fillcolor='rgba(0,204,102,0.05)', name='SOC %'))
        fig.add_hline(y=20, line_dash="dash", line_color="#ff4444", line_width=1)
        fig.add_hline(y=90, line_dash="dash", line_color="#333333", line_width=1)
        fig.update_layout(**dark_layout('🤖 AI Agent — SOC (%)')); st.plotly_chart(fig, use_container_width=True)

    st.markdown('<span class="sec-label">BATTERY STATE OF HEALTH (%)</span>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=rb_res['soh']*100, line=dict(color='#ff6666',width=1.5), fill='tozeroy', fillcolor='rgba(255,100,100,0.05)', name='SOH %'))
        fig.update_layout(**dark_layout('⚙ Rule-Based — SOH (%)')); st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=ai_res['soh']*100, line=dict(color='#00cc66',width=1.5), fill='tozeroy', fillcolor='rgba(0,204,102,0.05)', name='SOH %'))
        fig.update_layout(**dark_layout('🤖 AI Agent — SOH (%)')); st.plotly_chart(fig, use_container_width=True)

    st.markdown('<span class="sec-label">CUMULATIVE ENERGY NOT SERVED (kWh)</span>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=np.cumsum(rb_res['ens']), line=dict(color='#ff6666',width=1.5), fill='tozeroy', fillcolor='rgba(255,100,100,0.05)', name='Cumulative ENS'))
        fig.update_layout(**dark_layout('⚙ Rule-Based — Cumulative ENS')); st.plotly_chart(fig, use_container_width=True)
    with c6:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=np.cumsum(ai_res['ens']), line=dict(color='#00cc66',width=1.5), fill='tozeroy', fillcolor='rgba(0,204,102,0.05)', name='Cumulative ENS'))
        fig.update_layout(**dark_layout('🤖 AI Agent — Cumulative ENS')); st.plotly_chart(fig, use_container_width=True)

    days_30 = min(30*24, len(rb_res))
    st.markdown('<span class="sec-label">SOLAR vs LOAD — FIRST 30 DAYS (kW)</span>', unsafe_allow_html=True)
    c7, c8 = st.columns(2)
    with c7:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=rb_res['solar_kw'].values[:days_30], line=dict(color='#ffc107',width=1), name='Solar kW'))
        fig.add_trace(go.Scatter(y=rb_res['load_kw'].values[:days_30],  line=dict(color='#ff6666',width=1), name='Load kW'))
        fig.update_layout(**dark_layout('⚙ Rule-Based — Solar vs Load')); st.plotly_chart(fig, use_container_width=True)
    with c8:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=ai_res['solar_kw'].values[:days_30], line=dict(color='#ffc107',width=1), name='Solar kW'))
        fig.add_trace(go.Scatter(y=ai_res['load_kw'].values[:days_30],  line=dict(color='#00cc66',width=1), name='Load kW'))
        fig.update_layout(**dark_layout('🤖 AI Agent — Solar vs Load')); st.plotly_chart(fig, use_container_width=True)

    render_comparison_table(rb_kpis, ai_kpis)

else:
    st.markdown("""
    <div class="ready-box">
      <div class="ready-icon">⚡</div>
      <div class="ready-title">READY TO SIMULATE</div>
      <div class="ready-sub">Select a location and period in the sidebar, then click RUN SIMULATION</div>
    </div>
    """, unsafe_allow_html=True)

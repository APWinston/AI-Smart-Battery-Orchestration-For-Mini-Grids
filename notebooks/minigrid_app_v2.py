"""
AI Smart Battery Orchestration — Mini-Grid App v2
New features vs v1:
  • Live weather via Open-Meteo API (free, no key required)
  • Custom load profile builder (slider per hour OR CSV upload)
  • AI Decision Explainer — per-step reasoning panel
  • Live Mode: step through next 24 h hour-by-hour with real weather
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
from gymnasium import spaces
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Mini-Grid Orchestration",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Nunito:wght@600;700;800&display=swap');

/* ── Base ── */
.stApp, .main, .block-container {
  background: #f0f4f8 !important;
  font-family: 'Inter', sans-serif;
}
section[data-testid="stSidebar"] {
  background: #1e3a5f !important;
}
section[data-testid="stSidebar"] * { color: #e8f0fe !important; }
section[data-testid="stSidebar"] hr { border-color: #2d5a8e !important; }
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stSelectbox label {
  color: #93c5fd !important; font-size: 13px !important;
}

/* ── Button ── */
.stButton > button {
  background: linear-gradient(135deg, #1e3a5f, #2563eb) !important;
  border: none !important;
  color: #ffffff !important;
  font-family: 'Nunito', sans-serif !important;
  font-weight: 700 !important;
  font-size: 15px !important;
  border-radius: 10px !important;
  width: 100% !important;
  padding: 12px !important;
  box-shadow: 0 4px 14px rgba(37,99,235,0.35) !important;
  transition: all 0.2s !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
  box-shadow: 0 6px 20px rgba(37,99,235,0.45) !important;
  transform: translateY(-1px) !important;
}

/* ── App header ── */
.app-header {
  background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 60%, #0ea5e9 100%);
  border-radius: 16px;
  padding: 24px 32px;
  margin-bottom: 24px;
  box-shadow: 0 8px 32px rgba(37,99,235,0.2);
}
.app-title {
  font-family: 'Nunito', sans-serif;
  font-size: 28px;
  font-weight: 800;
  color: #ffffff !important;
  margin: 0;
  letter-spacing: 0.5px;
}
.app-sub {
  font-family: 'Inter', sans-serif;
  font-size: 13px;
  color: rgba(255,255,255,0.75) !important;
  margin: 6px 0 0;
}

/* ── Section label ── */
.sec-label {
  font-family: 'Nunito', sans-serif;
  font-size: 13px;
  font-weight: 700;
  color: #1e3a5f !important;
  text-transform: uppercase;
  letter-spacing: 1.5px;
  border-left: 4px solid #2563eb;
  padding-left: 12px;
  margin: 24px 0 14px;
  display: block;
}

/* ── KPI cards ── */
.kpi-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:8px; }
.kpi-card {
  background: #ffffff;
  border-radius: 12px;
  padding: 16px;
  border-top: 4px solid #2563eb;
  box-shadow: 0 2px 8px rgba(0,0,0,0.07);
}
.kpi-card.red  { border-top-color: #ef4444; }
.kpi-card.yel  { border-top-color: #f59e0b; }
.kpi-card.blue { border-top-color: #0ea5e9; }
.kpi-card.grn  { border-top-color: #10b981; }
.kpi-lbl {
  font-family: 'Inter', sans-serif;
  font-size: 11px;
  font-weight: 500;
  color: #64748b !important;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  margin-bottom: 6px;
}
.kpi-val {
  font-family: 'Nunito', sans-serif;
  font-size: 26px;
  font-weight: 800;
  color: #1e3a5f !important;
  line-height: 1;
}
.kpi-val.red  { color: #ef4444 !important; }
.kpi-val.yel  { color: #d97706 !important; }
.kpi-val.blue { color: #0284c7 !important; }
.kpi-val.grn  { color: #059669 !important; }
.kpi-unit { font-size: 13px; font-weight: 500; opacity: 0.6; }

/* ── Weather card ── */
.wx-card {
  background: #ffffff;
  border-radius: 14px;
  padding: 18px 22px;
  display: flex;
  gap: 20px;
  align-items: center;
  margin-bottom: 16px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.08);
  border-left: 5px solid #0ea5e9;
}
.wx-icon { font-size: 42px; }
.wx-vals { display:flex; flex-direction:column; gap:4px; }
.wx-loc {
  font-family: 'Nunito', sans-serif;
  font-size: 16px;
  font-weight: 700;
  color: #1e3a5f !important;
}
.wx-row {
  font-family: 'Inter', sans-serif;
  font-size: 13px;
  color: #475569 !important;
}
.wx-row span { color: #2563eb !important; font-weight: 600; }

/* ── Comparison table ── */
.cmp-wrap {
  background: #ffffff;
  border-radius: 14px;
  padding: 22px;
  margin-top: 20px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.07);
}
.cmp-title {
  font-family: 'Nunito', sans-serif;
  font-size: 17px;
  font-weight: 700;
  color: #1e3a5f !important;
  margin-bottom: 16px;
}
.cmp-table { width:100%; border-collapse:collapse; }
.cmp-table th {
  font-family: 'Inter', sans-serif;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.8px;
  text-transform: uppercase;
  padding: 10px 14px;
  border-bottom: 2px solid #e2e8f0;
  text-align: left;
  background: #f8fafc;
}
.cmp-table th.h-metric { color: #64748b !important; }
.cmp-table th.h-rb  { color: #ef4444 !important; }
.cmp-table th.h-ai  { color: #2563eb !important; }
.cmp-table th.h-imp { color: #d97706 !important; }
.cmp-table td {
  padding: 10px 14px;
  font-size: 13px;
  border-bottom: 1px solid #f1f5f9;
  color: #334155 !important;
}
.cmp-table tr:hover td { background: #f8fafc; }
.cmp-table td.metric { color: #475569 !important; font-size: 12px; font-weight: 500; }
.cmp-table td.rb-v  { color: #ef4444 !important; font-family:'Nunito',sans-serif; font-size:16px; font-weight:700; }
.cmp-table td.ai-v  { color: #2563eb !important; font-family:'Nunito',sans-serif; font-size:16px; font-weight:700; }
.badge {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 20px;
  font-family: 'Inter', sans-serif;
  font-size: 11px;
  font-weight: 600;
}
.badge.good { background: #dcfce7; color: #166534 !important; }
.badge.bad  { background: #fee2e2; color: #991b1b !important; }

.info-box {
  background: #eff6ff;
  border: 1px solid #bfdbfe;
  border-radius: 10px;
  padding: 12px 16px;
  margin-top: 10px;
}
.info-box p { font-size: 12px; color: #1e40af !important; margin: 3px 0; }

/* Fix Streamlit native metric widget text visibility */
[data-testid="metric-container"] {
  background: #ffffff;
  border-radius: 12px;
  padding: 16px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.07);
}
[data-testid="metric-container"] label {
  color: #64748b !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.8px !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  color: #1e3a5f !important;
  font-family: 'Nunito', sans-serif !important;
  font-size: 26px !important;
  font-weight: 800 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
  color: #059669 !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# MODEL CLASSES (identical to training)
# ══════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════
# LIVE WEATHER — Open-Meteo (free, no key needed)
# ══════════════════════════════════════════════════════════════

LOCATION_COORDS = {
    "Tamale":     (9.4035,  -0.8424,  "Northern Ghana"),
    "Kumasi":     (6.6885,  -1.6244,  "Ashanti Region"),
    "Axim":       (4.8699,  -2.2372,  "Western Region"),
    "Accra":      (5.5560,  -0.1969,  "Greater Accra"),
    "Bolgatanga": (10.7856, -0.8514,  "Upper East"),
    "Akosombo":   (6.2950,  -0.0581,  "Eastern Region"),
    "Custom":     (None,    None,     "Custom coordinates"),
}

@st.cache_data(ttl=900)  # cache 15 min
def fetch_live_weather(lat, lon, location_name):
    """
    Fetch 3-day hourly forecast from Open-Meteo (free, no API key).
    timezone=GMT == UTC == Ghana local time (Ghana is UTC+0 year-round).
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly=shortwave_radiation,temperature_2m,precipitation"
        f"&forecast_days=3&timezone=GMT"
    )
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()["hourly"]
        # Parse times as naive (no tz) — they are already GMT = Ghana local
        datetimes = pd.to_datetime(data["time"]).tz_localize(None)
        df = pd.DataFrame({
            "datetime": datetimes,
            "ssrd_wm2": pd.array(data["shortwave_radiation"], dtype=float),
            "temp_c":   pd.array(data["temperature_2m"],      dtype=float),
            "tp":       pd.array(data["precipitation"],        dtype=float),
        })
        df["location"]      = location_name
        df["location_code"] = 0
        df["hour"]          = df["datetime"].dt.hour
        df["month"]         = df["datetime"].dt.month
        df["dayofweek"]     = df["datetime"].dt.dayofweek
        return df.copy(), None   # return copy so callers can mutate safely
    except Exception as e:
        return None, str(e)


def weather_icon(ssrd, hour):
    if hour < 6 or hour > 20:
        return "🌙"
    if ssrd > 600:
        return "☀️"
    if ssrd > 300:
        return "⛅"
    return "☁️"


# ══════════════════════════════════════════════════════════════
# LOAD RESOURCES
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def load_resources():
    df = pd.read_csv("../data/master_dataset.csv")
    df["datetime"]      = pd.to_datetime(df["datetime"])
    df["location_code"] = df["location"].map({"Tamale":0,"Kumasi":1,"Axim":2})
    df["hour"]          = df["datetime"].dt.hour
    df["month"]         = df["datetime"].dt.month
    df["dayofweek"]     = df["datetime"].dt.dayofweek

    scaler_X = MinMaxScaler(); scaler_y = MinMaxScaler()
    scaler_X.fit(df[["ssrd_wm2","tp","temp_c","load_kw",
                      "location_code","hour","month","dayofweek"]])
    scaler_y.fit(df[["ssrd_wm2","load_kw"]])

    lstm = MiniGridLSTM(8, 128, 2, 24, 2)
    lstm.load_state_dict(torch.load("../models/best_lstm.pth", map_location="cpu"))
    lstm.eval()

    ppo = PPO.load("../models/best_model.zip")

    class _Env(gym.Env):
        def __init__(self):
            super().__init__()
            self.action_space = spaces.Box(-1.,1.,(1,),np.float32)
            self.observation_space = spaces.Box(-np.inf,np.inf,(52,),np.float32)
        def reset(self,**kw): return np.zeros(52,np.float32),{}
        def step(self,a): return np.zeros(52,np.float32),0.,True,False,{}

    vec_norm = VecNormalize.load(
        "../models/vecnormalize_stats.pkl",
        DummyVecEnv([_Env])
    )
    vec_norm.training = False; vec_norm.norm_reward = False
    return df, scaler_X, scaler_y, lstm, ppo, vec_norm


# ══════════════════════════════════════════════════════════════
# RULE-BASED CONTROLLER
# ══════════════════════════════════════════════════════════════

def rule_based_action(soc, solar_kw, load_kw, soc_min=0.20, soc_max=0.90):
    net = solar_kw - load_kw
    if net > 0:
        return float(min(net / 100.0, 1.0)) if soc < soc_max else 0.0
    else:
        return -1.0 if soc > soc_min else 0.0


# ══════════════════════════════════════════════════════════════
# DECISION EXPLAINER LOGIC
# ══════════════════════════════════════════════════════════════

def explain_decision(action, soc, solar_kw, load_kw, soh,
                     solar_forecast_next6, load_forecast_next6):
    """Return human-readable reasoning for an AI action."""
    net = solar_kw - load_kw
    reasons = []
    decision_class = "hold"

    if action > 0.05:
        decision_class = "charge"
        label = f"⬆ CHARGE  +{action:.2f}"
        kw = action * 400
        reasons.append(f"Charging at {kw:.0f} kW")
        if solar_kw > load_kw:
            reasons.append(f"Solar surplus: {net:.1f} kW available")
        if soc < 0.5:
            reasons.append(f"SOC is low ({soc*100:.0f}%) — building reserve")
        avg_solar_next = np.mean(solar_forecast_next6)
        if avg_solar_next < solar_kw * 0.5:
            reasons.append("Solar declining — storing now for night coverage")
        if soh < 0.92:
            reasons.append(f"SOH {soh*100:.1f}% — moderate charge rate to limit stress")

    elif action < -0.05:
        decision_class = "discharge"
        label = f"⬇ DISCHARGE  {action:.2f}"
        kw = abs(action) * 400
        reasons.append(f"Discharging at {kw:.0f} kW")
        if load_kw > solar_kw:
            reasons.append(f"Load deficit: {load_kw - solar_kw:.1f} kW not covered by solar")
        if soc > 0.6:
            reasons.append(f"SOC healthy ({soc*100:.0f}%) — safe to discharge")
        avg_load_next = np.mean(load_forecast_next6)
        if avg_load_next > load_kw:
            reasons.append("Rising load forecast — strategic pre-emptive discharge")
        if soc <= 0.25:
            reasons.append("⚠ SOC critically low — minimising discharge depth")

    else:
        label = "⏸ HOLD"
        reasons.append("Net energy balanced — no battery action needed")
        if abs(net) < 2:
            reasons.append(f"Solar ≈ Load ({solar_kw:.1f} ≈ {load_kw:.1f} kW)")
        if soc > 0.85:
            reasons.append("Battery near full — avoiding calendar aging")
        if 0.45 < soc < 0.65:
            reasons.append("SOC in optimal mid-range — holding to reduce wear")

    # ENS risk flag
    if action >= 0 and load_kw > solar_kw and soc < 0.3:
        decision_class = "blackout"
        reasons.append("⚠ BLACKOUT RISK: Load unserved this hour")

    return decision_class, label, reasons


# ══════════════════════════════════════════════════════════════
# PRECOMPUTE FORECASTS (batch LSTM)
# ══════════════════════════════════════════════════════════════

def precompute_forecasts(loc_df, lstm, scaler_X, scaler_y, start_idx, steps):
    feat_cols = ["ssrd_wm2","tp","temp_c","load_kw",
                 "location_code","hour","month","dayofweek"]
    X_all = scaler_X.transform(loc_df[feat_cols].values)
    end   = min(start_idx + steps + 24, len(loc_df) - 5)
    indices = list(range(start_idx, end))
    windows = np.stack([X_all[i-24:i] for i in indices])
    all_fc  = []
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


# ══════════════════════════════════════════════════════════════
# SIMULATION ENGINE
# ══════════════════════════════════════════════════════════════

def run_simulation(loc_df, lstm, scaler_X, scaler_y, ppo, vec_norm,
                   controller, years, progress_cb=None):
    steps     = min(years * 365 * 24, len(loc_df) - 50)
    start_idx = 24

    forecasts    = precompute_forecasts(loc_df, lstm, scaler_X, scaler_y, start_idx, steps)
    solar_arr    = (loc_df["ssrd_wm2"].values / 1000.0) * 750.0 * 0.75
    load_arr_all = loc_df["load_kw"].values
    hour_arr     = loc_df["hour"].values
    month_arr    = loc_df["month"].values

    soc_out  = np.zeros(steps); soh_out  = np.zeros(steps)
    ens_out  = np.zeros(steps); act_out  = np.zeros(steps)
    sol_out  = np.zeros(steps); load_out = np.zeros(steps)
    curt_out = np.zeros(steps)   # curtailed energy per hour (kWh)
    decisions = []

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
            action_arr, _ = ppo.predict(obs_norm, deterministic=True)
            action = float(np.clip(action_arr[0], -1.0, 1.0))
        else:
            action = rule_based_action(soc, solar_kw, load_kw)

        soc_before    = soc
        residual_load = max(0.0, load_kw - solar_kw)
        solar_surplus = max(0.0, solar_kw - load_kw)

        if action > 0:
            charge_kw     = min(action*max_rate, (soc_max-soc)*battery_capacity)
            actual_charge = min(charge_kw, solar_surplus)  # only charge from surplus
            soc          += (actual_charge*charge_eff) / battery_capacity
            net_load      = residual_load
            # Curtailment: surplus solar that couldn't be stored (battery full or rate limited)
            curtailed = max(0.0, solar_surplus - actual_charge)
        else:
            discharge_kw = min(abs(action)*max_rate, (soc-soc_min)*battery_capacity)
            soc         -= discharge_kw / battery_capacity
            net_load     = max(0.0, residual_load - discharge_kw*discharge_eff)
            # Curtailment: all surplus solar is curtailed when discharging
            curtailed    = solar_surplus

        soc         = float(np.clip(soc, soc_min, soc_max))
        ens_step    = min(max(0.0, net_load), load_kw)
        cycle_depth = abs(soc - soc_before)
        cal_stress  = max(0.0, soc - 0.85) * cal_deg
        soh         = max(0.0, soh - cycle_depth*soh_deg - cal_stress)

        soc_out[i]=soc;  soh_out[i]=soh;  ens_out[i]=ens_step
        act_out[i]=action; sol_out[i]=solar_kw; load_out[i]=load_kw
        curt_out[i]=curtailed

        # Store decision for explainer (every hour, first 7 days)
        if i < 24*7:
            dc, lbl, rsns = explain_decision(
                action, soc, solar_kw, load_kw, soh, sf[:6], lf[:6])
            decisions.append({
                "hour":  hour_arr[idx],
                "day":   i // 24 + 1,
                "step":  i,
                "class": dc,
                "label": lbl,
                "reasons": rsns,
                "soc":   soc,
                "soh":   soh,
                "solar": solar_kw,
                "load":  load_kw,
                "ens":   ens_step,
            })

        if progress_cb and i % 5000 == 0:
            progress_cb(i / steps)

    return pd.DataFrame({
        "soc": soc_out, "soh": soh_out, "ens": ens_out,
        "solar_kw": sol_out, "load_kw": load_out, "action": act_out,
        "curtailed_kw": curt_out,
    }), decisions


# ══════════════════════════════════════════════════════════════
# LIVE MODE — 24 h simulation on real weather + custom load
# ══════════════════════════════════════════════════════════════

def run_live_simulation(wx_df, load_profile_24h, lstm, scaler_X, scaler_y, ppo, vec_norm):
    """
    Step-by-step 24 h simulation on live weather data starting from NOW.
    Returns per-hour results + AI decisions with explanations.
    """

    import datetime as _dt

    # ── Step 1: Find current hour in the API data ──────────────────
    now_naive = pd.Timestamp(_dt.datetime.utcnow()).floor("h")

    # Build df from wx_df — inject load profile
    rows = []
    for i in range(len(wx_df)):
        row = wx_df.iloc[i]
        h = int(row["hour"])
        rows.append({
            "datetime":      pd.Timestamp(row["datetime"]),
            "ssrd_wm2":      float(row["ssrd_wm2"]),
            "tp":            float(row["tp"]),
            "temp_c":        float(row["temp_c"]),
            "load_kw":       float(load_profile_24h[h % 24]),
            "location_code": int(row["location_code"]),
            "hour":          h,
            "month":         int(row["month"]),
            "dayofweek":     int(row["dayofweek"]),
        })
    df_api = pd.DataFrame(rows)

    if len(df_api) < 25:
        return None, "Not enough weather data from API"

    # Find which row matches NOW
    diffs = (df_api["datetime"] - now_naive).abs()
    now_idx_api = int(diffs.idxmin())
    matched = df_api.iloc[now_idx_api]["datetime"]
    solar_now = float(df_api.iloc[now_idx_api]["ssrd_wm2"])
    print(f"[SIM] now={now_naive} matched={matched} api_idx={now_idx_api} solar={solar_now:.1f} W/m2")

    # ── Step 2: Prepend 24 padding rows so LSTM always has a lookback ──
    # Use the first API row repeated — values don't matter much since
    # the LSTM will see real data from row 24 onward
    pad_row = df_api.iloc[0].copy()
    pad_rows = []
    for p in range(24):
        pr = pad_row.copy()
        pr["datetime"] = matched - _dt.timedelta(hours=24-p)
        pr["hour"] = int(pr["datetime"].hour)
        pad_rows.append(pr)
    df_pad = pd.DataFrame(pad_rows)

    # Concatenate: 24 pad rows + full API data
    df_full = pd.concat([df_pad, df_api], ignore_index=True)

    # Now now_idx in df_full = 24 + now_idx_api
    start_idx = 24 + now_idx_api
    simulate_hours = min(24, len(df_full) - start_idx - 1)
    if simulate_hours < 1:
        return None, "Not enough future weather data — try again in a moment"

    print(f"[SIM] start_idx={start_idx}  simulate_hours={simulate_hours}  hour_at_start={df_full.iloc[start_idx]['hour']}")

    feat_cols = ["ssrd_wm2","tp","temp_c","load_kw",
                 "location_code","hour","month","dayofweek"]
    X_all = scaler_X.transform(df_full[feat_cols].values)

    results  = []
    decisions = []
    soc = 0.5; soh = 1.0
    battery_capacity = 2000.0; max_rate = 400.0
    charge_eff = 0.95; discharge_eff = 0.95
    soc_min = 0.20; soc_max = 0.90
    soh_deg = 0.00005; cal_deg = 0.000002

    for i in range(simulate_hours):
        idx = start_idx + i
        # LSTM forecast from lookback
        window = X_all[idx-24:idx]
        X_t    = torch.FloatTensor(window).unsqueeze(0)
        with torch.no_grad():
            fc = lstm(X_t).numpy()[0]
        fc_inv = scaler_y.inverse_transform(fc.reshape(-1,2))
        sf = np.clip(fc_inv[:,0], 0, None)
        lf = np.clip(fc_inv[:,1], 0, None)

        solar_kw = (df_full.iloc[idx]["ssrd_wm2"] / 1000.0) * 750.0 * 0.75
        load_kw  = float(df_full.iloc[idx]["load_kw"])
        hour     = int(df_full.iloc[idx]["hour"])
        month    = int(df_full.iloc[idx]["month"])

        obs = np.concatenate([[soc, soh], sf/1000.0, lf/50.0,
            [hour/23.0, month/12.0]]).astype(np.float32)
        obs_norm  = vec_norm.normalize_obs(obs)
        action_a, _ = ppo.predict(obs_norm, deterministic=True)
        ai_action = float(np.clip(action_a[0], -1.0, 1.0))
        rb_action = rule_based_action(soc, solar_kw, load_kw)

        # Step with AI action
        soc_before    = soc
        residual_load = max(0.0, load_kw - solar_kw)
        if ai_action > 0:
            charge_kw = min(ai_action*max_rate, (soc_max-soc)*battery_capacity)
            soc      += (charge_kw*charge_eff) / battery_capacity
            net_load  = residual_load
        else:
            discharge_kw = min(abs(ai_action)*max_rate, (soc-soc_min)*battery_capacity)
            soc         -= discharge_kw / battery_capacity
            net_load     = max(0.0, residual_load - discharge_kw*discharge_eff)
        soc         = float(np.clip(soc, soc_min, soc_max))
        ens_step    = min(max(0.0, net_load), load_kw)
        cycle_depth = abs(soc - soc_before)
        cal_stress  = max(0.0, soc - 0.85) * cal_deg
        soh         = max(0.0, soh - cycle_depth*soh_deg - cal_stress)

        dc, lbl, rsns = explain_decision(
            ai_action, soc, solar_kw, load_kw, soh, sf[:6], lf[:6])

        # Curtailment for live simulation
        solar_surplus_live = max(0.0, solar_kw - load_kw)
        if ai_action > 0:
            actual_chg_live = min(ai_action * 400.0, solar_surplus_live)
            curtailed_live  = max(0.0, solar_surplus_live - actual_chg_live)
        else:
            curtailed_live  = solar_surplus_live

        results.append({
            "hour": hour, "solar_kw": solar_kw, "load_kw": load_kw,
            "soc": soc, "soh": soh, "ens": ens_step,
            "ai_action": ai_action, "rb_action": rb_action,
            "curtailed_kw": curtailed_live,
            "sf_solar": sf, "sf_load": lf,
        })
        row_dt = df_full.iloc[idx]["datetime"]
        decisions.append({
            "hour":     hour,
            "datetime": pd.Timestamp(row_dt).strftime("%H:%M") if row_dt else f"H{hour:02d}",
            "step":  i+1, "class": dc, "label": lbl,
            "reasons": rsns, "soc": soc, "soh": soh,
            "solar": solar_kw, "load": load_kw, "ens": ens_step,
        })

    return pd.DataFrame(results), decisions


# ══════════════════════════════════════════════════════════════
# COMPUTE KPIs
# ══════════════════════════════════════════════════════════════

# Ghana grid export tariff — ECG buys surplus solar at ~GHS 0.45/kWh (approx 2025 rate)
# Adjust this value to reflect current ECG feed-in tariff
EXPORT_TARIFF_GHS = 0.45   # GHS per kWh exported
EXPORT_EFFICIENCY = 0.95   # inverter export efficiency

def compute_kpis(res_df):
    ens  = res_df["ens"].values
    soh  = res_df["soh"].values
    soc  = res_df["soc"].values
    act  = res_df["action"].values
    sol  = res_df["solar_kw"].values
    load = res_df["load_kw"].values
    curt = res_df["curtailed_kw"].values if "curtailed_kw" in res_df.columns else np.zeros(len(ens))

    total_load      = np.sum(load)
    total_ens       = np.sum(ens)
    solar_used      = np.sum(np.minimum(sol, load))
    scr             = round(float(min(solar_used / max(np.sum(sol),1)*100, 100)), 1)
    served_pct      = round(float(np.clip((1-total_ens/max(total_load,1))*100,0,100)), 1)
    deg_total       = 1.0 - float(soh[-1])
    n_years         = len(soh) / 8760
    deg_per_yr      = deg_total / max(n_years, 0.001)
    lifespan        = round(0.20 / deg_per_yr, 1) if deg_per_yr > 0 else 99.0

    # Curtailment and export revenue
    total_curtailed = round(float(np.sum(curt)), 1)
    export_kwh      = round(total_curtailed * EXPORT_EFFICIENCY, 1)
    export_revenue  = round(export_kwh * EXPORT_TARIFF_GHS, 2)
    curtail_pct     = round(float(total_curtailed / max(np.sum(sol),1) * 100), 1)

    return {
        "ENS":            round(float(np.sum(ens)), 1),
        "LOLP":           round(float(np.mean(ens > 0) * 100), 1),
        "SOH":            round(float(soh[-1] * 100), 1),
        "EFC":            round(float(np.sum(np.abs(act)) * 0.5), 0),
        "SCR":            scr,
        "SOC_STD":        round(float(np.std(soc)), 4),
        "SERVED_PCT":     served_pct,
        "LIFESPAN":       lifespan,
        "CURTAILED_KWH":  total_curtailed,
        "CURTAIL_PCT":    curtail_pct,
        "EXPORT_KWH":     export_kwh,
        "EXPORT_REVENUE": export_revenue,
    }


# ══════════════════════════════════════════════════════════════
# RENDER HELPERS
# ══════════════════════════════════════════════════════════════

def dark_layout(title, height=240):
    return dict(
        title=dict(text=title, font=dict(color="#1e3a5f", family="Nunito", size=13), x=0),
        paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
        font=dict(color="#475569", family="Inter", size=10),
        margin=dict(l=40, r=10, t=36, b=30),
        xaxis=dict(gridcolor="#e2e8f0", showgrid=True, linecolor="#cbd5e1", color="#64748b"),
        yaxis=dict(gridcolor="#e2e8f0", showgrid=True, linecolor="#cbd5e1", color="#64748b"),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#e2e8f0",
                    borderwidth=1, font=dict(color="#475569", size=10)),
        height=height,
    )


def render_kpis(kpis):
    served = kpis.get("SERVED_PCT","N/A")
    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card red">
        <div class="kpi-lbl">Energy Not Served</div>
        <div class="kpi-val red">{kpis["ENS"]:,.0f}<span class="kpi-unit"> kWh</span></div>
      </div>
      <div class="kpi-card red">
        <div class="kpi-lbl">Loss of Load Prob</div>
        <div class="kpi-val red">{kpis["LOLP"]}<span class="kpi-unit"> %</span></div>
      </div>
      <div class="kpi-card">
        <div class="kpi-lbl">State of Health</div>
        <div class="kpi-val">{kpis["SOH"]}<span class="kpi-unit"> %</span></div>
      </div>
      <div class="kpi-card blue">
        <div class="kpi-lbl">Projected Lifespan</div>
        <div class="kpi-val blue">{kpis["LIFESPAN"]}<span class="kpi-unit"> yrs</span></div>
      </div>
      <div class="kpi-card yel">
        <div class="kpi-lbl">Equiv. Full Cycles</div>
        <div class="kpi-val yel">{int(kpis["EFC"]):,}</div>
      </div>
      <div class="kpi-card blue">
        <div class="kpi-lbl">Solar Self-Consumption</div>
        <div class="kpi-val blue">{kpis["SCR"]}<span class="kpi-unit"> %</span></div>
      </div>
      <div class="kpi-card">
        <div class="kpi-lbl">Load Served</div>
        <div class="kpi-val">{served}<span class="kpi-unit"> %</span></div>
      </div>
      <div class="kpi-card">
        <div class="kpi-lbl">SOC Std Dev</div>
        <div class="kpi-val">{kpis["SOC_STD"]:.4f}</div>
      </div>
      <div class="kpi-card yel">
        <div class="kpi-lbl">Curtailed Energy</div>
        <div class="kpi-val yel">{kpis.get("CURTAILED_KWH",0):,.0f}<span class="kpi-unit"> kWh</span></div>
      </div>
      <div class="kpi-card grn">
        <div class="kpi-lbl">Potential Export Revenue</div>
        <div class="kpi-val grn">GHS {kpis.get("EXPORT_REVENUE",0):,.0f}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_comparison_table(rb_kpis, ai_kpis):
    def badge(val, good):
        cls = "good" if good else "bad"
        return f'<span class="badge {cls}">{val}</span>'
    metrics = [
        ("Energy Not Served (kWh)",    "ENS",            False, "{:,.0f}"),
        ("Loss of Load Prob (%)",      "LOLP",           False, "{:.1f}%"),
        ("State of Health (%)",        "SOH",            True,  "{:.1f}%"),
        ("Projected Lifespan (yr)",    "LIFESPAN",       True,  "{:.1f} yr"),
        ("Equiv. Full Cycles",         "EFC",            False, "{:,.0f}"),
        ("Solar Self-Consumption (%)", "SCR",            True,  "{:.1f}%"),
        ("Load Served (%)",            "SERVED_PCT",     True,  "{:.1f}%"),
        ("SOC Std Deviation",          "SOC_STD",        False, "{:.4f}"),
        ("Curtailed Energy (kWh)",     "CURTAILED_KWH",  False, "{:,.0f}"),
        ("Curtailment Rate (%)",       "CURTAIL_PCT",    False, "{:.1f}%"),
        ("Exportable Energy (kWh)",    "EXPORT_KWH",     True,  "{:,.0f}"),
        ("Potential Revenue (GHS)",    "EXPORT_REVENUE", True,  "GHS {:,.0f}"),
    ]
    rows = []
    for label, key, hib, fmt in metrics:
        rb_v = rb_kpis.get(key,0); ai_v = ai_kpis.get(key,0)
        if hib:
            pct  = (ai_v - rb_v) / max(abs(rb_v),0.001) * 100
            good = pct > 0
            imp  = f"▲ {abs(pct):.1f}%" if good else f"▼ {abs(pct):.1f}%"
        else:
            pct  = (rb_v - ai_v) / max(abs(rb_v),0.001) * 100
            good = pct > 0
            imp  = f"▼ {abs(pct):.1f}%" if good else f"▲ {abs(pct):.1f}%"
        rows.append(f'<tr>'
                    f'<td class="metric">{label}</td>'
                    f'<td class="rb-v">{fmt.format(rb_v)}</td>'
                    f'<td class="ai-v">{fmt.format(ai_v)}</td>'
                    f'<td>{badge(imp,good)}</td>'
                    f'</tr>')
    st.markdown(f"""
    <div class="cmp-wrap">
      <div class="cmp-title">⚡ FINAL KPI COMPARISON</div>
      <table class="cmp-table">
        <thead><tr>
          <th class="h-metric">Metric</th>
          <th class="h-rb">⚙ Rule-Based</th>
          <th class="h-ai">🤖 AI (LSTM+PPO)</th>
          <th class="h-imp">AI Improvement</th>
        </tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)


def render_decision_panel(decisions, title="🧠 AI DECISION LOG — FIRST 7 DAYS"):
    import streamlit.components.v1 as components

    st.markdown(f'<span class="sec-label">{title}</span>', unsafe_allow_html=True)

    shown = decisions[:min(48, len(decisions))]

    # Border colours per decision class
    border_map = {
        "charge":    "#10b981",
        "discharge": "#f97316",
        "hold":      "#f59e0b",
        "blackout":  "#ef4444",
    }
    action_color_map = {
        "charge":    "#059669",
        "discharge": "#ea580c",
        "hold":      "#d97706",
        "blackout":  "#dc2626",
    }

    rows_html = ""
    for d in shown:
        dc       = d["class"]
        label    = d["label"]
        rsns     = d["reasons"]
        soc_pct  = d["soc"] * 100
        soh_pct  = d["soh"] * 100
        sol_pct  = min(d["solar"] / 150 * 100, 100)
        soc_col  = "#00cc66" if soc_pct > 50 else "#ffc107" if soc_pct > 30 else "#ff4444"
        ens_flag = " &#9889; ENS!" if d["ens"] > 0 else ""
        border   = border_map.get(dc, "#1a4a1a")
        acolor   = action_color_map.get(dc, "#00ff88")
        reasons_html = "".join(f"<div>&rarr; {r}</div>" for r in rsns)

        rows_html += f"""
        <div style="display:flex;gap:12px;align-items:flex-start;margin-bottom:10px;
                    padding:12px 14px;background:#f8fafc;border-radius:10px;
                    border-left:4px solid {border};">
          <div style="font-family:'Inter',sans-serif;font-size:11px;font-weight:600;
                      color:#64748b;min-width:44px;padding-top:2px;text-align:center;">
            <div style="font-size:14px;color:#1e3a5f;font-weight:800;">{d.get('datetime', f"H{{d['hour']:02d}}")}</div>
          </div>
          <div style="flex:1;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
              <span style="font-family:'Nunito',sans-serif;font-size:14px;
                           font-weight:800;color:{acolor};">{label}{ens_flag}</span>
              <span style="font-family:'Inter',sans-serif;font-size:11px;color:#64748b;">
                &#9728; {d['solar']:.1f} kW &nbsp;|&nbsp; &#127968; {d['load']:.1f} kW
              </span>
            </div>
            <div style="font-family:'Inter',sans-serif;font-size:12px;
                        color:#475569;line-height:1.7;">{reasons_html}</div>
            <div style="display:flex;gap:14px;align-items:center;margin-top:8px;">
              <div style="display:flex;flex-direction:column;gap:2px;">
                <div style="width:64px;height:7px;background:#e2e8f0;border-radius:4px;overflow:hidden;">
                  <div style="width:{soc_pct:.0f}%;height:100%;background:{soc_col};border-radius:4px;"></div>
                </div>
                <div style="font-size:10px;color:#64748b;font-family:'Inter',sans-serif;">
                  SOC {soc_pct:.0f}%
                </div>
              </div>
              <div style="display:flex;flex-direction:column;gap:2px;">
                <div style="width:64px;height:7px;background:#e2e8f0;border-radius:4px;overflow:hidden;">
                  <div style="width:{soh_pct:.0f}%;height:100%;background:#0ea5e9;border-radius:4px;"></div>
                </div>
                <div style="font-size:10px;color:#64748b;font-family:'Inter',sans-serif;">
                  SOH {soh_pct:.1f}%
                </div>
              </div>
              <div style="display:flex;flex-direction:column;gap:2px;">
                <div style="width:64px;height:7px;background:#e2e8f0;border-radius:4px;overflow:hidden;">
                  <div style="width:{sol_pct:.0f}%;height:100%;background:#f59e0b;border-radius:4px;"></div>
                </div>
                <div style="font-size:10px;color:#64748b;font-family:'Inter',sans-serif;">
                  Solar {d['solar']:.0f} W
                </div>
              </div>
            </div>
          </div>
        </div>
        """

    full_html = f"""
    <!DOCTYPE html><html><head>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Nunito:wght@700;800&display=swap" rel="stylesheet">
    <style>
      * {{ margin:0; padding:0; box-sizing:border-box; }}
      body {{ background:#f0f4f8; font-family:'Inter',sans-serif; padding:12px; }}
    </style>
    </head><body>
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;padding:16px;box-shadow:0 2px 12px rgba(0,0,0,0.07);">
      <div style="font-family:'Nunito',sans-serif;font-size:16px;font-weight:800;
                  color:#1e3a5f;margin-bottom:14px;">{title}</div>
      {rows_html}
    </div>
    </body></html>
    """

    # Calculate height based on number of decisions shown
    panel_height = max(300, len(shown) * 110)
    components.html(full_html, height=panel_height, scrolling=True)


# ══════════════════════════════════════════════════════════════
# DEFAULT LOAD PROFILES (kW per hour)
# ══════════════════════════════════════════════════════════════

DEFAULT_LOAD_PROFILES = {
    "Residential (typical Ghana)": [
        8, 7, 7, 7, 8, 10, 14, 18, 20, 18, 16, 16,
        17, 16, 15, 16, 18, 24, 28, 30, 28, 22, 16, 10
    ],
    "Mixed Commercial": [
        10, 9, 8, 8, 9, 12, 18, 25, 30, 32, 34, 33,
        32, 33, 32, 30, 28, 26, 24, 22, 20, 18, 15, 12
    ],
    "Agricultural / Irrigation": [
        5, 5, 5, 5, 5, 8, 15, 28, 35, 38, 38, 36,
        34, 36, 38, 36, 30, 20, 12, 8, 7, 6, 5, 5
    ],
    "Flat (constant)": [20] * 24,
}


# ══════════════════════════════════════════════════════════════
# APP LAYOUT
# ══════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="app-header">
  <p class="app-title">⚡ AI SMART BATTERY ORCHESTRATION</p>
  <p class="app-sub">Live Weather  ·  Custom Load Profiles  ·  AI Decision Explainer  ·  LSTM + PPO Agent</p>
</div>
""", unsafe_allow_html=True)

# Load models
with st.spinner("Loading models..."):
    try:
        df_train, scaler_X, scaler_y, lstm, ppo, vec_norm = load_resources()
        models_ok = True
    except Exception as e:
        st.error(f"Could not load models: {e}")
        models_ok = False
        st.stop()

# ── SIDEBAR ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙ SETTINGS")
    st.markdown("---")

    app_mode = st.radio(
        "Mode",
        ["📡 Live Weather (24 h)", "📊 Historical Simulation"],
        index=0,
    )
    st.markdown("---")

    location = st.selectbox(
        "📍 Location",
        list(LOCATION_COORDS.keys()),
        index=0,
    )

    if location == "Custom":
        custom_lat = st.number_input("Latitude",  value=7.9465,  format="%.4f")
        custom_lon = st.number_input("Longitude", value=-1.0232, format="%.4f")
        LOCATION_COORDS["Custom"] = (custom_lat, custom_lon, "Custom")

    if app_mode == "📊 Historical Simulation":
        years = st.selectbox("📅 Period", [1,2,3,4,5,6],
                             format_func=lambda x: f"{x} Year{'s' if x>1 else ''}")
    st.markdown("---")

    # Load profile builder
    st.markdown("### 🔌 LOAD PROFILE")
    load_source = st.radio("Load data source",
                           ["Preset profile", "Custom (sliders)", "Upload CSV"])

    if load_source == "Preset profile":
        preset = st.selectbox("Profile", list(DEFAULT_LOAD_PROFILES.keys()))
        load_profile = DEFAULT_LOAD_PROFILES[preset]

    elif load_source == "Custom (sliders)":
        st.caption("Set hourly demand (kW)")
        load_profile = []
        cols = st.columns(4)
        for h in range(24):
            with cols[h % 4]:
                val = st.slider(f"H{h:02d}", 1, 60,
                               DEFAULT_LOAD_PROFILES["Residential (typical Ghana)"][h],
                               key=f"load_h{h}", label_visibility="collapsed")
            load_profile.append(val)

    else:
        uploaded = st.file_uploader("Upload CSV (column: load_kw, 24 rows)",
                                    type=["csv"])
        if uploaded:
            ldf = pd.read_csv(uploaded)
            load_profile = ldf["load_kw"].values[:24].tolist()
            st.success(f"Loaded: mean {np.mean(load_profile):.1f} kW")
        else:
            load_profile = DEFAULT_LOAD_PROFILES["Residential (typical Ghana)"]
            st.info("Using default residential profile")

    st.markdown("---")
    run_btn = st.button("▶  RUN" if app_mode == "📊 Historical Simulation" else "⚡ FETCH & SIMULATE")


# ══════════════════════════════════════════════════════════════
# LIVE WEATHER MODE
# ══════════════════════════════════════════════════════════════

if app_mode == "📡 Live Weather (24 h)":
    lat, lon, region = LOCATION_COORDS[location]

    # Always show weather card (auto-fetch)
    st.markdown('<span class="sec-label">LIVE WEATHER CONDITIONS</span>',
                unsafe_allow_html=True)

    # Initialise wx_df so it is always defined in this scope
    wx_df = None

    if lat is not None:
        with st.spinner(f"Fetching live weather for {location}..."):
            wx_df, wx_err = fetch_live_weather(lat, lon, location)

        if wx_err or wx_df is None:
            st.error(f"Weather API error for {location}: {wx_err}. Try again in a moment.")
            wx_df = None
        else:
            # ── Find current hour index (Ghana = UTC+0 = GMT) ──────────
            import datetime as _dt
            now_naive = pd.Timestamp(_dt.datetime.utcnow()).floor("h")  # truncate to hour
            # wx_df["datetime"] is already naive UTC (tz_localize(None) in fetch)
            time_diffs = (wx_df["datetime"] - now_naive).abs()
            now_i = int(time_diffs.idxmin())
            now_row = wx_df.iloc[now_i]

            icon = weather_icon(float(now_row["ssrd_wm2"]), int(now_row["hour"]))
            solar_kw_now = (float(now_row["ssrd_wm2"]) / 1000.0) * 750.0 * 0.75
            now_label = pd.Timestamp(now_row["datetime"]).strftime("%Y-%m-%d %H:%M")

            st.markdown(f"""
            <div class="wx-card">
              <div class="wx-icon">{icon}</div>
              <div class="wx-vals">
                <div class="wx-loc">{location} &#8212; {region}</div>
                <div class="wx-row">&#128336; Now (Ghana / UTC): <span>{now_label}</span></div>
                <div class="wx-row">&#127777; Temperature: <span>{float(now_row["temp_c"]):.1f} &deg;C</span></div>
                <div class="wx-row">&#9728; Solar irradiance: <span>{float(now_row["ssrd_wm2"]):.0f} W/m&sup2;</span>
                    &nbsp;&rarr; PV output: <span>{solar_kw_now:.1f} kW</span></div>
                <div class="wx-row">&#127783; Precipitation: <span>{float(now_row["tp"]):.1f} mm</span></div>
                <div class="wx-row">&#128197; Forecast range: <span>{wx_df["datetime"].iloc[0].strftime("%Y-%m-%d %H:%M")}</span>
                    to <span>{wx_df["datetime"].iloc[-1].strftime("%Y-%m-%d %H:%M")}</span></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── 24h forecast charts from NOW ───────────────────────────
            st.markdown('<span class="sec-label">24-HOUR SOLAR FORECAST (LSTM)</span>',
                        unsafe_allow_html=True)

            wx_24 = wx_df.iloc[now_i : now_i + 24].copy().reset_index(drop=True)
            dt_labels = wx_24["datetime"].dt.strftime("%H:%M").tolist()
            solar_24  = (wx_24["ssrd_wm2"].values / 1000.0) * 750.0 * 0.75
            load_24   = [load_profile[int(h) % 24] for h in wx_24["hour"].values]

            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=dt_labels, y=solar_24,
                                     marker_color="#f59e0b", name="Solar PV (kW)", opacity=0.85))
                fig.add_trace(go.Scatter(x=dt_labels, y=load_24,
                                         line=dict(color="#ef4444", width=2), name="Load (kW)"))
                fig.update_layout(**dark_layout(f"Solar PV vs Load — {now_label} to next 24h", height=260))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=dt_labels, y=wx_24["temp_c"].values,
                                          line=dict(color="#0ea5e9", width=2), fill="tozeroy",
                                          fillcolor="rgba(14,165,233,0.08)", name="Temp (C)"))
                fig2.update_layout(**dark_layout(f"Temperature — {now_label} to next 24h", height=260))
                st.plotly_chart(fig2, use_container_width=True)

    if run_btn and wx_df is not None:
        with st.spinner("Running AI simulation on live weather..."):
            live_res, live_decisions = run_live_simulation(
                wx_df, load_profile, lstm, scaler_X, scaler_y, ppo, vec_norm)

        if live_res is None:
            st.error(f"Simulation error: {live_decisions}")
        else:
            import datetime as _dt
            _now = _dt.datetime.utcnow()
            _end = _now + _dt.timedelta(hours=24)
            time_label = _now.strftime("%H:%M")
            end_label  = _end.strftime("%Y-%m-%d %H:%M")
            st.markdown(f'<span class="sec-label">SIMULATION RESULTS &nbsp;·&nbsp; {time_label} TODAY &rarr; {end_label}</span>',
                        unsafe_allow_html=True)

            hours = live_res["hour"].values

            # Summary metrics
            total_ens  = live_res["ens"].sum()
            final_soc  = live_res["soc"].iloc[-1] * 100
            final_soh  = live_res["soh"].iloc[-1] * 100
            lolp_24    = (live_res["ens"] > 0).mean() * 100
            ens_color  = "#ef4444" if total_ens > 0 else "#059669"
            lolp_color = "#ef4444" if lolp_24 > 5 else "#d97706" if lolp_24 > 0 else "#059669"
            soc_color  = "#059669" if final_soc > 50 else "#d97706" if final_soc > 30 else "#ef4444"
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:20px;">
              <div style="background:#fff;border-radius:12px;padding:18px;box-shadow:0 2px 8px rgba(0,0,0,0.07);border-top:4px solid {ens_color};">
                <div style="font-size:11px;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px;">Total ENS</div>
                <div style="font-family:Nunito,sans-serif;font-size:28px;font-weight:800;color:{ens_color};line-height:1;">{total_ens:.1f}<span style="font-size:14px;font-weight:500;color:#94a3b8;"> kWh</span></div>
                <div style="font-size:11px;color:#94a3b8;margin-top:4px;">Energy Not Served</div>
              </div>
              <div style="background:#fff;border-radius:12px;padding:18px;box-shadow:0 2px 8px rgba(0,0,0,0.07);border-top:4px solid {lolp_color};">
                <div style="font-size:11px;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px;">LOLP</div>
                <div style="font-family:Nunito,sans-serif;font-size:28px;font-weight:800;color:{lolp_color};line-height:1;">{lolp_24:.1f}<span style="font-size:14px;font-weight:500;color:#94a3b8;"> %</span></div>
                <div style="font-size:11px;color:#94a3b8;margin-top:4px;">Loss of Load Probability</div>
              </div>
              <div style="background:#fff;border-radius:12px;padding:18px;box-shadow:0 2px 8px rgba(0,0,0,0.07);border-top:4px solid {soc_color};">
                <div style="font-size:11px;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px;">Final SOC</div>
                <div style="font-family:Nunito,sans-serif;font-size:28px;font-weight:800;color:{soc_color};line-height:1;">{final_soc:.1f}<span style="font-size:14px;font-weight:500;color:#94a3b8;"> %</span></div>
                <div style="font-size:11px;color:#94a3b8;margin-top:4px;">Battery State of Charge</div>
              </div>
              <div style="background:#fff;border-radius:12px;padding:18px;box-shadow:0 2px 8px rgba(0,0,0,0.07);border-top:4px solid #0ea5e9;">
                <div style="font-size:11px;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px;">Final SOH</div>
                <div style="font-family:Nunito,sans-serif;font-size:28px;font-weight:800;color:#0284c7;line-height:1;">{final_soh:.2f}<span style="font-size:14px;font-weight:500;color:#94a3b8;"> %</span></div>
                <div style="font-size:11px;color:#94a3b8;margin-top:4px;">Battery State of Health</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # SOC + action side by side
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(range(len(live_res))),
                                          y=live_res["soc"].values*100,
                                          line=dict(color="#00cc66",width=2),
                                          fill="tozeroy",
                                          fillcolor="rgba(0,204,102,0.07)",
                                          name="SOC %"))
                fig.add_hline(y=20, line_dash="dash", line_color="#ff4444", line_width=1)
                fig.add_hline(y=90, line_dash="dash", line_color="#333", line_width=1)
                fig.update_layout(**dark_layout("Battery SOC — 24h (AI Agent)", 240))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                colors = ["#00cc66" if a > 0.05 else "#ff6b35" if a < -0.05 else "#ffc107"
                          for a in live_res["ai_action"].values]
                fig = go.Figure()
                fig.add_trace(go.Bar(x=list(range(len(live_res))),
                                      y=live_res["ai_action"].values,
                                      marker_color=colors, name="AI Action"))
                fig.add_trace(go.Scatter(x=list(range(len(live_res))),
                                          y=live_res["rb_action"].values,
                                          line=dict(color="#ff6666",width=1.5,dash="dash"),
                                          name="Rule-Based Action"))
                fig.add_hline(y=0, line_color="#333", line_width=1)
                fig.update_layout(**dark_layout("AI vs Rule-Based Actions (+charge / −discharge)", 240))
                st.plotly_chart(fig, use_container_width=True)

            # Solar / Load / ENS / Curtailment
            st.markdown('<span class="sec-label">ENERGY FLOWS</span>', unsafe_allow_html=True)
            dt_x = list(range(len(live_res)))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dt_x, y=live_res["solar_kw"].values,
                                      line=dict(color="#f59e0b",width=2), name="Solar PV (kW)"))
            fig.add_trace(go.Scatter(x=dt_x, y=live_res["load_kw"].values,
                                      line=dict(color="#10b981",width=2), name="Load (kW)"))
            fig.add_trace(go.Bar(x=dt_x, y=live_res["ens"].values,
                                  marker_color="rgba(239,68,68,0.6)", name="ENS (kWh)"))
            fig.add_trace(go.Bar(x=dt_x, y=live_res["curtailed_kw"].values,
                                  marker_color="rgba(245,158,11,0.5)", name="Curtailed (kWh)"))
            fig.update_layout(**dark_layout("Solar / Load / ENS / Curtailed — 24h", 280))
            st.plotly_chart(fig, use_container_width=True)

            # Live export revenue summary
            live_curtailed = live_res["curtailed_kw"].sum()
            live_export    = live_curtailed * 0.95
            live_revenue   = live_export * EXPORT_TARIFF_GHS
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px;">
              <div style="background:#fff;border-radius:12px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,0.07);border-top:4px solid #f59e0b;">
                <div style="font-size:11px;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:6px;">Curtailed Energy</div>
                <div style="font-family:Nunito,sans-serif;font-size:24px;font-weight:800;color:#d97706;">{live_curtailed:.1f}<span style="font-size:13px;font-weight:500;color:#94a3b8;"> kWh</span></div>
                <div style="font-size:11px;color:#94a3b8;margin-top:3px;">Solar surplus not stored</div>
              </div>
              <div style="background:#fff;border-radius:12px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,0.07);border-top:4px solid #0ea5e9;">
                <div style="font-size:11px;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:6px;">Exportable Energy</div>
                <div style="font-family:Nunito,sans-serif;font-size:24px;font-weight:800;color:#0284c7;">{live_export:.1f}<span style="font-size:13px;font-weight:500;color:#94a3b8;"> kWh</span></div>
                <div style="font-size:11px;color:#94a3b8;margin-top:3px;">At 95% inverter efficiency</div>
              </div>
              <div style="background:#fff;border-radius:12px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,0.07);border-top:4px solid #10b981;">
                <div style="font-size:11px;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:6px;">Potential Revenue</div>
                <div style="font-family:Nunito,sans-serif;font-size:24px;font-weight:800;color:#059669;">GHS {live_revenue:.2f}</div>
                <div style="font-size:11px;color:#94a3b8;margin-top:3px;">@ GHS {EXPORT_TARIFF_GHS}/kWh (ECG tariff)</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Decision explainer
            from datetime import datetime, timezone, timedelta
            now_str  = datetime.now(timezone.utc).strftime("%H:%M")
            end_str  = (datetime.now(timezone.utc) + timedelta(hours=24)).strftime("%H:%M, %d %b")
            render_decision_panel(live_decisions, f"AI Decision Log  —  {now_str} today to {end_str}")

            # Download button
            csv = live_res.drop(columns=["sf_solar","sf_load"], errors="ignore").to_csv(index=False)
            st.download_button("⬇ Download 24h results CSV", csv,
                               "live_24h_results.csv", "text/csv")

    elif run_btn and wx_df is None:
        st.error("Cannot simulate — weather data unavailable.")


# ══════════════════════════════════════════════════════════════
# HISTORICAL SIMULATION MODE
# ══════════════════════════════════════════════════════════════

else:
    # Build location df with custom load
    loc_key = location if location in df_train["location"].unique() else "Tamale"
    loc_df  = df_train[df_train["location"] == loc_key].reset_index(drop=True).copy()

    # Inject custom load profile
    loc_df["load_kw"] = loc_df["hour"].apply(lambda h: load_profile[h])

    with st.sidebar:
        st.markdown("---")
        info_map = {
            "Tamale":     ("Northern",  "Hot & Semi-Arid", "9.40°N",  "0.85°W"),
            "Kumasi":     ("Ashanti",   "Humid Tropical",  "6.67°N",  "1.62°W"),
            "Axim":       ("Western",   "Coastal Wet",     "4.87°N",  "2.24°W"),
            "Accra":      ("Gr. Accra", "Coastal",         "5.56°N",  "0.20°W"),
            "Bolgatanga": ("Upper E.",  "Hot Semi-Arid",   "10.79°N", "0.85°W"),
            "Akosombo":   ("Eastern",   "Sub-humid",       "6.30°N",  "0.06°E"),
        }
        reg, clim, la, lo = info_map.get(loc_key, ("—","—","—","—"))
        avg_load = np.mean(load_profile)
        st.markdown(f"""
        **📍 {loc_key}**  
        Region: {reg} | Climate: {clim}  
        Avg load: **{avg_load:.1f} kW**  
        Battery: 2000 kWh LiFePO₄  
        Steps: {years*365*24:,} hours
        """)

    if run_btn:
        prog_bar = st.progress(0, text="Running Rule-Based...")
        rb_res, rb_dec = run_simulation(
            loc_df, lstm, scaler_X, scaler_y, ppo, vec_norm,
            "RB", years, lambda p: prog_bar.progress(p*0.5, text="Rule-Based..."))
        prog_bar.progress(0.5, text="Running AI Agent...")
        ai_res, ai_dec = run_simulation(
            loc_df, lstm, scaler_X, scaler_y, ppo, vec_norm,
            "AI", years, lambda p: prog_bar.progress(0.5+p*0.5, text="AI Agent..."))
        prog_bar.progress(1.0, text="Done!"); prog_bar.empty()

        rb_kpis = compute_kpis(rb_res)
        ai_kpis = compute_kpis(ai_res)
        st.session_state.update({
            "rb_res": rb_res, "ai_res": ai_res,
            "rb_kpis": rb_kpis, "ai_kpis": ai_kpis,
            "rb_dec": rb_dec, "ai_dec": ai_dec,
            "sim_loc": loc_key, "sim_years": years,
        })

    if "rb_res" in st.session_state:
        rb_res   = st.session_state["rb_res"]
        ai_res   = st.session_state["ai_res"]
        rb_kpis  = st.session_state["rb_kpis"]
        ai_kpis  = st.session_state["ai_kpis"]
        ai_dec   = st.session_state.get("ai_dec", [])
        sim_loc  = st.session_state["sim_loc"]
        sim_yrs  = st.session_state["sim_years"]

        st.markdown(f'<span class="sec-label">RESULTS — {sim_loc} · {sim_yrs} Year{"s" if sim_yrs>1 else ""}</span>',
                    unsafe_allow_html=True)

        col_rb, col_vs, col_ai = st.columns([10,1,10])
        with col_rb:
            st.markdown('<span style="font-family:Nunito,sans-serif;font-size:13px;font-weight:700;color:#ef4444;letter-spacing:1px;">⚙ Rule-Based Controller</span>',
                        unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            render_kpis(rb_kpis)
        with col_vs:
            st.markdown('<div style="text-align:center;font-family:Rajdhani,sans-serif;font-size:22px;font-weight:700;color:#94a3b8;padding-top:40px;">VS</div>',
                        unsafe_allow_html=True)
        with col_ai:
            st.markdown('<span style="font-family:Nunito,sans-serif;font-size:13px;font-weight:700;color:#2563eb;letter-spacing:1px;">🤖 AI Agent · LSTM + PPO</span>',
                        unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            render_kpis(ai_kpis)

        # Charts
        st.markdown('<span class="sec-label">BATTERY STATE OF CHARGE</span>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=rb_res["soc"].values*100,
                                      line=dict(color="#ff6666",width=0.8),
                                      fill="tozeroy", fillcolor="rgba(255,100,100,0.05)",name="SOC %"))
            fig.add_hline(y=20, line_dash="dash", line_color="#ff4444", line_width=1)
            fig.add_hline(y=90, line_dash="dash", line_color="#333", line_width=1)
            fig.update_layout(**dark_layout("⚙ Rule-Based — SOC (%)"))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=ai_res["soc"].values*100,
                                      line=dict(color="#00cc66",width=0.8),
                                      fill="tozeroy", fillcolor="rgba(0,204,102,0.05)",name="SOC %"))
            fig.add_hline(y=20, line_dash="dash", line_color="#ff4444", line_width=1)
            fig.add_hline(y=90, line_dash="dash", line_color="#333", line_width=1)
            fig.update_layout(**dark_layout("🤖 AI Agent — SOC (%)"))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<span class="sec-label">BATTERY STATE OF HEALTH</span>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=rb_res["soh"].values*100,
                                      line=dict(color="#ff6666",width=1.5),
                                      fill="tozeroy", fillcolor="rgba(255,100,100,0.05)",name="SOH %"))
            fig.update_layout(**dark_layout("⚙ Rule-Based — SOH (%)"))
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=ai_res["soh"].values*100,
                                      line=dict(color="#00cc66",width=1.5),
                                      fill="tozeroy", fillcolor="rgba(0,204,102,0.05)",name="SOH %"))
            fig.update_layout(**dark_layout("🤖 AI Agent — SOH (%)"))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<span class="sec-label">CUMULATIVE ENERGY NOT SERVED (kWh)</span>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=np.cumsum(rb_res["ens"].values),
                                  line=dict(color="#ef4444",width=2),
                                  fill="tozeroy", fillcolor="rgba(239,68,68,0.05)",name="Rule-Based"))
        fig.add_trace(go.Scatter(y=np.cumsum(ai_res["ens"].values),
                                  line=dict(color="#10b981",width=2),
                                  fill="tozeroy", fillcolor="rgba(16,185,129,0.05)",name="AI Agent"))
        fig.update_layout(**dark_layout("Cumulative ENS — AI vs Rule-Based", height=280))
        st.plotly_chart(fig, use_container_width=True)

        # Curtailment chart
        st.markdown('<span class="sec-label">CURTAILED ENERGY & EXPORT REVENUE POTENTIAL</span>', unsafe_allow_html=True)
        c_curt1, c_curt2 = st.columns(2)
        with c_curt1:
            fig_c = go.Figure()
            fig_c.add_trace(go.Scatter(y=np.cumsum(rb_res["curtailed_kw"].values),
                                        line=dict(color="#f59e0b",width=2),
                                        fill="tozeroy", fillcolor="rgba(245,158,11,0.08)",
                                        name="Rule-Based Curtailment"))
            fig_c.add_trace(go.Scatter(y=np.cumsum(ai_res["curtailed_kw"].values),
                                        line=dict(color="#0ea5e9",width=2),
                                        fill="tozeroy", fillcolor="rgba(14,165,233,0.08)",
                                        name="AI Agent Curtailment"))
            fig_c.update_layout(**dark_layout("Cumulative Curtailed Energy (kWh)", height=260))
            st.plotly_chart(fig_c, use_container_width=True)
        with c_curt2:
            # Export revenue potential over time
            rb_rev = np.cumsum(rb_res["curtailed_kw"].values * 0.95 * EXPORT_TARIFF_GHS)
            ai_rev = np.cumsum(ai_res["curtailed_kw"].values * 0.95 * EXPORT_TARIFF_GHS)
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatter(y=rb_rev, line=dict(color="#f59e0b",width=2),
                                        fill="tozeroy", fillcolor="rgba(245,158,11,0.08)",
                                        name="Rule-Based Revenue"))
            fig_r.add_trace(go.Scatter(y=ai_rev, line=dict(color="#0ea5e9",width=2),
                                        fill="tozeroy", fillcolor="rgba(14,165,233,0.08)",
                                        name="AI Agent Revenue"))
            fig_r.update_layout(**dark_layout(f"Potential Export Revenue (GHS @ {EXPORT_TARIFF_GHS}/kWh)", height=260))
            st.plotly_chart(fig_r, use_container_width=True)

        # Export revenue info box
        rb_total_rev = ai_kpis.get("EXPORT_REVENUE", 0)
        ai_total_rev = rb_kpis.get("EXPORT_REVENUE", 0)
        st.markdown(f"""
        <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;padding:14px 18px;margin-bottom:16px;">
          <div style="font-family:Nunito,sans-serif;font-weight:700;color:#166534;font-size:14px;margin-bottom:8px;">
            &#9889; Grid Export Revenue Potential
          </div>
          <div style="font-family:Inter,sans-serif;font-size:12px;color:#166534;line-height:1.8;">
            Curtailed solar energy could be exported to the ECG grid at the feed-in tariff rate of
            <strong>GHS {EXPORT_TARIFF_GHS}/kWh</strong>. This represents additional revenue that would be
            unlocked with a grid interconnection agreement — requiring no changes to the AI control algorithm.
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Actions comparison (first 30 days)
        days_30 = min(30*24, len(rb_res))
        st.markdown('<span class="sec-label">BATTERY ACTIONS — FIRST 30 DAYS</span>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=rb_res["action"].values[:days_30],
                                  line=dict(color="#ff6666",width=0.8),name="Rule-Based"))
        fig.add_trace(go.Scatter(y=ai_res["action"].values[:days_30],
                                  line=dict(color="#00cc66",width=0.8),name="AI Agent"))
        fig.add_hline(y=0, line_color="#1a4a1a", line_width=1)
        fig.update_layout(**dark_layout("Action (+charge / −discharge) — First 30 Days", height=220))
        st.plotly_chart(fig, use_container_width=True)

        # Decision explainer (AI)
        if ai_dec:
            render_decision_panel(ai_dec, "🧠 AI DECISION LOG — FIRST 7 DAYS")

        # Full comparison table
        render_comparison_table(rb_kpis, ai_kpis)

        # Download
        combined = ai_res.copy()
        combined.columns = [f"ai_{c}" for c in combined.columns]
        combined[["rb_soc","rb_soh","rb_ens","rb_action"]] = rb_res[["soc","soh","ens","action"]].values
        csv = combined.to_csv(index=False)
        st.download_button("⬇ Download Full Results CSV", csv,
                           f"{sim_loc}_{sim_yrs}yr_results.csv", "text/csv")

    else:
        st.markdown("""
        <div style="text-align:center;padding:80px 20px;">
          <div style="font-size:48px;color:#bfdbfe;">⚡</div>
          <div style="font-family:Rajdhani,sans-serif;font-size:22px;color:#1e3a5f;letter-spacing:1px;margin-top:16px;">Ready to Simulate</div>
          <div style="font-family:Space Mono,monospace;font-size:11px;color:#64748b;margin-top:10px;">
            Configure load profile and location in sidebar, then click RUN
          </div>
        </div>
        """, unsafe_allow_html=True)

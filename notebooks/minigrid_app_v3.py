"""
AI Smart Battery Orchestration — Mini-Grid App v3
New features vs v1:
  • Live weather via Open-Meteo API (free, no key, true hourly, real solar irradiance)
  • Custom load profile builder (slider per hour OR CSV upload)
  • AI Decision Explainer — per-step reasoning panel
  • Live Mode: 72h simulation with real weather data
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
# LIVE WEATHER — OpenWeatherMap (free key required)
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

def fetch_live_weather(lat, lon, location_name, api_key=None):
    """
    Fetch 72-hour hourly forecast from Open-Meteo (free, no API key required).
    Returns a DataFrame with schema:
        datetime, ssrd_wm2, temp_c, tp, location, location_code,
        hour, month, dayofweek
    ssrd_wm2 comes directly from Open-Meteo's shortwave_radiation variable [W/m²].
    """
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude":              lat,
                "longitude":             lon,
                "hourly":                "shortwave_radiation,temperature_2m,precipitation",
                "forecast_days":         4,        # ~96 h; we use 72
                "timezone":              "UTC",
            },
            timeout=20,
        )
        if r.status_code != 200:
            try:    msg = r.json().get("reason", r.text)
            except: msg = r.text
            return None, f"Open-Meteo error {r.status_code}: {msg}"

        data    = r.json()
        hourly  = data["hourly"]
        times   = hourly["time"]                     # ISO-8601 strings, hourly
        ssrd    = hourly["shortwave_radiation"]       # W/m², already hourly
        temp    = hourly["temperature_2m"]            # °C
        precip  = hourly["precipitation"]             # mm/h

        rows = []
        for t, s, tc, tp in zip(times, ssrd, temp, precip):
            dt = pd.Timestamp(t).tz_localize(None)
            rows.append({
                "datetime":      dt,
                "ssrd_wm2":      float(s)  if s  is not None else 0.0,
                "temp_c":        float(tc) if tc is not None else 25.0,
                "tp":            float(tp) if tp is not None else 0.0,
                "location":      location_name,
                "location_code": 0,
                "hour":          dt.hour,
                "month":         dt.month,
                "dayofweek":     dt.dayofweek,
            })

        return pd.DataFrame(rows).copy(), None

    except requests.exceptions.ConnectionError:
        return None, "Network error — check your internet connection."
    except requests.exceptions.Timeout:
        return None, "Request timed out — try again."
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
    df = pd.read_csv("../data/master_dataset_scaled.csv")
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
    lstm.load_state_dict(torch.load("../models/best_lstm_scaled.pth", map_location="cpu"))
    lstm.eval()

    ppo = PPO.load("../models/best_model")

    class _Env(gym.Env):
        def __init__(self):
            super().__init__()
            self.action_space = spaces.Box(-1.,1.,(1,),np.float32)
            self.observation_space = spaces.Box(-np.inf,np.inf,(52,),np.float32)
        def reset(self,**kw): return np.zeros(52,np.float32),{}
        def step(self,a): return np.zeros(52,np.float32),0.,True,False,{}

    vec_norm = VecNormalize.load(
        "../models/vecnormalize_srep_avg_650kwh_4m.pkl",
        DummyVecEnv([_Env])
    )
    vec_norm.training = False; vec_norm.norm_reward = False
    return df, scaler_X, scaler_y, lstm, ppo, vec_norm


@st.cache_data
def load_new_location_df(location, train_df):
    """
    Load CDS ERA5 data for unseen locations (Accra, Bolgatanga, Akosombo).
    Mirrors build_eval_dataset() in phase6_evaluation_new_locations.py exactly.
    Returns (loc_df, location_code) or (None, None) if files not found.
    """
    LOCATION_CODES = {"Accra": 3, "Bolgatanga": 4, "Akosombo": 5}
    loc_code = LOCATION_CODES.get(location)
    if loc_code is None:
        return None, None

    data_path = "../data/"
    try:
        ssrd   = pd.read_csv(data_path + f"Solar Irradiance {location}.csv")
        precip = pd.read_csv(data_path + f"Precipitation {location}.csv")
        temp   = pd.read_csv(data_path + f"2m temperature {location}.csv")
    except FileNotFoundError:
        return None, None

    # Parse datetimes — strip timezone to match training data
    for d in [ssrd, precip, temp]:
        d["valid_time"] = pd.to_datetime(d["valid_time"]).dt.tz_localize(None)

    # Merge weather
    df = ssrd[["valid_time","ssrd"]].rename(columns={"valid_time":"datetime"})
    df = df.merge(precip[["valid_time","tp"]].rename(columns={"valid_time":"datetime"}), on="datetime")
    df = df.merge(temp[["valid_time","t2m"]].rename(columns={"valid_time":"datetime"}),   on="datetime")

    # Convert units — identical to training pipeline
    df["ssrd_wm2"] = df["ssrd"] / 3600
    df["temp_c"]   = df["t2m"] - 273.15

    # Load profile — tile Nigeria load same as training
    nigeria = pd.read_excel(data_path + "Nigeria Load data.xlsx", header=3)
    nigeria["date time"] = pd.to_datetime(nigeria["date time"])
    nigeria_clean = nigeria[["date time","National Suppressed Demand"]].copy()
    nigeria_clean.columns = ["datetime","demand_mw"]
    max_demand = nigeria_clean["demand_mw"].max()
    nigeria_clean["load_kw"] = (nigeria_clean["demand_mw"] / max_demand) * 50 * 6.5

    hourly_index = pd.date_range(start=df["datetime"].min(),
                                  end=df["datetime"].max(), freq="h")
    n_repeats = int(np.ceil(len(hourly_index) / len(nigeria_clean)))
    load_tiled = pd.DataFrame({
        "datetime": hourly_index,
        "load_kw":  np.tile(nigeria_clean["load_kw"].values, n_repeats)[:len(hourly_index)]
    })
    load_tiled["datetime"] = pd.to_datetime(load_tiled["datetime"]).dt.tz_localize(None)

    df = df.merge(load_tiled, on="datetime", how="inner")

    # Add time features
    df["location"]      = location
    df["location_code"] = loc_code
    df["hour"]          = df["datetime"].dt.hour
    df["month"]         = df["datetime"].dt.month
    df["dayofweek"]     = df["datetime"].dt.dayofweek
    df["tp"]            = df["tp"].fillna(0.0)

    df = df[["datetime","location","location_code","ssrd_wm2","tp",
             "temp_c","load_kw","hour","month","dayofweek"]].reset_index(drop=True)
    return df, loc_code


# ══════════════════════════════════════════════════════════════
# RULE-BASED CONTROLLER
# ══════════════════════════════════════════════════════════════

def rule_based_action(soc, solar_kw, load_kw, soc_min=0.20, soc_max=0.90):
    net = solar_kw - load_kw
    if net > 0:
        return np.array([min(net / 100.0, 1.0)]) if soc < soc_max else np.array([0.0])
    else:
        return np.array([-1.0]) if soc > soc_min else np.array([0.0])


# ══════════════════════════════════════════════════════════════
# DECISION EXPLAINER LOGIC
# ══════════════════════════════════════════════════════════════

def explain_decision(action, soc, solar_kw, load_kw, soh,
                     solar_forecast_next6, load_forecast_next6,
                     controller="AI"):
    """Return human-readable reasoning for a controller action."""
    MAX_RATE_KW = 130.0   # 0.2C on 650 kWh — must match phase3
    net = solar_kw - load_kw
    reasons = []
    decision_class = "hold"
    is_ai = (controller == "AI")

    if action > 0.05:
        decision_class = "charge"
        label = f"⬆ CHARGE  +{action:.2f}"
        kw = action * MAX_RATE_KW
        reasons.append(f"Charging at {kw:.0f} kW")
        if solar_kw > load_kw:
            reasons.append(f"Solar surplus: {net:.1f} kW available")
        if solar_kw == 0:
            reasons.append("No solar — charging from stored reserve (pre-emptive)")
        if not is_ai:
            reasons.append("Reactive: solar > load → charge at proportional rate")
        if is_ai:
            if soc < 0.5:
                reasons.append(f"SOC is low ({soc*100:.0f}%): building reserve")
            avg_solar_next = np.mean(solar_forecast_next6)
            if avg_solar_next < solar_kw * 0.5:
                reasons.append("LSTM forecast: solar declining, storing now for night coverage")
            if soh < 0.92:
                reasons.append(f"SOH {soh*100:.1f}%: moderate charge rate to limit stress")

    elif action < -0.05:
        decision_class = "discharge"
        label = f"⬇ DISCHARGE  {action:.2f}"
        kw = abs(action) * MAX_RATE_KW
        reasons.append(f"Discharging at {kw:.0f} kW")
        if load_kw > solar_kw:
            reasons.append(f"Load deficit: {load_kw - solar_kw:.1f} kW not covered by solar")
        if not is_ai:
            reasons.append("Reactive: load > solar → discharge at maximum rate")
        if is_ai:
            if soc > 0.6:
                reasons.append(f"SOC healthy ({soc*100:.0f}%): safe to discharge")
            avg_load_next = np.mean(load_forecast_next6)
            if avg_load_next > load_kw:
                reasons.append("LSTM forecast: rising load, strategic pre-emptive discharge")
            if soc <= 0.25:
                reasons.append("⚠ SOC critically low: minimising discharge depth")

    else:
        label = "⏸ HOLD"
        if not is_ai:
            reasons.append("Solar ≈ Load: no net surplus or deficit")
        else:
            reasons.append("Net energy balanced: no battery action needed")
            if abs(net) < 2:
                reasons.append(f"Solar ≈ Load ({solar_kw:.1f} ≈ {load_kw:.1f} kW)")
            if soc > 0.85:
                reasons.append("Battery near full: avoiding calendar aging")
            if 0.45 < soc < 0.65:
                reasons.append("SOC in optimal mid-range: holding to reduce wear")

    # ENS risk flag — applies to both
    if action >= 0 and load_kw > solar_kw and soc < 0.3:
        decision_class = "blackout"
        reasons.append("⚠ BLACKOUT RISK: Load unserved this hour")

    return decision_class, label, reasons


# ══════════════════════════════════════════════════════════════
# SIMULATION ENGINE — mirrors phase6_evaluation.py exactly
# ══════════════════════════════════════════════════════════════

def _get_forecast(step_idx, loc_df, scaler_X, scaler_y, lstm):
    """Identical to MiniGridEnv._get_forecast()"""
    feat_cols = ["ssrd_wm2","tp","temp_c","load_kw","location_code","hour","month","dayofweek"]
    lookback  = loc_df.iloc[step_idx-24:step_idx]
    X = scaler_X.transform(lookback[feat_cols].values)
    X_t = torch.FloatTensor(X).unsqueeze(0)
    with torch.no_grad():
        fc = lstm(X_t).numpy()[0]
    fc_inv = scaler_y.inverse_transform(fc.reshape(-1,2))
    return np.clip(fc_inv[:,0],0,None), np.clip(fc_inv[:,1],0,None)


def run_simulation(loc_df, lstm, scaler_X, scaler_y, ppo, vec_norm,
                   controller, years, progress_cb=None):
    """
    Direct reimplementation of phase6_evaluation.py evaluate() loop.
    Uses step-by-step LSTM inference — identical to the evaluation environment.
    """
    steps     = min(years * 365 * 24, len(loc_df) - 50)
    step_idx  = 24   # matches env.step_idx = 24 in reset()

    # Battery parameters — must match phase3/phase4 exactly
    battery_capacity    = 650.0
    max_charge_rate     = 130.0
    max_discharge_rate  = 130.0
    charge_efficiency   = 0.95
    discharge_efficiency= 0.95
    soc_min             = 0.20
    soc_max             = 0.90
    usable_range        = soc_max - soc_min  # 0.70
    solar_area_m2       = 176.7
    solar_norm          = solar_area_m2 * 0.75   # 132.525 kW peak
    load_norm           = 18.958 * 2.0            # MEAN_LOAD_KW * 2

    soc = 0.5; soh = 1.0; efc_total = 0.0
    daily_soc_min = 0.5; daily_soc_max = 0.5
    soc_out  = np.zeros(steps); soh_out  = np.zeros(steps)
    ens_out  = np.zeros(steps); act_out  = np.zeros(steps)
    sol_out  = np.zeros(steps); load_out = np.zeros(steps)
    curt_out = np.zeros(steps); efc_out  = np.zeros(steps)
    decisions = []

    for i in range(steps):
        # ── Get forecast (step-by-step, matches _get_forecast) ──
        sf, lf = _get_forecast(step_idx, loc_df, scaler_X, scaler_y, lstm)

        # ── Build observation (matches _get_obs) ─────────────────
        row = loc_df.iloc[step_idx]
        obs = np.concatenate([[soc, soh], sf/solar_norm, lf/load_norm,
            [row['hour']/23.0, row['month']/12.0]]).astype(np.float32)

        # ── Get action ───────────────────────────────────────────
        solar_kw = (row['ssrd_wm2'] / 1000.0) * solar_area_m2 * 0.75
        load_kw  = row['load_kw']

        if controller == "AI":
            obs_norm = vec_norm.normalize_obs(obs.reshape(1,-1))[0]
            action_arr, _ = ppo.predict(obs_norm, deterministic=True)
            action = float(np.clip(action_arr.flatten()[0], -1.0, 1.0))
        else:
            action_arr = rule_based_action(soc, solar_kw, load_kw)
            action = float(np.clip(action_arr[0], -1.0, 1.0))

        # ── Step physics (matches phase3/phase4 exactly) ─────────
        soc_before    = soc
        residual_load = max(0.0, load_kw - solar_kw)
        solar_surplus = max(0.0, solar_kw - load_kw)

        if action > 0:
            charge_kw = min(action * max_charge_rate,
                            (soc_max - soc) * battery_capacity)
            soc      += (charge_kw * charge_efficiency) / battery_capacity
            net_load  = residual_load
            curtailed = max(0.0, solar_surplus - charge_kw)
        else:
            discharge_kw = min(abs(action) * max_discharge_rate,
                               (soc - soc_min) * battery_capacity)
            soc         -= discharge_kw / battery_capacity
            net_load     = max(0.0, residual_load - discharge_kw * discharge_efficiency)
            curtailed    = solar_surplus

        soc         = float(np.clip(soc, soc_min, soc_max))
        ens_step    = min(max(0.0, net_load), load_kw)
        # ── DOD+EFC degradation (daily accumulator) ──────────────
        daily_soc_min = min(daily_soc_min, soc)
        daily_soc_max = max(daily_soc_max, soc)
        deg_cycle = 0.0
        if i % 24 == 23:
            daily_swing = daily_soc_max - daily_soc_min
            dod = daily_swing / usable_range
            if dod > 1e-6:
                efc_total += dod
                cycle_life = 3500.0 * (1.0 / max(dod, 0.01)) ** 1.5
                deg_cycle  = 0.20 / cycle_life
            daily_soc_min = soc; daily_soc_max = soc
        soh = max(0.0, soh - deg_cycle)

        soc_out[i]=soc; soh_out[i]=soh; ens_out[i]=ens_step
        act_out[i]=action; sol_out[i]=solar_kw; load_out[i]=load_kw
        curt_out[i]=curtailed; efc_out[i]=efc_total

        # ── Store decisions (first 7 days) ────────────────────────
        if i < 24*7:
            dc, lbl, rsns = explain_decision(action, soc, solar_kw, load_kw, soh, sf[:6], lf[:6], controller)
            decisions.append({
                "hour":     int(row['hour']),
                "datetime": f"D{i//24+1} H{int(row['hour']):02d}",
                "day":      i // 24 + 1,
                "step":     i,
                "class":    dc, "label": lbl, "reasons": rsns,
                "soc":      soc, "soh": soh,
                "solar":    solar_kw, "load": load_kw, "ens": ens_step,
            })

        step_idx += 1   # matches self.step_idx += 1

        if progress_cb and i % 5000 == 0:
            progress_cb(i / steps)

    return pd.DataFrame({
        "soc": soc_out, "soh": soh_out, "ens": ens_out,
        "solar_kw": sol_out, "load_kw": load_out, "action": act_out,
        "curtailed_kw": curt_out, "efc": efc_out,
    }), decisions


# ══════════════════════════════════════════════════════════════
# LIVE MODE — 24 h simulation on real weather + custom load
# ══════════════════════════════════════════════════════════════

def run_live_simulation(wx_df, load_profile_24h, lstm, scaler_X, scaler_y, ppo, vec_norm,
                        controller="AI"):
    """
    Step-by-step simulation on live Open-Meteo weather data starting from NOW.
    Physics, degradation model and observation normalisation are identical
    to phase3_environment.py / phase5_evaluation.py — rainflow cycle aging
    [Xu 2016] + Arrhenius calendar aging [Wang 2014] + lithium plating.
    Returns per-hour results DataFrame + decisions list.
    """

    import datetime as _dt

    # ── System constants — must match phase3/phase4/phase5 exactly ────
    BATTERY_KWH   = 650.0
    MAX_RATE_KW   = 130.0          # 0.2C
    CHARGE_EFF    = 0.95
    DISCHARGE_EFF = 0.95
    SOC_MIN       = 0.20
    SOC_MAX       = 0.90
    USABLE_RANGE  = SOC_MAX - SOC_MIN   # 0.70
    SOLAR_AREA_M2 = 176.7
    SOLAR_NORM    = SOLAR_AREA_M2 * 0.75   # 132.525 kW
    LOAD_NORM     = 18.958 * 2.0           # 37.916 kW
    FEAT_COLS     = ["ssrd_wm2","tp","temp_c","load_kw",
                     "location_code","hour","month","dayofweek"]

    # ── Step 1: Inject load profile ───────────────────────────────────
    # wx_df is already interpolated to hourly by the caller
    now_naive = pd.Timestamp(_dt.datetime.utcnow()).floor("h")

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

    # ── Step 2: Find current-hour index in API data ───────────────────
    diffs = (df_api["datetime"] - now_naive).abs()
    now_idx_api = int(diffs.idxmin())
    matched = df_api.iloc[now_idx_api]["datetime"]

    # ── Step 3: Prepend 24 padding rows for LSTM lookback ────────────
    pad_row  = df_api.iloc[0].copy()
    pad_rows = []
    for p in range(24):
        pr = pad_row.copy()
        pr["datetime"] = matched - _dt.timedelta(hours=24 - p)
        pr["hour"]     = int(pr["datetime"].hour)
        pad_rows.append(pr)
    df_full = pd.concat([pd.DataFrame(pad_rows), df_api], ignore_index=True)

    start_idx      = 24 + now_idx_api
    simulate_hours = min(72, len(df_full) - start_idx - 1)
    if simulate_hours < 1:
        return None, "Not enough future weather data — try again in a moment"

    X_all = scaler_X.transform(df_full[FEAT_COLS].values)

    # ── Battery state ─────────────────────────────────────────────────
    soc = 0.5;  soh = 1.0;  efc = 0.0
    # Rainflow half-cycle tracking (mirrors phase3 _compute_degradation)
    rf_direction  = 0
    rf_half_start = 0.5
    prev_soc      = 0.5
    daily_dod     = 0.0

    results   = []
    decisions = []

    for i in range(simulate_hours):
        idx = start_idx + i

        # ── LSTM forecast ─────────────────────────────────────────────
        window = X_all[idx - 24:idx]
        X_t    = torch.FloatTensor(window).unsqueeze(0)
        with torch.no_grad():
            fc = lstm(X_t).numpy()[0]
        fc_inv = scaler_y.inverse_transform(fc.reshape(-1, 2))
        sf = np.clip(fc_inv[:, 0], 0, None)
        lf = np.clip(fc_inv[:, 1], 0, None)

        row_data  = df_full.iloc[idx]
        solar_kw  = (float(row_data["ssrd_wm2"]) / 1000.0) * SOLAR_AREA_M2 * 0.75
        load_kw   = float(row_data["load_kw"])
        temp_c    = float(row_data["temp_c"])
        hour      = int(row_data["hour"])
        month     = int(row_data["month"])

        # ── Build observation (identical to SingleBatteryEnv._get_obs) ─
        obs = np.concatenate([
            [soc, soh],
            sf / SOLAR_NORM,
            lf / LOAD_NORM,
            [hour / 23.0, month / 12.0]
        ]).astype(np.float32)

        # ── Get action ────────────────────────────────────────────────
        if controller == "AI":
            obs_norm     = vec_norm.normalize_obs(obs.reshape(1, -1))[0]
            action_arr, _ = ppo.predict(obs_norm, deterministic=True)
            action = float(np.clip(action_arr.flatten()[0], -1.0, 1.0))
        else:
            action = float(np.clip(rule_based_action(soc, solar_kw, load_kw)[0], -1.0, 1.0))

        # ── Physics (identical to SingleBatteryEnv.step) ──────────────
        soc_before    = soc
        residual_load = max(0.0, load_kw - solar_kw)
        solar_surplus = max(0.0, solar_kw - load_kw)

        if action > 0:
            ckw      = min(action * MAX_RATE_KW, (SOC_MAX - soc) * BATTERY_KWH)
            soc     += (ckw * CHARGE_EFF) / BATTERY_KWH
            net_load = residual_load
            curtailed_kw = max(0.0, solar_surplus - ckw)
        else:
            dkw      = min(abs(action) * MAX_RATE_KW, (soc - SOC_MIN) * BATTERY_KWH)
            soc     -= dkw / BATTERY_KWH
            net_load = max(0.0, residual_load - dkw * DISCHARGE_EFF)
            curtailed_kw = solar_surplus

        soc      = float(np.clip(soc, SOC_MIN, SOC_MAX))
        ens_step = min(max(0.0, net_load), load_kw)

        # ── Degradation — exact copy of _compute_degradation ──────────
        # Rainflow half-cycle detection
        soc_change = soc - prev_soc
        deg_cycle  = 0.0
        if abs(soc_change) > 1e-4:
            new_dir = 1 if soc_change > 0 else -1
            if new_dir != rf_direction and rf_direction != 0:
                half_dod = abs(soc - rf_half_start) / USABLE_RANGE
                if half_dod > 1e-4:
                    efc          += half_dod * 0.5
                    cycle_life    = 3500.0 * (1.0 / max(half_dod, 0.01)) ** 1.5
                    deg_cycle     = (0.20 / cycle_life) * 0.5
                    daily_dod     = half_dod
                rf_half_start = soc
            rf_direction = new_dir
        prev_soc = soc

        # Arrhenius calendar aging
        cell_temp = temp_c + 5.0 + abs(action) * 3.0
        T_kelvin  = max(cell_temp + 273.15, 273.15)
        k_cal     = 14876.0 * np.exp(-24500.0 / (8.314 * T_kelvin))
        soc_stress = (0.70 + 0.60 * (soc - 0.50) ** 2 if soc >= 0.50
                      else 0.70 + 0.30 * (soc - 0.50) ** 2)
        deg_cal = 1.423e-6 * k_cal * soc_stress

        # Lithium plating (low-SOC penalty)
        deg_low_soc = 2e-4 * max(0.0, 0.30 - soc) ** 2

        soh = max(0.0, soh - (deg_cycle + deg_cal + deg_low_soc))

        # ── Explain decision ──────────────────────────────────────────
        dc, lbl, rsns = explain_decision(
            action, soc, solar_kw, load_kw, soh, sf[:6], lf[:6], controller)

        row_dt = row_data["datetime"]
        results.append({
            "hour": hour, "solar_kw": solar_kw, "load_kw": load_kw,
            "soc": soc, "soh": soh, "ens": ens_step,
            "action": action, "curtailed_kw": curtailed_kw,
            "efc": efc,
            "sf_solar": sf, "sf_load": lf,
        })
        decisions.append({
            "hour":    hour,
            "datetime": pd.Timestamp(row_dt).strftime("%H:%M"),
            "step": i + 1, "class": dc, "label": lbl,
            "reasons": rsns, "soc": soc, "soh": soh,
            "solar": solar_kw, "load": load_kw, "ens": ens_step,
        })

    return pd.DataFrame(results), decisions


# ══════════════════════════════════════════════════════════════
# COMPUTE KPIs
# ══════════════════════════════════════════════════════════════

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

    total_curtailed = round(float(np.sum(curt)), 1)
    curtail_pct     = round(float(total_curtailed / max(np.sum(sol),1) * 100), 1)

    return {
        "ENS":           round(float(np.sum(ens)), 1),
        "LOLP":          round(float(np.mean(ens > 0) * 100), 1),
        "SOH":           round(float(soh[-1] * 100), 1),
        "EFC":           round(float(res_df["efc"].iloc[-1]) if "efc" in res_df.columns
                               else float(np.sum(np.abs(act)) * 0.5), 0),
        "SCR":           scr,
        "MEAN_SOC":      round(float(np.mean(soc)) * 100, 1),
        "SOC_STD":       round(float(np.std(soc)), 4),
        "SERVED_PCT":    served_pct,
        "LIFESPAN":      lifespan,
        "CURTAILED_KWH": total_curtailed,
        "CURTAIL_PCT":   curtail_pct,
    }


# ══════════════════════════════════════════════════════════════
# RENDER HELPERS
# ══════════════════════════════════════════════════════════════

def dark_layout(title, height=240):
    return dict(
        title=dict(text=title, font=dict(color="#0f172a", family="Nunito", size=13), x=0),
        paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
        font=dict(color="#0f172a", family="Inter", size=11),
        margin=dict(l=40, r=10, t=36, b=30),
        xaxis=dict(gridcolor="#e2e8f0", showgrid=True, linecolor="#94a3b8",
                   color="#0f172a", tickfont=dict(color="#0f172a", size=11)),
        yaxis=dict(gridcolor="#e2e8f0", showgrid=True, linecolor="#94a3b8",
                   color="#0f172a", tickfont=dict(color="#0f172a", size=11)),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#e2e8f0",
                    borderwidth=1, font=dict(color="#0f172a", size=11)),
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
        <div class="kpi-lbl">Mean SOC</div>
        <div class="kpi-val">{kpis.get("MEAN_SOC",0):.1f}<span class="kpi-unit"> %</span></div>
      </div>
      <div class="kpi-card">
        <div class="kpi-lbl">SOC Std Deviation</div>
        <div class="kpi-val">{kpis.get("SOC_STD",0):.4f}</div>
      </div>
      <div class="kpi-card yel">
        <div class="kpi-lbl">Solar Curtailment</div>
        <div class="kpi-val yel">{kpis.get("CURTAIL_PCT",0):.1f}<span class="kpi-unit"> %</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_comparison_table(rb_kpis, ai_kpis):
    def badge(val, good):
        cls = "good" if good else "bad"
        return f'<span class="badge {cls}">{val}</span>'
    metrics = [
        ("Energy Not Served (kWh)",    "ENS",          False, "{:,.0f}"),
        ("Loss of Load Prob (%)",      "LOLP",         False, "{:.1f}%"),
        ("State of Health (%)",        "SOH",          True,  "{:.1f}%"),
        ("Projected Lifespan (yr)",    "LIFESPAN",     True,  "{:.1f} yr"),
        ("Equiv. Full Cycles",         "EFC",          False, "{:,.0f}"),
        ("Solar Self-Consumption (%)", "SCR",          True,  "{:.1f}%"),
        ("Load Served (%)",            "SERVED_PCT",   True,  "{:.1f}%"),
        ("Mean SOC (%)",               "MEAN_SOC",     True,  "{:.1f}%"),
        ("SOC Std Deviation",          "SOC_STD",      False, "{:.4f}"),
        ("Solar Curtailment (%)",      "CURTAIL_PCT",  False, "{:.1f}%"),
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


def render_decision_panel(decisions, title="🧠 AI DECISION LOG: FIRST 7 DAYS"):
    import streamlit.components.v1 as components

    st.markdown(f'<span class="sec-label">{title}</span>', unsafe_allow_html=True)

    shown = decisions[:min(72, len(decisions))]

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
        sol_pct  = min(d["solar"] / 132.525 * 100, 100)
        soc_col  = "#00cc66" if soc_pct > 50 else "#ffc107" if soc_pct > 30 else "#ff4444"
        ens_flag = " &#9889; ENS!" if d["ens"] > 0 else ""
        border   = border_map.get(dc, "#1a4a1a")
        acolor   = action_color_map.get(dc, "#00ff88")
        reasons_html = "".join(
            f'<div style="padding:2px 0;"><span style="color:{border};font-weight:700;">&#8226;</span> {r}</div>'
            for r in rsns
        )

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
              <span style="font-family:'Nunito',sans-serif;font-size:15px;
                           font-weight:800;color:{acolor};">{label}{ens_flag}</span>
              <span style="font-family:'Inter',sans-serif;font-size:11px;color:#64748b;">
                &#9728; <strong>{d['solar']:.1f} kW</strong> &nbsp; &#127968; <strong>{d['load']:.1f} kW</strong>
              </span>
            </div>
            <div style="font-family:'Inter',sans-serif;font-size:12px;font-weight:500;
                        color:#374151;line-height:1.8;">{reasons_html}</div>
            <div style="display:flex;gap:14px;align-items:center;margin-top:8px;">
              <div style="display:flex;flex-direction:column;gap:2px;">
                <div style="width:64px;height:7px;background:#e2e8f0;border-radius:4px;overflow:hidden;">
                  <div style="width:{soc_pct:.0f}%;height:100%;background:{soc_col};border-radius:4px;"></div>
                </div>
                <div style="font-size:10px;font-weight:600;color:#475569;font-family:'Inter',sans-serif;">
                  SOC <strong>{soc_pct:.0f}%</strong>
                </div>
              </div>
              <div style="display:flex;flex-direction:column;gap:2px;">
                <div style="width:64px;height:7px;background:#e2e8f0;border-radius:4px;overflow:hidden;">
                  <div style="width:{soh_pct:.0f}%;height:100%;background:#0ea5e9;border-radius:4px;"></div>
                </div>
                <div style="font-size:10px;font-weight:600;color:#475569;font-family:'Inter',sans-serif;">
                  SOH <strong>{soh_pct:.1f}%</strong>
                </div>
              </div>
              <div style="display:flex;flex-direction:column;gap:2px;">
                <div style="width:64px;height:7px;background:#e2e8f0;border-radius:4px;overflow:hidden;">
                  <div style="width:{sol_pct:.0f}%;height:100%;background:#f59e0b;border-radius:4px;"></div>
                </div>
                <div style="font-size:10px;font-weight:600;color:#475569;font-family:'Inter',sans-serif;">
                  Solar <strong>{d['solar']:.0f} kW</strong>
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
# DEFAULT LOAD PROFILES (kW per hour) — scaled to ~1,318 people
# Must match training: mean load = 18.958 kW, max ~37.9 kW (LOAD_NORM/2)
# ══════════════════════════════════════════════════════════════

DEFAULT_LOAD_PROFILES = {
    "Residential (typical Ghana)": [
         8,  7,  7,  7,  8, 10, 14, 18, 20, 18, 16, 16,
        17, 16, 15, 16, 18, 24, 28, 30, 28, 22, 16, 10
    ],
    "Mixed Commercial": [
        10,  9,  8,  8,  9, 12, 18, 25, 30, 32, 34, 33,
        32, 33, 32, 30, 28, 26, 24, 22, 20, 18, 15, 12
    ],
    "Agricultural / Irrigation": [
         5,  5,  5,  5,  5,  8, 15, 28, 35, 38, 38, 36,
        34, 36, 38, 36, 30, 20, 12,  8,  7,  6,  5,  5
    ],
    "Flat (constant)": [round(18.958, 1)] * 24,
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

    years = 6  # default; overridden below when Historical mode is active

    if app_mode == "📊 Historical Simulation":
        years = st.selectbox(
            "📅 Period", [1, 2, 3, 4, 5, 6],
            index=5,
            format_func=lambda x: f"Year {x} (2020–{2019+x})"
        )
    st.markdown("---")

    # Load profile builder — only relevant for Live Weather mode
    if app_mode == "📡 Live Weather (24 h)":
        st.markdown("### 🌤 WEATHER SOURCE")
        st.info("Open-Meteo · Free · No API key · Hourly · Real solar irradiance")
        st.markdown("---")
        st.markdown("### 🔌 LOAD PROFILE")
        load_source = st.radio("Load data source",
                               ["Preset profile", "Custom (sliders)", "Upload CSV"])

        if load_source == "Preset profile":
            preset = st.selectbox("Profile", list(DEFAULT_LOAD_PROFILES.keys()))
            load_profile = DEFAULT_LOAD_PROFILES[preset]

        elif load_source == "Custom (sliders)":
            st.caption("Set hourly demand (kW) — ~1,318 people, mean 18.958 kW")
            load_profile = []
            cols = st.columns(4)
            for h in range(24):
                with cols[h % 4]:
                    val = st.slider(f"H{h:02d}", 2, 40,
                                   DEFAULT_LOAD_PROFILES["Residential (typical Ghana)"][h],
                                   step=1,
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
    else:
        # Historical mode — uses actual dataset load
        load_profile = DEFAULT_LOAD_PROFILES["Residential (typical Ghana)"]
        st.markdown("### 📊 LOAD DATA")
        st.info("Using actual load from dataset — matches evaluation exactly.")

    st.markdown("---")
    if app_mode == "📡 Live Weather (24 h)":
        run_btn = st.button("⚡ FETCH & SIMULATE")
    else:
        run_btn = False  # historical mode shows results instantly — no button needed


# ══════════════════════════════════════════════════════════════
# LIVE WEATHER MODE
# ══════════════════════════════════════════════════════════════

if app_mode == "📡 Live Weather (24 h)":
    lat, lon, region = LOCATION_COORDS[location]

    # Always show weather card (auto-fetch)
    st.markdown('<span class="sec-label">LIVE WEATHER CONDITIONS</span>',
                unsafe_allow_html=True)

    wx_df = None

    if lat is not None:
        with st.spinner(f"Fetching live weather for {location}..."):
            wx_df, wx_err = fetch_live_weather(lat, lon, location)

        if wx_err or wx_df is None:
            st.error(f"Weather fetch error for {location}: {wx_err}. Try again in a moment.")
            wx_df = None
        else:
            # Open-Meteo returns true hourly data — no interpolation needed
            import datetime as _dt
            now_naive = pd.Timestamp(_dt.datetime.utcnow()).floor("h")
            time_diffs = (wx_df["datetime"] - now_naive).abs()
            now_i = int(time_diffs.idxmin())
            now_row = wx_df.iloc[now_i]

            icon = weather_icon(float(now_row["ssrd_wm2"]), int(now_row["hour"]))
            solar_kw_now = (float(now_row["ssrd_wm2"]) / 1000.0) * 176.7 * 0.75
            now_label = pd.Timestamp(now_row["datetime"]).strftime("%Y-%m-%d %H:%M")

            st.markdown(f"""
            <div class="wx-card">
              <div class="wx-icon">{icon}</div>
              <div class="wx-vals">
                <div class="wx-loc">{location}: {region}</div>
                <div class="wx-row">&#128336; Now (Ghana / UTC): <span>{now_label}</span></div>
                <div class="wx-row">&#127777; Temperature: <span>{float(now_row["temp_c"]):.1f} &deg;C</span></div>
                <div class="wx-row">&#9728; Solar irradiance: <span>{float(now_row["ssrd_wm2"]):.0f} W/m&sup2;</span>
                    &nbsp;&rarr; PV output: <span>{solar_kw_now:.1f} kW</span></div>
                <div class="wx-row">&#127783; Precipitation: <span>{float(now_row["tp"]):.1f} mm/h</span></div>
                <div class="wx-row">&#128197; Forecast range: <span>{wx_df["datetime"].iloc[0].strftime("%Y-%m-%d %H:%M")}</span>
                    to <span>{wx_df["datetime"].iloc[-1].strftime("%Y-%m-%d %H:%M")}</span></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── 24h forecast charts from NOW ───────────────────────────
            st.markdown('<span class="sec-label">24-HOUR WEATHER FORECAST</span>',
                        unsafe_allow_html=True)

            wx_24 = wx_df.iloc[now_i : now_i + 24].copy().reset_index(drop=True)
            dt_labels = wx_24["datetime"].dt.strftime("%H:%M").tolist()
            solar_24  = (wx_24["ssrd_wm2"].values / 1000.0) * 176.7 * 0.75
            load_24   = [load_profile[int(h) % 24] for h in wx_24["hour"].values]

            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=dt_labels, y=solar_24,
                                     marker_color="#f59e0b", name="Solar PV (kW)", opacity=0.85))
                fig.add_trace(go.Scatter(x=dt_labels, y=load_24,
                                         line=dict(color="#ef4444", width=2), name="Load (kW)"))
                fig.update_layout(**dark_layout(f"Solar PV vs Load — next 24h", height=260))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=dt_labels, y=wx_24["temp_c"].values,
                                          line=dict(color="#0ea5e9", width=2), fill="tozeroy",
                                          fillcolor="rgba(14,165,233,0.08)", name="Temp (°C)"))
                fig2.update_layout(**dark_layout(f"Temperature — next 24h", height=260))
                st.plotly_chart(fig2, use_container_width=True)

    if run_btn and wx_df is not None:
        with st.spinner("Running AI simulation on live weather..."):
            live_res, live_decisions = run_live_simulation(
                wx_df, load_profile, lstm, scaler_X, scaler_y, ppo, vec_norm)
        with st.spinner("Running Rule-Based simulation on live weather..."):
            live_res_rb, live_decisions_rb = run_live_simulation(
                wx_df, load_profile, lstm, scaler_X, scaler_y, ppo, vec_norm,
                controller="RB")

        if live_res is None:
            st.error(f"Simulation error: {live_decisions}")
        else:
            import datetime as _dt
            _now = _dt.datetime.utcnow()
            _end = _now + _dt.timedelta(hours=72)
            time_label = _now.strftime("%H:%M")
            end_label  = _end.strftime("%Y-%m-%d %H:%M")
            n_hours    = len(live_res)
            st.markdown(f'<span class="sec-label">SIMULATION RESULTS · {time_label} TODAY → {end_label} ({n_hours}h)</span>',
                        unsafe_allow_html=True)

            st.markdown("""
            <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;
                        padding:12px 16px;margin-bottom:16px;">
              <p style="font-size:12px;color:#1e40af;margin:0;">
                <strong>ℹ Note:</strong> The AI agent is optimised for long-term reliability and
                battery longevity — it preserves SOC strategically rather than discharging
                aggressively. Over short windows it may appear conservative compared to the
                rule-based controller. Performance advantages are most visible over multi-day
                and multi-year horizons (see Historical Simulation mode).
              </p>
            </div>
            """, unsafe_allow_html=True)

            hours = live_res["hour"].values

            # ── KPIs via shared render_kpis ────────────────────────
            ai_kpis = compute_kpis(live_res)
            rb_kpis = compute_kpis(live_res_rb) if live_res_rb is not None else ai_kpis

            st.markdown('<span class="sec-label">KPI SUMMARY: RULE-BASED vs AI</span>',
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

            render_comparison_table(rb_kpis, ai_kpis)

            # SOC + action side by side
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(range(len(live_res))),
                                          y=live_res["soc"].values*100,
                                          line=dict(color="#00cc66",width=2),
                                          fill="tozeroy",
                                          fillcolor="rgba(0,204,102,0.07)",
                                          name="AI SOC %"))
                if live_res_rb is not None:
                    fig.add_trace(go.Scatter(x=list(range(len(live_res_rb))),
                                              y=live_res_rb["soc"].values*100,
                                              line=dict(color="#ff6666",width=2,dash="dash"),
                                              name="Rule-Based SOC %"))
                fig.add_hline(y=20, line_dash="dash", line_color="#ff4444", line_width=1)
                fig.add_hline(y=90, line_dash="dash", line_color="#333", line_width=1)
                fig.update_layout(**dark_layout("Battery SOC: 24h (AI vs Rule-Based)", 240))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                colors = ["#00cc66" if a > 0.05 else "#ff6b35" if a < -0.05 else "#ffc107"
                          for a in live_res["action"].values]
                fig = go.Figure()
                fig.add_trace(go.Bar(x=list(range(len(live_res))),
                                      y=live_res["action"].values,
                                      marker_color=colors, name="AI Action"))
                if live_res_rb is not None:
                    fig.add_trace(go.Scatter(x=list(range(len(live_res_rb))),
                                              y=live_res_rb["action"].values,
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
            fig.update_layout(**dark_layout("Solar, Load, ENS and Curtailed: 24h", 280))
            st.plotly_chart(fig, use_container_width=True)

            # Decision explainer — Rule-Based vs AI side by side
            from datetime import datetime, timezone, timedelta
            now_str  = datetime.now(timezone.utc).strftime("%H:%M")
            end_str  = (datetime.now(timezone.utc) + timedelta(hours=24)).strftime("%H:%M, %d %b")
            st.markdown('<span class="sec-label">DECISION LOG: RULE-BASED vs AI (NEXT 24 HOURS)</span>',
                        unsafe_allow_html=True)
            col_rb_live, col_ai_live = st.columns(2)
            with col_rb_live:
                render_decision_panel(live_decisions_rb, f"⚙ Rule-Based: {now_str} to {end_str}")
            with col_ai_live:
                render_decision_panel(live_decisions, f"🧠 AI Agent: {now_str} to {end_str}")

            # Download button
            csv = live_res.drop(columns=["sf_solar","sf_load"], errors="ignore").to_csv(index=False)
            st.download_button("⬇ Download 24h results CSV", csv,
                               "live_24h_results.csv", "text/csv")

    elif run_btn and wx_df is None:
        st.error("Cannot simulate — weather data unavailable.")



# ══════════════════════════════════════════════════════════════
# HISTORICAL RESULTS MODE — reads eval_results_by_year.csv
# ══════════════════════════════════════════════════════════════

else:
    TRAIN_LOCS = ["Tamale", "Kumasi", "Axim"]
    loc_split  = "train" if location in TRAIN_LOCS else "unseen"
    is_new_loc = loc_split == "unseen"

    # ── Sidebar location info ─────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        if is_new_loc:
            st.info("📡 Unseen location — ERA5 CDS data")
        info_map = {
            "Tamale":     ("Northern",  "Hot & Semi-Arid"),
            "Kumasi":     ("Ashanti",   "Humid Tropical"),
            "Axim":       ("Western",   "Coastal Wet"),
            "Accra":      ("Gr. Accra", "Coastal"),
            "Bolgatanga": ("Upper E.",  "Hot Semi-Arid"),
            "Akosombo":   ("Eastern",   "Sub-humid"),
        }
        reg, clim = info_map.get(location, ("—", "—"))
        data_src  = "ERA5 CDS (unseen)" if is_new_loc else "Training dataset"
        st.markdown(f"""
        **📍 {location}**  
        Region: {reg} | Climate: {clim}  
        Data: {data_src}  
        Battery: 650 kWh LiFePO₄ | Solar: 132.5 kWp  
        📊 Reads pre-computed CSV instantly
        """)

    # ── Load CSV ──────────────────────────────────────────────
    @st.cache_data
    def load_eval_csv():
        try:
            return pd.read_csv("../data/eval_results_by_year.csv"), None
        except FileNotFoundError:
            return None, "eval_results_by_year.csv not found — run phase5_evaluation.py first."

    eval_df, eval_err = load_eval_csv()
    if eval_df is None:
        st.error(f"⚠ {eval_err}")
        st.info("Run phase5_evaluation.py to generate the results CSV, then restart the app.")
        st.stop()

    # ── Filter: location + year <= selected ──────────────────
    csv_key  = f"{location} ({loc_split})"
    loc_all  = eval_df[eval_df["location"] == csv_key]
    if loc_all.empty:
        st.warning(f"No data for '{csv_key}'. Available: {eval_df['location'].unique().tolist()}")
        st.stop()

    loc_filtered = loc_all[loc_all["year"] <= years]
    if loc_filtered.empty:
        st.warning(f"No data for '{csv_key}' up to year {years}.")
        st.stop()

    # ── Aggregate KPIs across selected years ─────────────────
    def kpis_from_years(ctrl):
        yrs  = loc_filtered[loc_filtered["controller"] == ctrl].sort_values("year")
        if yrs.empty:
            return {}
        last        = yrs.iloc[-1]
        curtail_pct = float(yrs["curtail_pct"].mean())
        scr         = round(min((1.0 - curtail_pct / 100.0) * 100.0, 100.0), 1)
        return {
            "ENS":        round(float(yrs["total_ens_kwh"].sum()), 1),
            "LOLP":       round(float(yrs["lolp_pct"].mean()), 1),
            "SOH":        round(float(last["final_soh"]) * 100, 1),
            "LIFESPAN":   round(float(last["lifespan_years"]), 1),
            "EFC":        round(float(yrs["efc_this_year"].sum()), 0),
            "SCR":        scr,
            "MEAN_SOC":   round(float(yrs["mean_soc"].mean()) * 100, 1),
            "SOC_STD":    round(float(yrs["soc_std"].mean()), 4),
            "SERVED_PCT": round(float(yrs["load_served_pct"].mean()), 1),
            "CURTAILED_KWH": 0,
            "CURTAIL_PCT": round(curtail_pct, 1),
        }

    rb_kpis = kpis_from_years("Rule-Based")
    ai_kpis = kpis_from_years("PPO")
    if not rb_kpis or not ai_kpis:
        st.warning("Missing Rule-Based or PPO rows for this location.")
        st.stop()

    # ── Header ────────────────────────────────────────────────
    data_note = " (ERA5 CDS)" if is_new_loc else ""
    yr_range  = f"Jan 2020 – End {2019 + years}" if years < 6 else "Jan 2020 – Jan 2026"
    yr_label  = f"{years} Year{'s' if years > 1 else ''} ({yr_range})"
    st.markdown(f'<span class="sec-label">RESULTS: {location}{data_note} · {yr_label}</span>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;
                padding:10px 16px;margin-bottom:16px;">
      <p style="font-size:12px;color:#1e40af;margin:0;">
        <strong>ℹ Pre-computed results from phase5_evaluation.py</strong> —
        exact numbers from the 6-year evaluation run, no re-simulation.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI cards ─────────────────────────────────────────────
    col_rb, col_vs, col_ai = st.columns([10, 1, 10])
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

    # ── SOH per-year table ────────────────────────────────────
    year_labels = ["End 2020","End 2021","End 2022","End 2023","End 2024","End 2025"]

    def soh_color(v):
        if v >= 0.95: return "#059669"
        if v >= 0.90: return "#2563eb"
        if v >= 0.85: return "#d97706"
        return "#ef4444"

    st.markdown('<span class="sec-label">STATE OF HEALTH — END OF EACH YEAR</span>',
                unsafe_allow_html=True)
    rb_yr = loc_filtered[loc_filtered["controller"] == "Rule-Based"].sort_values("year")
    ai_yr = loc_filtered[loc_filtered["controller"] == "PPO"].sort_values("year")
    rb_cells = "".join(
        f'<td style="color:{soh_color(float(r["final_soh"]))};font-weight:700;font-size:14px;text-align:center;padding:10px 12px;">' +
        f'{float(r["final_soh"])*100:.2f}%</td>'
        for _, r in rb_yr.iterrows())
    ai_cells = "".join(
        f'<td style="color:{soh_color(float(r["final_soh"]))};font-weight:700;font-size:14px;text-align:center;padding:10px 12px;">' +
        f'{float(r["final_soh"])*100:.2f}%</td>'
        for _, r in ai_yr.iterrows())
    yr_headers = "".join(
        f'<th style="text-align:center;padding:8px 12px;font-size:11px;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:0.8px;">{year_labels[int(r["year"])-1]}</th>'
        for _, r in rb_yr.iterrows())
    st.markdown(f"""
    <div style="background:#fff;border-radius:14px;padding:20px;box-shadow:0 2px 12px rgba(0,0,0,0.07);margin-bottom:16px;">
      <div style="font-family:Nunito,sans-serif;font-size:14px;font-weight:700;color:#1e3a5f;margin-bottom:14px;">🔋 Battery State of Health by Year</div>
      <table style="width:100%;border-collapse:collapse;">
        <thead><tr>
          <th style="text-align:left;padding:8px 12px;font-size:11px;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:0.8px;">Controller</th>
          {yr_headers}
        </tr></thead>
        <tbody>
          <tr style="border-top:1px solid #f1f5f9;">
            <td style="padding:10px 12px;font-weight:700;color:#ef4444;">⚙ Rule-Based</td>{rb_cells}
          </tr>
          <tr style="border-top:1px solid #f1f5f9;">
            <td style="padding:10px 12px;font-weight:700;color:#2563eb;">🤖 AI Agent</td>{ai_cells}
          </tr>
        </tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

    # ── Comparison table ──────────────────────────────────────
    render_comparison_table(rb_kpis, ai_kpis)

    # ── SOH trajectory chart — all locations ─────────────────
    st.markdown('<span class="sec-label">SOH ACROSS ALL LOCATIONS</span>', unsafe_allow_html=True)
    fig_soh = go.Figure()
    for ctrl, base_color, dash in [("PPO","#2563eb","solid"), ("Rule-Based","#ef4444","dot")]:
        for loc_name in eval_df["location"].unique():
            ld = eval_df[(eval_df["location"]==loc_name) & (eval_df["controller"]==ctrl) & (eval_df["year"]<=years)].sort_values("year")
            if ld.empty: continue
            unseen = "unseen" in str(loc_name)
            lname  = str(loc_name).replace(" (train)","").replace(" (unseen)","")
            clr    = "#7c3aed" if (unseen and ctrl=="PPO") else base_color
            xvals  = [year_labels[int(r["year"])-1] for _, r in ld.iterrows()]
            yvals  = [float(r["final_soh"])*100 for _, r in ld.iterrows()]
            fig_soh.add_trace(go.Scatter(x=xvals, y=yvals,
                name=f"{lname} ({'AI' if ctrl=='PPO' else 'RB'})",
                line=dict(color=clr, width=2 if ctrl=="PPO" else 1,
                          dash="solid" if (ctrl=="PPO" and not unseen) else ("dash" if ctrl=="PPO" else "dot")),
                mode="lines+markers", marker=dict(size=4),
                opacity=1.0 if ctrl=="PPO" else 0.5))
    fig_soh.add_hline(y=80, line_dash="dot", line_color="#ef4444", line_width=1.5,
                      annotation_text="80% EOL", annotation_font_color="#ef4444")
    fig_soh.update_layout(**dark_layout("Battery SOH (%) — All Locations", height=340))
    st.plotly_chart(fig_soh, use_container_width=True)

    # ── Load served bar chart — all locations ─────────────────
    st.markdown('<span class="sec-label">LOAD SERVED — ALL LOCATIONS</span>', unsafe_allow_html=True)
    ppo_grp = eval_df[(eval_df["controller"]=="PPO") & (eval_df["year"]<=years)].groupby("location")["load_served_pct"].mean().reset_index()
    rb_grp  = eval_df[(eval_df["controller"]=="Rule-Based") & (eval_df["year"]<=years)].groupby("location")["load_served_pct"].mean().reset_index()
    loc_labels_clean = ppo_grp["location"].str.replace(" (train)","",regex=False).str.replace(" (unseen)","",regex=False)
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(name="AI (PPO)", x=loc_labels_clean, y=ppo_grp["load_served_pct"].values,
        marker_color=["#2563eb" if "train" in l else "#7c3aed" for l in ppo_grp["location"]],
        opacity=0.9, text=[f"{v:.1f}%" for v in ppo_grp["load_served_pct"].values], textposition="outside"))
    fig_bar.add_trace(go.Bar(name="Rule-Based", x=loc_labels_clean, y=rb_grp["load_served_pct"].values,
        marker_color="rgba(239,68,68,0.4)", opacity=0.9,
        text=[f"{v:.1f}%" for v in rb_grp["load_served_pct"].values], textposition="outside"))
    layout_bar = dark_layout("Load Served (%) — PPO vs Rule-Based, All Locations", height=300)
    layout_bar["barmode"] = "group"
    layout_bar["yaxis"]   = dict(**layout_bar.get("yaxis",{}), range=[0, 105])
    fig_bar.update_layout(**layout_bar)
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Download ──────────────────────────────────────────────
    st.download_button("⬇ Download Results CSV",
                       loc_filtered.to_csv(index=False),
                       f"{location}_{years}yr_results.csv", "text/csv")

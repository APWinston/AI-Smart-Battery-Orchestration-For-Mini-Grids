"""
Ghana Mini-Grid Operator Decision Support Tool
================================================
Operators enter:
  • Their location (Ghana districts)
  • Battery capacity (kWh)
  • Solar PV size (kWp)
  • Typical daily load (kWh/day)
  • Current State of Charge (%)

The app fetches live weather from Open-Meteo (free, hourly, no API key),
runs the trained LSTM + PPO agent scaled to their system, and gives them:
  • A clear NOW action: Charge / Discharge / Hold
  • Hour-by-hour plan for the next 24 hours
  • Blackout risk alerts
  • SOC trajectory chart
  • Plain-English reasoning at every hour
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
import datetime as _dt
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Ghana Mini-Grid Operator",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Nunito:wght@600;700;800&display=swap');

.stApp, .main, .block-container { background:#f0f4f8 !important; font-family:'Inter',sans-serif; }
section[data-testid="stSidebar"] { background:#1e3a5f !important; }
section[data-testid="stSidebar"] * { color:#e8f0fe !important; }
section[data-testid="stSidebar"] hr { border-color:#2d5a8e !important; }
section[data-testid="stSidebar"] label { color:#93c5fd !important; font-size:13px !important; }
section[data-testid="stSidebar"] .stSlider > div { color:#e8f0fe !important; }

.stButton > button {
  background:linear-gradient(135deg,#1e3a5f,#2563eb) !important;
  border:none !important; color:#fff !important;
  font-family:'Nunito',sans-serif !important; font-weight:700 !important;
  font-size:16px !important; border-radius:10px !important;
  width:100% !important; padding:14px !important;
  box-shadow:0 4px 14px rgba(37,99,235,.35) !important;
}
.stButton > button:hover {
  background:linear-gradient(135deg,#2563eb,#3b82f6) !important;
  transform:translateY(-1px) !important;
}

.app-header {
  background:linear-gradient(135deg,#1e3a5f 0%,#2563eb 60%,#0ea5e9 100%);
  border-radius:16px; padding:24px 32px; margin-bottom:24px;
  box-shadow:0 8px 32px rgba(37,99,235,.2);
}
.app-title { font-family:'Nunito',sans-serif; font-size:26px; font-weight:800; color:#fff !important; margin:0; }
.app-sub { font-family:'Inter',sans-serif; font-size:13px; color:rgba(255,255,255,.8) !important; margin:6px 0 0; }

.sec-label {
  font-family:'Nunito',sans-serif; font-size:12px; font-weight:700;
  color:#1e3a5f !important; text-transform:uppercase; letter-spacing:1.5px;
  border-left:4px solid #2563eb; padding-left:12px;
  margin:24px 0 14px; display:block;
}

/* NOW action banner */
.action-banner {
  border-radius:16px; padding:24px 28px; margin-bottom:20px;
  box-shadow:0 4px 20px rgba(0,0,0,.12);
  display:flex; align-items:center; gap:20px;
}
.action-banner.charge    { background:linear-gradient(135deg,#059669,#10b981); }
.action-banner.discharge { background:linear-gradient(135deg,#d97706,#f59e0b); }
.action-banner.hold      { background:linear-gradient(135deg,#2563eb,#0ea5e9); }
.action-banner.blackout  { background:linear-gradient(135deg,#991b1b,#ef4444); }
.action-icon  { font-size:52px; }
.action-text  { flex:1; }
.action-main  { font-family:'Nunito',sans-serif; font-size:28px; font-weight:800; color:#fff !important; }
.action-sub   { font-family:'Inter',sans-serif;  font-size:14px; color:rgba(255,255,255,.85) !important; margin-top:4px; }
.action-kw    { font-family:'Nunito',sans-serif; font-size:22px; font-weight:700; color:rgba(255,255,255,.9) !important; text-align:right; }

/* Stat cards */
.stat-row    { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:16px; }
.stat-card   { background:#fff; border-radius:12px; padding:16px; box-shadow:0 2px 8px rgba(0,0,0,.07); border-top:4px solid #2563eb; }
.stat-card.grn  { border-top-color:#10b981; }
.stat-card.red  { border-top-color:#ef4444; }
.stat-card.yel  { border-top-color:#f59e0b; }
.stat-card.blu  { border-top-color:#0ea5e9; }
.stat-lbl { font-size:11px; font-weight:600; color:#64748b !important; text-transform:uppercase; letter-spacing:.8px; margin-bottom:6px; }
.stat-val { font-family:'Nunito',sans-serif; font-size:28px; font-weight:800; color:#1e3a5f !important; line-height:1; }
.stat-val.grn { color:#059669 !important; }
.stat-val.red { color:#ef4444 !important; }
.stat-val.yel { color:#d97706 !important; }
.stat-val.blu { color:#0284c7 !important; }
.stat-unit { font-size:13px; font-weight:500; opacity:.6; }

/* Alert box */
.alert-box { border-radius:10px; padding:12px 16px; margin-bottom:12px; }
.alert-box.red  { background:#fee2e2; border-left:4px solid #ef4444; }
.alert-box.yel  { background:#fef3c7; border-left:4px solid #f59e0b; }
.alert-box.grn  { background:#dcfce7; border-left:4px solid #10b981; }
.alert-box p { font-size:13px; color:#1e293b !important; margin:0; font-weight:500; }

/* Hour plan table */
.plan-table { width:100%; border-collapse:collapse; margin-top:8px; }
.plan-table th {
  font-family:'Inter',sans-serif; font-size:11px; font-weight:600;
  letter-spacing:.8px; text-transform:uppercase; padding:10px 12px;
  border-bottom:2px solid #e2e8f0; background:#f8fafc; color:#64748b !important; text-align:left;
}
.plan-table td { padding:9px 12px; font-size:13px; border-bottom:1px solid #f1f5f9; color:#334155 !important; }
.plan-table tr:hover td { background:#f8fafc; }
.pill {
  display:inline-block; padding:3px 10px; border-radius:20px;
  font-size:11px; font-weight:700; font-family:'Nunito',sans-serif;
}
.pill.charge    { background:#dcfce7; color:#166534 !important; }
.pill.discharge { background:#fef3c7; color:#92400e !important; }
.pill.hold      { background:#dbeafe; color:#1e40af !important; }
.pill.blackout  { background:#fee2e2; color:#991b1b !important; }

[data-testid="metric-container"] { background:#fff; border-radius:12px; padding:16px; box-shadow:0 2px 8px rgba(0,0,0,.07); }

/* Mobile responsive */
@media (max-width: 640px) {
  .stat-row { grid-template-columns:1fr 1fr !important; }
  .action-banner { flex-direction:column; padding:18px; }
  .action-kw { text-align:left !important; font-size:16px !important; }
  .action-main { font-size:22px !important; }
  .block-container { padding:0.5rem 0.5rem !important; }
  .plan-table td, .plan-table th { padding:7px 8px !important; font-size:12px !important; }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# GHANA LOCATIONS
# ══════════════════════════════════════════════════════════════

GHANA_LOCATIONS = {
    # Northern / Savannah
    "Tamale (Northern)":        (9.4035,  -0.8424),
    "Bolgatanga (Upper East)":  (10.7856, -0.8514),
    "Wa (Upper West)":          (10.0601, -2.5099),
    "Damongo (Savannah)":       (9.0836,  -1.8232),
    "Bole (Savannah)":          (9.0343,  -2.4851),
    "Nalerigu (NE Region)":     (10.5333, -0.3667),
    "Bawku (Upper East)":       (11.0594, -0.2421),
    # Brong-Ahafo / Bono
    "Sunyani (Bono)":           (7.3349,  -2.3280),
    "Techiman (Bono East)":     (7.5892,  -1.9379),
    "Kintampo (Bono East)":     (8.0594,  -1.7300),
    "Berekum (Bono)":           (7.4561,  -2.5880),
    # Ashanti
    "Kumasi (Ashanti)":         (6.6885,  -1.6244),
    "Obuasi (Ashanti)":         (6.2000,  -1.6667),
    "Mampong (Ashanti)":        (7.0608,  -1.4013),
    # Eastern
    "Akosombo (Eastern)":       (6.2950,  -0.0581),
    "Koforidua (Eastern)":      (6.0942,  -0.2574),
    "Nkawkaw (Eastern)":        (6.5575,  -0.7661),
    # Greater Accra / Central
    "Accra (Greater Accra)":    (5.5560,  -0.1969),
    "Tema (Greater Accra)":     (5.6698,  -0.0166),
    "Cape Coast (Central)":     (5.1053,  -1.2466),
    "Kasoa (Central)":          (5.5341,  -0.4220),
    # Western
    "Axim (Western)":           (4.8699,  -2.2372),
    "Takoradi (Western)":       (4.8982,  -1.7593),
    "Tarkwa (Western)":         (5.3009,  -1.9992),
    # Volta
    "Ho (Volta)":               (6.6008,  0.4716),
    "Hohoe (Volta)":            (7.1531,  0.4769),
    "Keta (Volta)":             (5.9167,  0.9833),
    # Custom
    "📍 Enter custom coordinates": (None, None),
}

DEFAULT_LOAD_PROFILES = {
    "Residential (villages)":    [ 8, 7, 7, 7, 8,10,14,18,20,18,16,16,17,16,15,16,18,24,28,30,28,22,16,10],
    "Mixed (commercial + homes)": [10, 9, 8, 8, 9,12,18,25,30,32,34,33,32,33,32,30,28,26,24,22,20,18,15,12],
    "Agric / irrigation":         [ 5, 5, 5, 5, 5, 8,15,28,35,38,38,36,34,36,38,36,30,20,12, 8, 7, 6, 5, 5],
}


# ══════════════════════════════════════════════════════════════
# MODEL CLASSES
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
# TRAINED SYSTEM CONSTANTS (must match phase3/phase5 exactly)
# ══════════════════════════════════════════════════════════════

TRAIN_BATTERY_KWH  = 650.0
TRAIN_SOLAR_KWP    = 132.525          # SOLAR_AREA_M2 * 0.75 * efficiency
TRAIN_SOLAR_AREA   = 176.7            # m²
TRAIN_MEAN_LOAD_KW = 18.958
TRAIN_SOC_MIN      = 0.20
TRAIN_SOC_MAX      = 0.90
TRAIN_MAX_RATE_KW  = 130.0            # 0.2C
TRAIN_SOLAR_NORM   = TRAIN_SOLAR_KWP
TRAIN_LOAD_NORM    = TRAIN_MEAN_LOAD_KW * 2.0
FEAT_COLS          = ["ssrd_wm2","tp","temp_c","load_kw","location_code","hour","month","dayofweek"]


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
    scaler_X.fit(df[FEAT_COLS])
    scaler_y.fit(df[["ssrd_wm2","load_kw"]])

    lstm = MiniGridLSTM(8, 128, 2, 24, 2)
    lstm.load_state_dict(torch.load("../models/best_lstm_scaled.pth", map_location="cpu"))
    lstm.eval()

    ppo = PPO.load("../models/best_model")

    class _Env(gym.Env):
        def __init__(self):
            super().__init__()
            self.action_space      = spaces.Box(-1., 1., (1,), np.float32)
            self.observation_space = spaces.Box(-np.inf, np.inf, (52,), np.float32)
        def reset(self, **kw): return np.zeros(52, np.float32), {}
        def step(self, a):     return np.zeros(52, np.float32), 0., True, False, {}

    vec_norm = VecNormalize.load(
        "../models/vecnormalize_srep_avg_650kwh_4m.pkl", DummyVecEnv([_Env]))
    vec_norm.training = False; vec_norm.norm_reward = False
    return df, scaler_X, scaler_y, lstm, ppo, vec_norm


# ══════════════════════════════════════════════════════════════
# OPEN-METEO WEATHER FETCH
# ══════════════════════════════════════════════════════════════

def fetch_weather(lat, lon):
    """Fetch 96-hour hourly forecast from Open-Meteo. No API key needed."""
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude":     lat,
            "longitude":    lon,
            "hourly":       "shortwave_radiation,temperature_2m,precipitation",
            "forecast_days": 2,
            "timezone":     "UTC",
        }, timeout=20)
        if r.status_code != 200:
            return None, f"Weather fetch failed ({r.status_code})"
        h = r.json()["hourly"]
        rows = []
        for t, s, tc, tp in zip(h["time"], h["shortwave_radiation"],
                                  h["temperature_2m"], h["precipitation"]):
            dt = pd.Timestamp(t).tz_localize(None)
            rows.append({
                "datetime": dt, "ssrd_wm2": float(s or 0),
                "temp_c": float(tc or 25), "tp": float(tp or 0),
                "location_code": 0, "hour": dt.hour,
                "month": dt.month, "dayofweek": dt.dayofweek,
            })
        return pd.DataFrame(rows), None
    except Exception as e:
        return None, str(e)


def weather_icon(ssrd, hour):
    if hour < 6 or hour > 20: return "🌙"
    if ssrd > 600: return "☀️"
    if ssrd > 300: return "⛅"
    return "☁️"


# ══════════════════════════════════════════════════════════════
# SIMULATION — scaled to operator's system
# ══════════════════════════════════════════════════════════════

def run_operator_simulation(wx_df, load_profile_24h, lstm, scaler_X, scaler_y,
                            ppo, vec_norm, initial_soc,
                            op_battery_kwh, op_solar_kwp, op_solar_area_m2):
    """
    Run the AI agent on live weather, scaled to the operator's system.

    The PPO agent was trained on the 650 kWh / 132.5 kWp reference system.
    Its observations use the training normalisation constants, so it outputs
    a fractional action in [−1, +1].

    We then apply that fraction to the operator's actual battery and solar:
        charge_kw  = action * (op_battery_kwh * 0.2)   # 0.2C rate
        solar_kw   = ssrd * op_solar_area_m2 * 0.75
        load_kw    = load_profile scaled to op system

    The LSTM forecast uses the training-scale observation (unchanged) so the
    agent's policy stays valid. Only the energy physics are rescaled.
    """
    now_naive = pd.Timestamp(_dt.datetime.utcnow()).floor("h")
    diffs = (wx_df["datetime"] - now_naive).abs()
    now_idx = int(diffs.idxmin())
    matched  = wx_df.iloc[now_idx]["datetime"]

    max_rate_op = op_battery_kwh * 0.2   # 0.2C in operator's kW

    # Build load_kw column (scaled proportionally to operator load)
    train_mean_load = TRAIN_MEAN_LOAD_KW
    op_mean_load    = np.mean(load_profile_24h)

    # Inject load into wx_df
    rows = []
    for i in range(len(wx_df)):
        row = wx_df.iloc[i]
        h   = int(row["hour"])
        # Load for the LSTM observation uses training-scale load
        train_load = float(load_profile_24h[h % 24]) * (train_mean_load / max(op_mean_load, 1e-3))
        rows.append({
            "datetime":      pd.Timestamp(row["datetime"]),
            "ssrd_wm2":      float(row["ssrd_wm2"]),
            "tp":            float(row["tp"]),
            "temp_c":        float(row["temp_c"]),
            "load_kw":       float(train_load),          # training-scale for LSTM
            "load_kw_op":    float(load_profile_24h[h % 24]),  # operator actual
            "location_code": int(row["location_code"]),
            "hour":          h,
            "month":         int(row["month"]),
            "dayofweek":     int(row["dayofweek"]),
        })
    df_api = pd.DataFrame(rows)

    if len(df_api) < 25:
        return None, "Not enough weather data"

    # Prepend 24 padding rows for LSTM lookback
    pad = df_api.iloc[0].copy()
    pads = []
    for p in range(24):
        pr = pad.copy()
        pr["datetime"] = matched - _dt.timedelta(hours=24-p)
        pr["hour"]     = int(pr["datetime"].hour)
        pads.append(pr)
    df_full = pd.concat([pd.DataFrame(pads), df_api], ignore_index=True)

    start_idx      = 24 + now_idx
    simulate_hours = min(24, len(df_full) - start_idx - 1)
    if simulate_hours < 1:
        return None, "Not enough future data"

    X_all = scaler_X.transform(df_full[FEAT_COLS].values)

    # Battery state — start at operator's current SOC
    soc           = float(np.clip(initial_soc / 100.0, TRAIN_SOC_MIN, TRAIN_SOC_MAX))
    soh           = 1.0
    efc           = 0.0
    rf_direction  = 0
    rf_half_start = soc
    prev_soc      = soc
    daily_dod     = 0.0

    results   = []
    decisions = []

    for i in range(simulate_hours):
        idx = start_idx + i

        # LSTM forecast (training-scale observation)
        X_t = torch.FloatTensor(X_all[idx-24:idx]).unsqueeze(0)
        with torch.no_grad():
            fc = lstm(X_t).numpy()[0]
        fc_inv = scaler_y.inverse_transform(fc.reshape(-1, 2))
        sf = np.clip(fc_inv[:, 0], 0, None)   # solar forecast W/m²
        lf = np.clip(fc_inv[:, 1], 0, None)   # load forecast kW (training scale)

        row_data      = df_full.iloc[idx]
        ssrd          = float(row_data["ssrd_wm2"])
        temp_c        = float(row_data["temp_c"])
        hour          = int(row_data["hour"])
        month         = int(row_data["month"])

        # Solar PV in operator's kW
        solar_kw_op   = (ssrd / 1000.0) * op_solar_area_m2 * 0.75
        # Solar PV in training scale (for observation)
        solar_kw_tr   = (ssrd / 1000.0) * TRAIN_SOLAR_AREA * 0.75
        # Load in operator's kW
        load_kw_op    = float(row_data["load_kw_op"])
        # Load in training scale (for observation)
        load_kw_tr    = float(row_data["load_kw"])

        # Build observation in TRAINING scale — agent policy stays valid
        obs = np.concatenate([
            [soc, soh],
            sf / TRAIN_SOLAR_NORM,
            lf / TRAIN_LOAD_NORM,
            [hour / 23.0, month / 12.0]
        ]).astype(np.float32)

        obs_norm      = vec_norm.normalize_obs(obs.reshape(1, -1))[0]
        action_arr, _ = ppo.predict(obs_norm, deterministic=True)
        action        = float(np.clip(action_arr.flatten()[0], -1.0, 1.0))

        # Physics in OPERATOR'S scale
        soc_before    = soc
        residual_load = max(0.0, load_kw_op - solar_kw_op)
        solar_surplus = max(0.0, solar_kw_op - load_kw_op)

        if action > 0:
            ckw          = min(action * max_rate_op, (TRAIN_SOC_MAX - soc) * op_battery_kwh)
            soc         += (ckw * 0.95) / op_battery_kwh
            net_load     = residual_load
            curtailed_kw = max(0.0, solar_surplus - ckw)
        else:
            dkw          = min(abs(action) * max_rate_op, (soc - TRAIN_SOC_MIN) * op_battery_kwh)
            soc         -= dkw / op_battery_kwh
            net_load     = max(0.0, residual_load - dkw * 0.95)
            curtailed_kw = solar_surplus

        soc      = float(np.clip(soc, TRAIN_SOC_MIN, TRAIN_SOC_MAX))
        ens_step = min(max(0.0, net_load), load_kw_op)

        # Degradation (phase3 exact)
        soc_change = soc - prev_soc
        deg_cycle  = 0.0
        if abs(soc_change) > 1e-4:
            new_dir = 1 if soc_change > 0 else -1
            if new_dir != rf_direction and rf_direction != 0:
                half_dod = abs(soc - rf_half_start) / (TRAIN_SOC_MAX - TRAIN_SOC_MIN)
                if half_dod > 1e-4:
                    efc          += half_dod * 0.5
                    cycle_life    = 3500.0 * (1.0 / max(half_dod, 0.01)) ** 1.5
                    deg_cycle     = (0.20 / cycle_life) * 0.5
                    daily_dod     = half_dod
                rf_half_start = soc
            rf_direction = new_dir
        prev_soc = soc
        cell_temp   = temp_c + 5.0 + abs(action) * 3.0
        T_k         = max(cell_temp + 273.15, 273.15)
        k_cal       = 14876.0 * np.exp(-24500.0 / (8.314 * T_k))
        soc_stress  = 0.70 + (0.60 if soc >= 0.50 else 0.30) * (soc - 0.50) ** 2
        deg_cal     = 1.423e-6 * k_cal * soc_stress
        deg_low_soc = 2e-4 * max(0.0, 0.30 - soc) ** 2
        soh         = max(0.0, soh - (deg_cycle + deg_cal + deg_low_soc))

        # Decision label and reasoning (operator-scale kW)
        dc, lbl, rsns = _explain(action, soc, solar_kw_op, load_kw_op, soh,
                                  sf, lf, max_rate_op, op_battery_kwh)

        dt_str = pd.Timestamp(row_data["datetime"]).strftime("%a %d %b %H:%M")
        results.append({
            "datetime": pd.Timestamp(row_data["datetime"]),
            "dt_str": dt_str, "hour": hour,
            "solar_kw": solar_kw_op, "load_kw": load_kw_op,
            "soc": soc, "soh": soh, "ens": ens_step,
            "action": action, "curtailed_kw": curtailed_kw, "efc": efc,
            "class": dc, "label": lbl, "reasons": rsns,
        })

    return pd.DataFrame(results), None


def _explain(action, soc, solar_kw, load_kw, soh, sf, lf, max_rate_kw, battery_kwh):
    """Operator-friendly plain-English decision explanation."""
    net     = solar_kw - load_kw
    reasons = []
    dc      = "hold"

    if action > 0.05:
        dc    = "charge"
        kw    = action * max_rate_kw
        label = f"⬆ CHARGE"
        reasons.append(f"Charge battery at {kw:.0f} kW")
        if solar_kw > load_kw:
            reasons.append(f"Solar surplus of {net:.1f} kW — store the excess")
        elif solar_kw == 0:
            reasons.append("Night-time: pre-charging to prepare for morning load")
        if soc < 0.45:
            reasons.append(f"Battery is at {soc*100:.0f}% — building reserve for overnight")
        if np.mean(sf) < solar_kw * 0.5:
            reasons.append("Solar expected to drop — store now while it's available")

    elif action < -0.05:
        dc    = "discharge"
        kw    = abs(action) * max_rate_kw
        label = f"⬇ DISCHARGE"
        reasons.append(f"Discharge battery at {kw:.0f} kW to cover load")
        if load_kw > solar_kw:
            reasons.append(f"Solar only covers {solar_kw:.1f} kW of {load_kw:.1f} kW demand")
        if soc > 0.6:
            reasons.append(f"Battery at {soc*100:.0f}% — safe to supply from storage")
        if soc <= 0.25:
            reasons.append(f"⚠ Battery low ({soc*100:.0f}%) — discharging slowly to conserve")

    else:
        label = "⏸ HOLD"
        reasons.append("Solar and load are balanced — no battery action needed")
        if abs(net) < 2:
            reasons.append(f"Solar ≈ Load ({solar_kw:.1f} ≈ {load_kw:.1f} kW)")
        if soc > 0.85:
            reasons.append("Battery nearly full — holding to protect battery health")
        if 0.45 < soc < 0.65:
            reasons.append(f"Battery at {soc*100:.0f}% — optimal range, no action needed")

    if action >= 0 and load_kw > solar_kw and soc < 0.30:
        dc = "blackout"
        reasons.append("⚠ BLACKOUT RISK — demand cannot be met this hour")

    return dc, label, reasons


# ══════════════════════════════════════════════════════════════
# RENDER HELPERS
# ══════════════════════════════════════════════════════════════

def dark_layout(title, height=260):
    return dict(
        title=dict(text=title, font=dict(color="#0f172a", family="Nunito", size=13), x=0),
        paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
        font=dict(color="#0f172a", family="Inter", size=11),
        margin=dict(l=40, r=10, t=36, b=30),
        xaxis=dict(gridcolor="#e2e8f0", showgrid=True, linecolor="#94a3b8",
                   color="#0f172a", tickfont=dict(color="#0f172a", size=10)),
        yaxis=dict(gridcolor="#e2e8f0", showgrid=True, linecolor="#94a3b8",
                   color="#0f172a", tickfont=dict(color="#0f172a", size=10)),
        legend=dict(bgcolor="rgba(255,255,255,.9)", bordercolor="#e2e8f0",
                    borderwidth=1, font=dict(color="#0f172a", size=11)),
        height=height,
    )


# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════

st.markdown("""
<div class="app-header">
  <p class="app-title">⚡ Ghana Mini-Grid Operator Tool</p>
  <p class="app-sub">AI battery decisions for Ghana mini-grid operators · Live weather · 24-hour plan</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════════════

with st.spinner("Loading AI models..."):
    try:
        df_train, scaler_X, scaler_y, lstm, ppo, vec_norm = load_resources()
    except Exception as e:
        st.error(f"Could not load models: {e}")
        st.stop()


# ══════════════════════════════════════════════════════════════
# SIDEBAR — OPERATOR INPUTS
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🔧 YOUR SYSTEM")
    st.markdown("---")

    # Location
    loc_name = st.selectbox("📍 Mini-Grid Location", list(GHANA_LOCATIONS.keys()), index=0)
    lat, lon  = GHANA_LOCATIONS[loc_name]
    if lat is None:
        lat = st.number_input("Latitude",  value=9.40, format="%.4f")
        lon = st.number_input("Longitude", value=-0.84, format="%.4f")

    st.markdown("---")

    # Battery
    st.markdown("### 🔋 Battery System")
    battery_kwh = st.number_input("Battery storage size (kWh)", min_value=10, max_value=5000,
                                   value=650, step=10)
    current_soc = st.slider("How full is your battery right now? (%)", 20, 100, 55, step=1)
    soc_color   = "🟢" if current_soc >= 50 else "🟡" if current_soc >= 30 else "🔴"
    st.caption(f"{soc_color} Battery is {current_soc}% full right now")

    st.markdown("---")

    # Solar
    st.markdown("### ☀️ Solar PV")
    solar_kwp = st.number_input("Solar panel size (kWp — the number on your inverter plate)", min_value=1, max_value=2000,
                                 value=133, step=1)
    # Derive panel area assuming standard 0.75 efficiency and 1000 W/m² STC
    solar_area_m2 = solar_kwp / 0.75

    st.markdown("---")

    # Load
    st.markdown("### 🏘 Community Load")
    load_preset = st.selectbox("What kind of community is this?", list(DEFAULT_LOAD_PROFILES.keys()))
    daily_load_kwh = st.number_input("Total electricity used per day (kWh/day)", min_value=10,
                                      max_value=50000, value=455, step=10)
    st.caption(f"≈ {daily_load_kwh/24:.1f} kW average")

    # Scale preset profile to match entered daily load
    base_profile  = DEFAULT_LOAD_PROFILES[load_preset]
    base_daily    = sum(base_profile)
    scale_factor  = daily_load_kwh / base_daily
    load_profile  = [round(v * scale_factor, 2) for v in base_profile]

    st.markdown("---")
    run_btn = st.button("⚡ GET RECOMMENDATIONS")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px;color:#93c5fd;line-height:1.6;">
    🌤 Weather: Open-Meteo (free, hourly)<br>
    🤖 AI trained on real Ghana mini-grid data<br>
    ☀ Solar: Real measured sunlight data<br>
    🔄 Updates: Each time you click
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# MAIN — FETCH + SIMULATE + DISPLAY
# ══════════════════════════════════════════════════════════════

if run_btn:
    # 1. Fetch weather
    with st.spinner(f"Fetching live weather for {loc_name}..."):
        wx_df, wx_err = fetch_weather(lat, lon)

    if wx_err or wx_df is None:
        st.error(f"⚠ Weather error: {wx_err}")
        st.stop()

    # 2. Run simulation
    with st.spinner("Calculating your 24-hour battery plan..."):
        res_df, sim_err = run_operator_simulation(
            wx_df, load_profile, lstm, scaler_X, scaler_y, ppo, vec_norm,
            current_soc, battery_kwh, solar_kwp, solar_area_m2)

    if sim_err or res_df is None:
        st.error(f"⚠ Simulation error: {sim_err}")
        st.stop()

    st.session_state["res_df"]   = res_df
    st.session_state["loc_name"] = loc_name
    st.session_state["bat_kwh"]  = battery_kwh
    st.session_state["sol_kwp"]  = solar_kwp
    st.session_state["soc_init"] = current_soc
    st.session_state["wx_df"]    = wx_df


# ── Display if results exist ────────────────────────────────
if "res_df" in st.session_state and st.session_state.get("loc_name") == loc_name:
    res_df     = st.session_state["res_df"]
    bat_kwh    = st.session_state["bat_kwh"]
    sol_kwp    = st.session_state["sol_kwp"]
    soc_init   = st.session_state["soc_init"]
    wx_df_disp = st.session_state.get("wx_df")

    now_row    = res_df.iloc[0]
    dc_now     = now_row["class"]
    lbl_now    = now_row["label"]
    soc_now    = now_row["soc"]          # SOC after hour 1 — used for trajectory only
    soc_input  = soc_init / 100.0        # operator's actual current SOC — used for banner & stat card
    solar_now  = now_row["solar_kw"]
    load_now   = now_row["load_kw"]
    icon_now   = weather_icon(float(wx_df_disp.iloc[0]["ssrd_wm2"]) if wx_df_disp is not None else 0,
                               int(now_row["hour"]))

    # ── Banner colours ────────────────────────────────────────
    banner_icons = {"charge":"🔋⬆","discharge":"🔋⬇","hold":"⏸","blackout":"🚨"}
    banner_icon  = banner_icons.get(dc_now, "⏸")

    # ── NOW ACTION BANNER ─────────────────────────────────────
    st.markdown(f"""
    <div class="action-banner {dc_now}">
      <div class="action-icon">{banner_icon}</div>
      <div class="action-text">
        <div class="action-main">{lbl_now} NOW</div>
        <div class="action-sub">
          {" · ".join(now_row["reasons"][:2])}
        </div>
      </div>
      <div class="action-kw">
        {icon_now} {solar_now:.1f} kW solar<br>
        🏘 {load_now:.1f} kW load<br>
        🔋 Battery {soc_input*100:.0f}% full
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── ALERTS ────────────────────────────────────────────────
    blackout_hours = res_df[res_df["class"] == "blackout"]
    low_soc_hours  = res_df[res_df["soc"] < 0.25]
    total_ens      = res_df["ens"].sum()

    if len(blackout_hours) > 0:
        st.markdown(f"""
        <div class="alert-box red">
          <p>🚨 <strong>Power cuts expected in {len(blackout_hours)} hour(s)</strong> over the next 24 hours —
          total unserved energy: {total_ens:.0f} kWh. Reduce non-essential loads where possible.</p>
        </div>""", unsafe_allow_html=True)
    elif len(low_soc_hours) > 0:
        st.markdown(f"""
        <div class="alert-box yel">
          <p>⚠ <strong>Battery will be very low (under 25%) in {len(low_soc_hours)} hour(s)</strong>.
          Monitor closely and consider reducing non-essential loads.</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-box grn">
          <p>✅ <strong>No blackout risk detected</strong> in the next 24 hours based on today's weather forecast.</p>
        </div>""", unsafe_allow_html=True)

    # ── SYSTEM STAT CARDS ─────────────────────────────────────
    total_load   = res_df["load_kw"].sum()
    served_pct   = round((1 - total_ens / max(total_load, 1)) * 100, 1)
    min_soc_24h  = res_df["soc"].min() * 100
    max_solar_24h= res_df["solar_kw"].max()
    soc_color_cls = "grn" if soc_input >= 0.5 else "yel" if soc_input >= 0.30 else "red"

    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-card {soc_color_cls}">
        <div class="stat-lbl">Battery Level Now</div>
        <div class="stat-val {soc_color_cls}">{soc_input*100:.0f}<span class="stat-unit"> %</span></div>
      </div>
      <div class="stat-card">
        <div class="stat-lbl">Solar Now</div>
        <div class="stat-val">{solar_now:.1f}<span class="stat-unit"> kW</span></div>
      </div>
      <div class="stat-card grn">
        <div class="stat-lbl">Power Supplied (today)</div>
        <div class="stat-val grn">{served_pct}<span class="stat-unit"> %</span></div>
      </div>
      <div class="stat-card {'red' if total_ens > 0 else 'grn'}">
        <div class="stat-lbl">Expected Power Cuts</div>
        <div class="stat-val {'red' if total_ens > 0 else 'grn'}">{total_ens:.0f}<span class="stat-unit"> kWh cut</span></div>
      </div>
      <div class="stat-card yel">
        <div class="stat-lbl">Lowest Battery (today)</div>
        <div class="stat-val yel">{min_soc_24h:.0f}<span class="stat-unit"> %</span></div>
      </div>
      <div class="stat-card blu">
        <div class="stat-lbl">Peak Solar Power</div>
        <div class="stat-val blu">{max_solar_24h:.1f}<span class="stat-unit"> kW</span></div>
      </div>
      <div class="stat-card">
        <div class="stat-lbl">Your Battery Size</div>
        <div class="stat-val">{bat_kwh}<span class="stat-unit"> kWh</span></div>
      </div>
      <div class="stat-card">
        <div class="stat-lbl">Your Solar Size</div>
        <div class="stat-val">{sol_kwp}<span class="stat-unit"> kWp peak</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SOC TRAJECTORY ───────────────────────────────────────
    st.markdown('<span class="sec-label">BATTERY LEVEL — NEXT 24 HOURS</span>',
                unsafe_allow_html=True)

    dt_labels = res_df["dt_str"].tolist()
    soc_vals  = (res_df["soc"].values * 100).tolist()

    # Colour each bar by SOC level; blackout hours always red
    blackout_set = set(res_df[res_df["class"] == "blackout"]["dt_str"].tolist())
    bar_colors = []
    for i, s in enumerate(soc_vals):
        if dt_labels[i] in blackout_set:
            bar_colors.append("#ef4444")
        elif s >= 50:   bar_colors.append("#10b981")
        elif s >= 30: bar_colors.append("#f59e0b")
        else:         bar_colors.append("#ef4444")

    fig_soc = go.Figure()
    fig_soc.add_trace(go.Bar(x=dt_labels, y=soc_vals, marker_color=bar_colors,
                              name="Battery Level %", opacity=0.85))
    fig_soc.add_hline(y=20, line_dash="dash", line_color="#ef4444", line_width=1.5,
                       annotation_text="Min 20%", annotation_font_color="#ef4444")
    fig_soc.add_hline(y=50, line_dash="dot", line_color="#10b981", line_width=1,
                       annotation_text="50%", annotation_font_color="#10b981")
    fig_soc.add_hline(y=90, line_dash="dash", line_color="#64748b", line_width=1,
                       annotation_text="Max 90%", annotation_font_color="#64748b")

    fig_soc.update_layout(**dark_layout("Battery Level (%) — Green: Good  ·  Yellow: Low  ·  Red: Very Low / Power Cut", 300))
    fig_soc.update_xaxes(tickangle=-45, nticks=24)
    st.plotly_chart(fig_soc, use_container_width=True)

    # ── SOLAR vs LOAD ─────────────────────────────────────────
    st.markdown('<span class="sec-label">SOLAR POWER vs COMMUNITY DEMAND</span>',
                unsafe_allow_html=True)

    fig_en = go.Figure()
    fig_en.add_trace(go.Bar(x=dt_labels, y=res_df["solar_kw"].values,
                             marker_color="#f59e0b", name="Solar PV (kW)", opacity=0.85))
    fig_en.add_trace(go.Scatter(x=dt_labels, y=res_df["load_kw"].values,
                                 line=dict(color="#2563eb", width=2), name="Load (kW)"))
    fig_en.add_trace(go.Bar(x=dt_labels, y=res_df["ens"].values,
                             marker_color="rgba(239,68,68,0.7)", name="Unserved (kWh)"))
    fig_en.update_layout(**dark_layout("Solar Power Generated vs Community Demand", 260))
    fig_en.update_xaxes(tickangle=-45, nticks=24)
    st.plotly_chart(fig_en, use_container_width=True)

    # ── 24-HOUR PLAN TABLE ────────────────────────────────────
    st.markdown('<span class="sec-label">24-HOUR PLAN</span>', unsafe_allow_html=True)

    pill_map = {
        "charge":    '<span class="pill charge">⬆ CHARGE</span>',
        "discharge": '<span class="pill discharge">⬇ DISCHARGE</span>',
        "hold":      '<span class="pill hold">⏸ HOLD</span>',
        "blackout":  '<span class="pill blackout">🚨 BLACKOUT RISK</span>',
    }

    table_rows = ""
    for _, row in res_df.iterrows():
        pill    = pill_map.get(row["class"], "")
        soc_pct = f"{row['soc']*100:.0f}%"
        soc_col = "#059669" if row["soc"] >= 0.5 else "#d97706" if row["soc"] >= 0.3 else "#ef4444"
        reason  = row["reasons"][0] if row["reasons"] else ""
        ens_txt = f'<span style="color:#ef4444;font-weight:600;">⚡ {row["ens"]:.1f} kWh cut</span>' if row["ens"] > 0 else "✅ OK"
        table_rows += f"""<tr>
          <td style="font-weight:600;color:#1e3a5f;">{row["dt_str"]}</td>
          <td>{pill}</td>
          <td style="color:{soc_col};font-weight:700;">{soc_pct}</td>
          <td style="color:#0284c7;">{row["solar_kw"]:.1f} kW</td>
          <td>{row["load_kw"]:.1f} kW</td>
          <td style="font-size:12px;color:#475569;">{reason}</td>
          <td>{ens_txt}</td>
        </tr>"""

    st.markdown(f"""
    <div style="background:#fff;border-radius:14px;padding:20px;box-shadow:0 2px 12px rgba(0,0,0,.07);overflow-x:auto;">
      <table class="plan-table">
        <thead><tr>
          <th>Time</th><th>What To Do</th><th>Battery Level</th>
          <th>Solar (kW)</th><th>Demand (kW)</th><th>Why</th><th>Power Cut?</th>
        </tr></thead>
        <tbody>{table_rows}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

    # ── DOWNLOAD ──────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    cols = ["dt_str","label","soc","solar_kw","load_kw","ens","reasons"]
    dl_df = res_df[cols].copy()
    dl_df["soc"] = (dl_df["soc"] * 100).round(1)
    dl_df["reasons"] = dl_df["reasons"].apply(lambda x: "; ".join(x))
    dl_df.columns = ["DateTime","Action","Battery_%","Solar_kW","Load_kW","PowerCut_kWh","Reasons"]
    st.download_button("⬇ Download 24h Plan (CSV)",
                       dl_df.to_csv(index=False),
                       f"battery_plan_24h_{loc_name.split('(')[0].strip().replace(' ','_')}.csv",
                       "text/csv")

else:
    # ── WELCOME SCREEN ────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;max-width:600px;margin:0 auto;">
      <div style="font-size:64px;margin-bottom:16px;">⚡🌍</div>
      <div style="font-family:Nunito,sans-serif;font-size:24px;font-weight:800;
                  color:#1e3a5f;margin-bottom:12px;">Welcome, Mini-Grid Operator</div>
      <div style="font-family:Inter,sans-serif;font-size:15px;color:#475569;line-height:1.7;margin-bottom:28px;">
        This tool uses an AI agent trained on real Ghana SREP data to help you
        manage your battery every hour — reducing blackouts and extending battery life.
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;text-align:left;margin-bottom:28px;">
        <div style="background:#fff;border-radius:12px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,.07);border-top:3px solid #2563eb;">
          <div style="font-size:20px;margin-bottom:8px;">📍</div>
          <div style="font-weight:700;color:#1e3a5f;margin-bottom:4px;">Enter your location</div>
          <div style="font-size:13px;color:#64748b;">Select from 27 Ghana districts</div>
        </div>
        <div style="background:#fff;border-radius:12px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,.07);border-top:3px solid #10b981;">
          <div style="font-size:20px;margin-bottom:8px;">🔋</div>
          <div style="font-weight:700;color:#1e3a5f;margin-bottom:4px;">Enter your system specs</div>
          <div style="font-size:13px;color:#64748b;">Battery kWh, Solar kWp, Daily load</div>
        </div>
        <div style="background:#fff;border-radius:12px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,.07);border-top:3px solid #f59e0b;">
          <div style="font-size:20px;margin-bottom:8px;">📊</div>
          <div style="font-weight:700;color:#1e3a5f;margin-bottom:4px;">Set current battery level</div>
          <div style="font-size:13px;color:#64748b;">Slide to your current battery level</div>
        </div>
        <div style="background:#fff;border-radius:12px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,.07);border-top:3px solid #8b5cf6;">
          <div style="font-size:20px;margin-bottom:8px;">⚡</div>
          <div style="font-weight:700;color:#1e3a5f;margin-bottom:4px;">Get your 24-hour plan</div>
          <div style="font-size:13px;color:#64748b;">Hour-by-hour decisions for today</div>
        </div>
      </div>
      <div style="font-family:Inter,sans-serif;font-size:13px;color:#94a3b8;">
        ← Fill in your system details in the sidebar and click <strong>GET RECOMMENDATIONS</strong>
      </div>
    </div>
    """, unsafe_allow_html=True)

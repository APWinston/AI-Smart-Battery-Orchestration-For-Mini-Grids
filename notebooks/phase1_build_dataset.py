#!/usr/bin/env python
# coding: utf-8
"""
Phase 1 — Build Scaled Dataset for Average Ghana SREP Mini-Grid
================================================================
Scales master_dataset.csv load to match the average SREP site:
  132.5 kWp solar | 650 kWh LiFePO4 battery | ~1,318 people
  Solar/Load ratio = 1.51 | 24.7h autonomy

Reference: Ghana SREP programme (AfDB/World Bank) — 35 mini-grids,
  4.525 MWp total = 132.5 kWp average per site, serving ~1,318 people

Output: ../data/master_dataset_scaled.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# SYSTEM CONSTANTS — Average Ghana SREP Site
# ============================================================
BATTERY_KWH      = 650.0
USABLE_KWH       = BATTERY_KWH * 0.70          # 455 kWh
MEAN_LOAD_KW     = USABLE_KWH / 24             # 18.958 kW (24h+ autonomy)
SOLAR_PEAK_KW    = 132.5                         # kWp at Ghana mean ssrd
SOLAR_AREA_M2    = 176.7                         # m2 — Solar/Load = 1.51
ORIG_MEAN_LOAD   = 192.9                        # kW from original dataset
SCALE_FACTOR     = MEAN_LOAD_KW / ORIG_MEAN_LOAD
POPULATION       = MEAN_LOAD_KW * 69.5         # ~1,318 people

# Verify solar/load ratio
_solar_mean = (216.0/1000.0) * SOLAR_AREA_M2 * 0.75
_ratio      = _solar_mean / MEAN_LOAD_KW

print("=" * 60)
print("  SCALING MASTER DATASET — Average Ghana SREP Site")
print("=" * 60)
print(f"  Solar peak:       {SOLAR_PEAK_KW:.0f} kWp (avg of 35 SREP sites)")
print(f"  Solar area:       {SOLAR_AREA_M2:.1f} m2")
print(f"  Battery:          {BATTERY_KWH:.0f} kWh LiFePO4")
print(f"  Usable:           {USABLE_KWH:.0f} kWh (70%)")
print(f"  Mean load:        {MEAN_LOAD_KW:.4f} kW")
print(f"  Population:       ~{POPULATION:.0f} people")
print(f"  Solar/Load ratio: {_ratio:.3f}  {'OK' if abs(_ratio-1.51)<0.02 else 'WARNING'}")
print(f"  Scale factor:     {SCALE_FACTOR:.5f}")
print()

# ============================================================
# LOAD AND SCALE
# ============================================================
print("Loading master_dataset.csv...")
df = pd.read_csv('../data/master_dataset.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
print(f"Loaded: {df.shape} | Locations: {df['location'].unique()}")
print(f"Original load — mean: {df['load_kw'].mean():.2f} kW")

df['load_kw'] = df['load_kw'] * SCALE_FACTOR
print(f"Scaled load   — mean: {df['load_kw'].mean():.4f} kW")
print()

# ============================================================
# VERIFY SOLAR/LOAD RATIO
# ============================================================
solar_mean_wm2 = df['ssrd_wm2'].mean()
solar_kw_mean  = (solar_mean_wm2/1000.0) * SOLAR_AREA_M2 * 0.75
ratio          = solar_kw_mean / df['load_kw'].mean()
print(f"Solar/Load ratio check:")
print(f"  Mean ssrd:    {solar_mean_wm2:.2f} W/m2")
print(f"  Solar area:   {SOLAR_AREA_M2} m2")
print(f"  Solar mean:   {solar_kw_mean:.4f} kW")
print(f"  Load mean:    {df['load_kw'].mean():.4f} kW")
print(f"  Ratio:        {ratio:.3f}  {'OK' if abs(ratio-1.51)<0.02 else 'WARNING'}")
print()

# ============================================================
# PER LOCATION STATS
# ============================================================
print("Per-location load stats:")
for loc in ['Tamale', 'Kumasi', 'Axim']:
    loc_df = df[df['location'] == loc]
    print(f"  {loc}: mean={loc_df['load_kw'].mean():.4f} kW | "
          f"max={loc_df['load_kw'].max():.4f} kW | "
          f"rows={len(loc_df):,}")

# ============================================================
# SAVE
# ============================================================
output_path = '../data/master_dataset_scaled.csv'
df.to_csv(output_path, index=False)
print(f"\nSaved: {output_path} | Shape: {df.shape}")

# ============================================================
# PLOT
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 6))
show    = 24 * 7
tamale  = df[df['location'] == 'Tamale'].reset_index(drop=True)
sol_kw  = (tamale['ssrd_wm2']/1000.0) * SOLAR_AREA_M2 * 0.75

axes[0].plot(tamale['load_kw'].values[:show], color='#ef4444',
             linewidth=1.2, label='Load (kW)')
axes[0].plot(sol_kw.values[:show], color='#f59e0b',
             linewidth=1.2, label='Solar (kW)', alpha=0.8)
axes[0].set_title(
    f'Average SREP Site — Tamale (First 7 Days) | '
    f'{SOLAR_PEAK_KW:.0f} kWp Solar | {BATTERY_KWH:.0f} kWh Battery | '
    f'Solar/Load = {ratio:.2f}', fontsize=11)
axes[0].set_ylabel('kW'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

tamale['hour'] = pd.to_datetime(tamale['datetime']).dt.hour
hourly_avg = tamale.groupby('hour')['load_kw'].mean()
axes[1].bar(hourly_avg.index, hourly_avg.values, color='#2563eb', alpha=0.8)
axes[1].axhline(y=MEAN_LOAD_KW, color='red', linestyle='--',
                linewidth=1.5, label=f'Mean: {MEAN_LOAD_KW:.2f} kW')
axes[1].set_title('Average Hourly Load Profile (Tamale)', fontsize=12)
axes[1].set_xlabel('Hour of Day'); axes[1].set_ylabel('kW')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../data/scaled_load_profile.png', dpi=150)
plt.show()
print("Plot saved: ../data/scaled_load_profile.png")
print("\nDataset ready. Run phase2_train_lstm.py next.")

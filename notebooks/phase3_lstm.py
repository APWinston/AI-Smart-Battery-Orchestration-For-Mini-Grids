#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

print("✅ All libraries imported successfully!")
print(f"PyTorch version: {torch.__version__}")

# ── 1. Load Data ───────────────────────────────────────────────
df = pd.read_csv('../data/master_dataset.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

print("✅ Master dataset loaded!")
print(f"Shape: {df.shape}")
print(f"Date range: {df['datetime'].min()} → {df['datetime'].max()}")
print(f"Locations: {df['location'].unique()}")

# ── 2. Feature Engineering ─────────────────────────────────────
df['location_code'] = df['location'].map({'Tamale': 0, 'Kumasi': 1, 'Axim': 2})
df['hour']          = df['datetime'].dt.hour
df['month']         = df['datetime'].dt.month
df['dayofweek']     = df['datetime'].dt.dayofweek

input_features = ['ssrd_wm2', 'tp', 'temp_c', 'load_kw',
                  'location_code', 'hour', 'month', 'dayofweek']
targets        = ['ssrd_wm2', 'load_kw']

# ── 3. Scale ───────────────────────────────────────────────────
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Sort by location then datetime — keeps each location's time series intact
df_sorted = df.sort_values(['location', 'datetime']).reset_index(drop=True)

X_scaled = scaler_X.fit_transform(df_sorted[input_features])
y_scaled = scaler_y.fit_transform(df_sorted[targets])

print(f"✅ Features scaled! X: {X_scaled.shape}, y: {y_scaled.shape}")

# ── 4. Create Sequences — Per Location ────────────────────────
# CRITICAL: sequences must be built per location separately
# otherwise sequences will cross the Tamale→Kumasi boundary
# and the LSTM will learn nonsense patterns

LOOKBACK = 24
FORECAST = 24

def create_sequences(X, y, lookback, forecast):
    Xs, ys = [], []
    for i in range(len(X) - lookback - forecast + 1):
        Xs.append(X[i : i + lookback])
        ys.append(y[i + lookback : i + lookback + forecast])
    return np.array(Xs), np.array(ys)

all_X, all_y = [], []
locations = ['Tamale', 'Kumasi', 'Axim']

print("Creating sequences per location...")
for loc in locations:
    loc_mask   = df_sorted['location'] == loc
    loc_idx    = df_sorted[loc_mask].index
    X_loc      = X_scaled[loc_idx]
    y_loc      = y_scaled[loc_idx]
    X_s, y_s   = create_sequences(X_loc, y_loc, LOOKBACK, FORECAST)
    all_X.append(X_s)
    all_y.append(y_s)
    print(f"  {loc}: {X_s.shape[0]:,} sequences")

X_seq = np.concatenate(all_X, axis=0)
y_seq = np.concatenate(all_y, axis=0)

print(f"\n✅ Total sequences: {X_seq.shape[0]:,}")
print(f"X_seq shape: {X_seq.shape}  → (samples, lookback, features)")
print(f"y_seq shape: {y_seq.shape}  → (samples, forecast, targets)")

# ── 5. Train/Val/Test Split (70/15/15) ─────────────────────────
# Shuffle before splitting so all locations are in every split
np.random.seed(42)
shuffle_idx = np.random.permutation(len(X_seq))
X_seq = X_seq[shuffle_idx]
y_seq = y_seq[shuffle_idx]

n         = len(X_seq)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

X_train, y_train = X_seq[:train_end],       y_seq[:train_end]
X_val,   y_val   = X_seq[train_end:val_end], y_seq[train_end:val_end]
X_test,  y_test  = X_seq[val_end:],          y_seq[val_end:]

print(f"\nTrain: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

# ── 6. DataLoaders ─────────────────────────────────────────────
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_val_t   = torch.FloatTensor(X_val)
y_val_t   = torch.FloatTensor(y_val)
X_test_t  = torch.FloatTensor(X_test)
y_test_t  = torch.FloatTensor(y_test)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                          batch_size=256, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t,   y_val_t),
                          batch_size=256, shuffle=False)

# ── 7. LSTM Model ──────────────────────────────────────────────
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
        h0  = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0  = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out.view(-1, self.forecast, self.output_size)

model = MiniGridLSTM(
    input_size  = 8,
    hidden_size = 128,
    num_layers  = 2,
    forecast    = 24,
    output_size = 2
)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n✅ LSTM Model built! Parameters: {total_params:,}")
print(model)

# ── 8. Training ────────────────────────────────────────────────
criterion     = nn.MSELoss()
optimizer     = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=3, factor=0.5)

EPOCHS        = 50   # increased from 30 — more epochs to converge
best_val_loss = float('inf')
train_losses, val_losses = [], []
patience_counter = 0
EARLY_STOP = 10   # stop if no improvement for 10 epochs

print("\n🚀 Starting training...")
print("-" * 60)

for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss   = criterion(output, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            val_loss += criterion(model(X_batch), y_batch).item()
    val_loss /= len(val_loader)

    scheduler.step(val_loss)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), '../models/best_lstm.pth')
        saved = "✅ saved"
    else:
        patience_counter += 1
        saved = ""

    if (epoch + 1) % 5 == 0 or saved:
        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f} {saved}")

    # Early stopping
    if patience_counter >= EARLY_STOP:
        print(f"\n⏹ Early stopping at epoch {epoch+1} — no improvement for {EARLY_STOP} epochs")
        break

print("-" * 60)
print(f"✅ Training complete! Best Val Loss: {best_val_loss:.6f}")
print("✅ Best model saved → ../models/best_lstm.pth")

# ── 9. Training Curve ──────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', color='blue',   linewidth=2)
plt.plot(val_losses,   label='Val Loss',   color='orange', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('LSTM Training & Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../data/lstm_training_curve.png', dpi=150)
plt.show()
print("✅ Training curve saved!")

# ── 10. Evaluate on Test Set ───────────────────────────────────
model.load_state_dict(torch.load('../models/best_lstm.pth'))
model.eval()

all_preds, all_actuals = [], []
with torch.no_grad():
    for i in range(0, len(X_test_t), 256):
        batch = X_test_t[i:i+256]
        all_preds.append(model(batch).numpy())
        all_actuals.append(y_test_t[i:i+256].numpy())

preds   = np.concatenate(all_preds,   axis=0)
actuals = np.concatenate(all_actuals, axis=0)

preds_inv   = scaler_y.inverse_transform(preds.reshape(-1, 2))
actuals_inv = scaler_y.inverse_transform(actuals.reshape(-1, 2))

print("\n📊 Test Set Performance:")
print("-" * 40)
for i, target in enumerate(['Solar Irradiance (W/m²)', 'Load (kW)']):
    mae  = mean_absolute_error(actuals_inv[:, i], preds_inv[:, i])
    rmse = np.sqrt(mean_squared_error(actuals_inv[:, i], preds_inv[:, i]))
    # Normalised MAE as % of mean
    nmae = mae / max(actuals_inv[:, i].mean(), 1) * 100
    print(f"{target}:")
    print(f"   MAE:  {mae:.2f}  ({nmae:.1f}% of mean)")
    print(f"   RMSE: {rmse:.2f}\n")

print("✅ Evaluation complete!")

# ── 11. Forecast vs Actual Plot ────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(15, 8))
hours = 24 * 7  # 7 days

axes[0].plot(actuals_inv[:hours, 0], label='Actual',    color='orange', linewidth=1.5)
axes[0].plot(preds_inv[:hours,   0], label='Predicted', color='blue',
             linewidth=1.5, linestyle='--')
axes[0].set_title('Solar Irradiance — Actual vs Predicted (7 Days)', fontsize=13)
axes[0].set_ylabel('W/m²')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(actuals_inv[:hours, 1], label='Actual',    color='green', linewidth=1.5)
axes[1].plot(preds_inv[:hours,   1], label='Predicted', color='red',
             linewidth=1.5, linestyle='--')
axes[1].set_title('Load Demand — Actual vs Predicted (7 Days)', fontsize=13)
axes[1].set_ylabel('kW')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../data/lstm_forecast_vs_actual.png', dpi=150)
plt.show()
print("✅ Forecast plot saved!")

#!/usr/bin/env python
# coding: utf-8
"""
Phase 2 — Train LSTM Forecaster on Scaled Dataset
===================================================
Trains LSTM on master_dataset_scaled.csv (18.96 kW mean load).
Architecture: 8 inputs, 128 hidden, 2 layers, 24h lookback,
24h forecast, outputs [solar W/m2, load kW].

Output: ../models/best_lstm_scaled.pth
"""

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

print(f"Libraries imported! PyTorch: {torch.__version__}")

# ============================================================
# 1. LOAD SCALED DATA
# ============================================================
df = pd.read_csv('../data/master_dataset_scaled.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
print(f"\nScaled dataset: {df.shape}")
print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"Mean load: {df['load_kw'].mean():.4f} kW (target: 18.96 kW)")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
df['location_code'] = df['location'].map({'Tamale':0,'Kumasi':1,'Axim':2})
df['hour']          = df['datetime'].dt.hour
df['month']         = df['datetime'].dt.month
df['dayofweek']     = df['datetime'].dt.dayofweek

input_features = ['ssrd_wm2','tp','temp_c','load_kw',
                  'location_code','hour','month','dayofweek']
targets        = ['ssrd_wm2','load_kw']

# ============================================================
# 3. SCALE FEATURES
# ============================================================
scaler_X  = MinMaxScaler()
scaler_y  = MinMaxScaler()
df_sorted = df.sort_values(['location','datetime']).reset_index(drop=True)
X_scaled  = scaler_X.fit_transform(df_sorted[input_features])
y_scaled  = scaler_y.fit_transform(df_sorted[targets])
print(f"Scaled! X: {X_scaled.shape} | y: {y_scaled.shape}")

# ============================================================
# 4. CREATE SEQUENCES PER LOCATION
# ============================================================
LOOKBACK = 24; FORECAST = 24

def create_sequences(X, y, lookback, forecast):
    Xs, ys = [], []
    for i in range(len(X) - lookback - forecast + 1):
        Xs.append(X[i:i+lookback])
        ys.append(y[i+lookback:i+lookback+forecast])
    return np.array(Xs), np.array(ys)

all_X, all_y = [], []
for loc in ['Tamale','Kumasi','Axim']:
    loc_idx  = df_sorted[df_sorted['location']==loc].index
    X_s, y_s = create_sequences(X_scaled[loc_idx], y_scaled[loc_idx],
                                 LOOKBACK, FORECAST)
    all_X.append(X_s); all_y.append(y_s)
    print(f"  {loc}: {X_s.shape[0]:,} sequences")

X_seq = np.concatenate(all_X); y_seq = np.concatenate(all_y)
print(f"Total: {X_seq.shape[0]:,} sequences")

# ============================================================
# 5. SPLIT (70/15/15)
# ============================================================
np.random.seed(42)
idx   = np.random.permutation(len(X_seq))
X_seq = X_seq[idx]; y_seq = y_seq[idx]
n     = len(X_seq)
X_train,y_train = X_seq[:int(n*.70)],            y_seq[:int(n*.70)]
X_val,  y_val   = X_seq[int(n*.70):int(n*.85)],  y_seq[int(n*.70):int(n*.85)]
X_test, y_test  = X_seq[int(n*.85):],            y_seq[int(n*.85):]
print(f"Train:{len(X_train):,} Val:{len(X_val):,} Test:{len(X_test):,}")

# ============================================================
# 6. DATALOADERS
# ============================================================
train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train),
               torch.FloatTensor(y_train)), batch_size=256, shuffle=True)
val_loader   = DataLoader(TensorDataset(torch.FloatTensor(X_val),
               torch.FloatTensor(y_val)),   batch_size=256, shuffle=False)

# ============================================================
# 7. LSTM MODEL
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

model = MiniGridLSTM(8, 128, 2, 24, 2)
print(f"\nLSTM: {sum(p.numel() for p in model.parameters()):,} parameters")

# ============================================================
# 8. TRAINING
# ============================================================
criterion   = nn.MSELoss()
optimizer   = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(
              optimizer, patience=3, factor=0.5)
EPOCHS=50; EARLY_STOP=10; best_val=float('inf'); patience_ct=0
train_losses=[]; val_losses=[]

print("\nTraining..."); print("-"*55)
for epoch in range(EPOCHS):
    model.train(); t_loss=0
    for Xb,yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); t_loss += loss.item()
    t_loss /= len(train_loader)

    model.eval(); v_loss=0
    with torch.no_grad():
        for Xb,yb in val_loader:
            v_loss += criterion(model(Xb),yb).item()
    v_loss /= len(val_loader)
    scheduler.step(v_loss)
    train_losses.append(t_loss); val_losses.append(v_loss)

    if v_loss < best_val:
        best_val=v_loss; patience_ct=0
        torch.save(model.state_dict(),'../models/best_lstm_scaled.pth')
        tag="saved"
    else:
        patience_ct+=1; tag=""

    if (epoch+1)%5==0 or tag:
        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] Train:{t_loss:.6f} Val:{v_loss:.6f} {tag}")
    if patience_ct>=EARLY_STOP:
        print(f"\nEarly stopping at epoch {epoch+1}"); break

print("-"*55)
print(f"Best Val Loss: {best_val:.6f}")
print("Saved: ../models/best_lstm_scaled.pth")

# ============================================================
# 9. TRAINING CURVE
# ============================================================
plt.figure(figsize=(10,4))
plt.plot(train_losses, label='Train', color='blue',   linewidth=2)
plt.plot(val_losses,   label='Val',   color='orange', linewidth=2)
plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
plt.title('LSTM Training — Average SREP Site (132.5 kWp / 650 kWh)')
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig('../data/lstm_scaled_training_curve.png', dpi=150)
plt.show(); print("Training curve saved!")

# ============================================================
# 10. TEST EVALUATION
# ============================================================
model.load_state_dict(torch.load('../models/best_lstm_scaled.pth'))
model.eval()
preds,actuals=[],[]
with torch.no_grad():
    for i in range(0,len(X_test),256):
        preds.append(model(torch.FloatTensor(X_test[i:i+256])).numpy())
        actuals.append(torch.FloatTensor(y_test[i:i+256]).numpy())

preds   = scaler_y.inverse_transform(np.concatenate(preds,  0).reshape(-1,2))
actuals = scaler_y.inverse_transform(np.concatenate(actuals,0).reshape(-1,2))

print("\nTest Set Performance:")
print("-"*40)
for i, name in enumerate(['Solar (W/m2)','Load (kW)']):
    mae  = mean_absolute_error(actuals[:,i], preds[:,i])
    rmse = np.sqrt(mean_squared_error(actuals[:,i], preds[:,i]))
    nmae = mae / max(actuals[:,i].mean(),1) * 100
    print(f"{name}: MAE={mae:.4f} ({nmae:.1f}% of mean) | RMSE={rmse:.4f}")

print("\nDone! Run phase3_environment.py next.")

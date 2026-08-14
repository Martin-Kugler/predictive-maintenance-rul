"""
src/models/sequence_models.py
LSTM and 1D-CNN in PyTorch to predict the RUL from multivariate
time series windows (n_sensors). Both receive input in the format 
(batch, window, n_features) and return a single RUL value per window.
"""

import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, window, n_features)
        out, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]           # (batch, hidden_size)
        return self.head(last_hidden).squeeze(-1)


class CNN1DRegressor(nn.Module):
    def __init__(self, n_features: int, window: int, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x arrives as (batch, window, n_features) -> Conv1d waits for (batch, channels, length)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        return self.head(x).squeeze(-1)


def train_torch_model(model, train_loader, val_loader, loss_fn, epochs=30, lr=1e-3, device="cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(loss_fn(pred, yb).item())

        history["train_loss"].append(sum(train_losses) / len(train_losses))
        history["val_loss"].append(sum(val_losses) / len(val_losses))
        print(f"Epoch {epoch+1}/{epochs} - train_loss: {history['train_loss'][-1]:.4f} "
              f"- val_loss: {history['val_loss'][-1]:.4f}")

    return model, history

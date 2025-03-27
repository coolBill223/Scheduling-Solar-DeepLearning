import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.datasets.data_loader import load_data
from models.solar_time_model_tiny import TinyMLP  # <-- Make sure this file exists with TinyMLP inside

# Load data and corresponding scalers
X, y, X_scaler, y_scaler = load_data()

# Check for NaNs in input and target tensors
print(f"NaNs in X: {torch.isnan(X).sum().item()}")
print(f"NaNs in y: {torch.isnan(y).sum().item()}")

# Split dataset into 70% training and 30% validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)

# Wrap data in PyTorch TensorDataset and DataLoader
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

# Initialize model
model = TinyMLP(input_dim=X.shape[1])

# Define loss function and optimizer
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)

# Setup for training loop
train_losses = []
val_losses = []
num_epochs = 300
best_val_loss = float("inf")
early_stopping_patience = 20
no_improve_count = 0

os.makedirs("checkpoints", exist_ok=True)

# Validation step
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        epoch_loss += loss.item()

    train_loss = epoch_loss / len(train_loader)
    val_loss = evaluate(model, val_loader)

    if np.isnan(train_loss) or np.isinf(train_loss):
        print(f"Warning: NaN or Inf detected in training loss at epoch {epoch}. Stopping training.")
        break

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch [{epoch}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "checkpoints/best_model.pth")
        no_improve_count = 0
    else:
        no_improve_count += 1

    # Early stopping
    if no_improve_count >= early_stopping_patience:
        print(f"Early stopping at epoch {epoch}")
        break

    # Adjust learning rate based on validation performance
    scheduler.step(val_loss)

# Save final model and scaler
torch.save(model.state_dict(), "checkpoints/final_model.pth")
joblib.dump(y_scaler, "checkpoints/y_scaler.pkl")

# Final evaluation: compute MSE and MPE
def evaluate_mse_mpe(model, X_tensor, y_tensor, scaler):
    model.eval()
    with torch.no_grad():
        preds = model(X_tensor)
        y_true = y_tensor.cpu().numpy().reshape(-1, 1)
        preds = preds.cpu().numpy().reshape(-1, 1)

        y_true = np.clip(scaler.inverse_transform(y_true).flatten(), 0, None)
        preds = np.clip(scaler.inverse_transform(preds).flatten(), 0, None)

        mse = np.mean((preds - y_true) ** 2)
        nonzero = y_true != 0
        mpe = np.mean(((preds - y_true) / y_true)[nonzero]) * 100

        return mse, mpe

mse, mpe = evaluate_mse_mpe(model, X_val, y_val, y_scaler)
print("Final Validation Evaluation:")
print(f"MSE: {mse:.4f}")
print(f"MPE: {mpe:.2f}%")

# Plot training and validation loss curves
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label="Train Loss", marker="o")
plt.plot(val_losses, label="Validation Loss", marker="s")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.grid()
plt.savefig("checkpoints/loss_curve.png")

print("Training complete. Best model saved to 'checkpoints/best_model.pth'")

import seaborn as sns

def plot_predictions(model, X_tensor, y_tensor, scaler, save_path="checkpoints/pred_vs_actual.png"):
    model.eval()
    with torch.no_grad():
        preds = model(X_tensor)
        y_true = y_tensor.cpu().numpy().reshape(-1, 1)
        preds = preds.cpu().numpy().reshape(-1, 1)

        y_true = scaler.inverse_transform(y_true).flatten()
        preds = scaler.inverse_transform(preds).flatten()

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=preds, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')  # 参考线
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual")
    plt.grid(True)
    plt.savefig(save_path)
    print(f"Prediction plot saved to {save_path}")

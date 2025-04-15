import os


base_dir = os.getcwd()
checkpoint_dir = os.path.join(base_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
debug_log_path = os.path.join(checkpoint_dir, "debug.log")


try:
    with open(debug_log_path, "a") as f:
        f.write("[STEP 0] train.py loaded\n")
except Exception as e:
    print(f"Logging failed: {e}")

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
import seaborn as sns
import argparse


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.datasets.data_loader import load_data
from models.solar_time_model_tiny import TinyMLP
from models.solar_tree_model import SklearnTreeWrapper
from models.solar_lr_model import SklearnLRWrapper

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            total_loss += loss.item()
    return total_loss / len(dataloader)


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


def plot_predictions(preds, y_true, name="mlp", out_dir=None):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true.flatten(), y=preds.flatten(), alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{name.upper()} Prediction vs Actual")
    plt.grid(True)
    path = os.path.join(checkpoint_dir, f"pred_vs_actual_{name}.png")
    plt.savefig(path)
    print(f"[{name}] Prediction plot saved to {path}")

def train_sklearn_model(model_cls, X_train, y_train, X_val, y_val, y_scaler, name):
    print(f"[{name}] Training started")
    model = model_cls()
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    y_true = y_scaler.inverse_transform(y_val.numpy().reshape(-1, 1)).flatten()
    preds = y_scaler.inverse_transform(preds).flatten()

    mse = np.mean((preds - y_true) ** 2)
    mpe = np.mean(((preds - y_true) / y_true)[y_true != 0]) * 100

    with open(os.path.join(checkpoint_dir, f"metrics_{name}.txt"), "w") as f:
        f.write(f"MSE: {mse:.4f}\nMPE: {mpe:.2f}%\n")

    plot_predictions(preds, y_true, name)
    print(f"[{name}] Done. MSE={mse:.4f}, MPE={mpe:.2f}%")


def main():
    with open(debug_log_path, "a") as f:
        f.write("[train.py] main() started\n")

    print("[train.py] Script has started.")

    data_path = os.path.join("data", "raw_data", "uploaded_data.xlsx")
    X, y, X_scaler, y_scaler = load_data(data_path)

    with open(debug_log_path, "a") as f:
        f.write("[STEP 2] data loaded\n")

    
    print(f"NaNs in X: {torch.isnan(X).sum().item()}")
    print(f"NaNs in y: {torch.isnan(y).sum().item()}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = TinyMLP(input_dim=X.shape[1])

    with open(debug_log_path, "a") as f:
        f.write("[STEP 3] model initialized\n")

    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)

    train_losses = []
    val_losses = []
    num_epochs = 300
    best_val_loss = float("inf")
    early_stopping_patience = 88
    no_improve_count = 0

    os.makedirs("checkpoints", exist_ok=True)

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
        val_loss = evaluate(model, val_loader, criterion)

        if np.isnan(train_loss) or np.isinf(train_loss):
            print(f"Warning: NaN or Inf detected in training loss at epoch {epoch}. Stopping training.")
            break

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break

        scheduler.step(val_loss)

    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "final_model.pth"))
    joblib.dump(y_scaler, os.path.join(checkpoint_dir, "y_scaler.pkl"))

    mse, mpe = evaluate_mse_mpe(model, X_val, y_val, y_scaler)
    print("Final Validation Evaluation:")
    print(f"MSE: {mse:.4f}")
    print(f"MPE: {mpe:.2f}%")
    with open(os.path.join(checkpoint_dir, "metrics.txt"), "w") as f:
        f.write(f"MSE: {mse:.4f}\nMPE: {mpe:.2f}%\n")

    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(val_losses, label="Validation Loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(checkpoint_dir, "loss_curve.png"))

    print("Training complete. Best model saved to 'checkpoints/best_model.pth'")
    with torch.no_grad():
        preds = model(X_val)
        y_true = y_scaler.inverse_transform(y_val.numpy())
        pred_vals = y_scaler.inverse_transform(preds.numpy())
    plot_predictions(pred_vals, y_true, name="mlp", out_dir=checkpoint_dir)

    
    train_sklearn_model(SklearnTreeWrapper, X_train, y_train, X_val, y_val, y_scaler, name="tree")
    train_sklearn_model(SklearnLRWrapper, X_train, y_train, X_val, y_val, y_scaler, name="lr")
    
    import sys
    sys.exit(0)


if __name__ == "__main__":
    main()

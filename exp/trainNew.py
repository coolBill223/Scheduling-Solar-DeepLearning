import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "checkpoints"))
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from models.solar_time_model_tiny import TinyMLP
from models.solar_tree_model import SklearnTreeWrapper
from models.solar_lr_model import SklearnLRWrapper
from models.solar_ga_model import train_ga_model
from data.datasets.data_loader_new import load_data
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return rmse, mae, r2
def plot_predictions(preds, y_true, name="mlp", out_dir="checkpoints"):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true.flatten(), y=preds.flatten(), alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{name.upper()} Prediction vs Actual")
    plt.grid(True)
    path = os.path.join(out_dir, f"pred_vs_actual_{name}.png")
    plt.savefig(path)
    print(f"[{name}] Prediction plot saved to {path}")

def evaluate_mse_mpe(model, X_tensor, y_tensor, scaler):
    model.eval()
    with torch.no_grad():
        preds = model(X_tensor)
        y_true = y_tensor.cpu().numpy().reshape(-1, 1)
        preds = preds.cpu().numpy().reshape(-1, 1)

        y_true = np.clip(np.expm1(scaler.inverse_transform(y_true).flatten()), 0, None)
        preds  = np.clip(np.expm1(scaler.inverse_transform(preds).flatten()), 0, None)


        mse = np.mean((preds - y_true) ** 2)
        nonzero = y_true != 0
        mpe = np.mean(((preds - y_true) / y_true)[nonzero]) * 100

        return mse, mpe


def train_sklearn_model(model_cls, X_train, y_train, X_val, y_val, y_scaler, name):
    print(f"[{name}] Training started")
    model = model_cls()
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    y_true = np.clip(np.expm1(y_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()), 0, None)
    preds  = np.clip(np.expm1(y_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()), 0, None)


    rmse, mae, r2 = evaluate_regression_metrics(y_true, preds)

    print(f"[{name} Metrics] RMSE: {rmse:.2f} min")
    print(f"[{name} Metrics] MAE: {mae:.2f} min")
    print(f"[{name} Metrics] R2: {r2:.4f}")

    # Save metrics
    with open(os.path.join(CHECKPOINT_DIR, f"metrics_{name}.txt"), "w") as f:
        f.write(f"RMSE: {rmse:.2f} mins\n")
        f.write(f"MAE: {mae:.2f} mins\n")
        f.write(f"R2: {r2:.4f}\n")


    # Bias correction
    bias = np.mean(y_true - preds)
    preds_biased = preds + bias
    print(f"[{name} Bias] Bias = {bias:.2f} minutes")

    # Save bias
    with open(os.path.join(CHECKPOINT_DIR, f"{name}_bias.txt"), "w") as f:
        f.write(str(bias))

    # Save plot
    plot_predictions(preds_biased, y_true, name)

    # Confirm path
    img_path = os.path.join(CHECKPOINT_DIR, f"pred_vs_actual_{name}.png")
    if os.path.exists(img_path):
        print(f"[{name}] Prediction plot saved at {img_path}")
    else:
        print(f"[{name}] WARNING: Prediction plot not found.")
    
    print(f"[{name}] Done. RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}")


    
    
def train_stacked_model():
    # Load data
    
    
    excel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw_data", "Data.xlsx")
    excel_path = os.path.normpath(excel_path)
    X_train, X_val, y_train, y_val, y_scaler, X_scaler = load_data(excel_path)

    
    
    joblib.dump(X_scaler, os.path.join(CHECKPOINT_DIR, "X_scaler.pkl"))
    joblib.dump(y_scaler, os.path.join(CHECKPOINT_DIR, "y_scaler.pkl"))
    
    # Train Tree
    tree = SklearnTreeWrapper()
    tree.fit(X_train, y_train)
    joblib.dump(tree, os.path.join(CHECKPOINT_DIR, "tree_model.pkl")) 
    tree_train_pred = tree.predict(X_train).reshape(-1, 1)
    tree_val_pred = tree.predict(X_val).reshape(-1, 1)


    # Train LR
    lr = SklearnLRWrapper()
    lr.fit(X_train, y_train)
    joblib.dump(lr, os.path.join(CHECKPOINT_DIR, "lr_model.pkl"))
    lr_train_pred = lr.predict(X_train).reshape(-1, 1)
    lr_val_pred = lr.predict(X_val).reshape(-1, 1)

    # Stack original features + tree + lr
    X_train_stacked = np.hstack([X_train, tree_train_pred, lr_train_pred])
    X_val_stacked = np.hstack([X_val, tree_val_pred, lr_val_pred])

    # Train MLP
    model = TinyMLP(input_dim=X_train_stacked.shape[1])
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_tensor = torch.tensor(X_train_stacked, dtype=torch.float32)
    y_tensor = y_train.clone().detach().view(-1, 1)



    X_val_tensor = torch.tensor(X_val_stacked, dtype=torch.float32)
    y_val_tensor = y_val.clone().detach().view(-1, 1)

    train_losses = []
    val_losses = []
    best_loss = float("inf")
    patience = 40
    wait = 0
    

    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        preds = model(X_tensor)
        loss = loss_fn(preds, y_tensor)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_tensor)
            val_loss = loss_fn(val_preds, y_val_tensor).item()
        val_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            wait = 0
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Evaluation
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pth")))
    with torch.no_grad():
        pred_vals = model(X_val_tensor)

    
    y_true = np.clip(np.expm1(y_scaler.inverse_transform(y_val_tensor.numpy()).flatten()), 0, None)
    preds  = np.clip(np.expm1(y_scaler.inverse_transform(pred_vals.numpy()).flatten()), 0, None)

    
    rmse, mae, r2 = evaluate_regression_metrics(y_true, preds)

    
    print(f"[MLP Metrics] RMSE: {rmse:.2f} min")
    print(f"[MLP Metrics] MAE: {mae:.2f} min")
    print(f"[MLP Metrics] R2: {r2:.4f}")

    
    with open(os.path.join(CHECKPOINT_DIR, "metrics_mlp.txt"), "w") as f:
        f.write(f"RMSE: {rmse:.2f} mins\n")
        f.write(f"MAE: {mae:.2f} mins\n")
        f.write(f"R2: {r2:.4f}\n")

    with open(os.path.join(CHECKPOINT_DIR, "metrics.txt"), "w") as f:
        f.write(f"RMSE: {rmse:.2f} mins\n")
        f.write(f"MAE: {mae:.2f} mins\n")
        f.write(f"R2: {r2:.4f}\n")

    
    bias = np.mean(y_true - preds)
    preds_biased = preds + bias

    
    rmse_biased, mae_biased, r2_biased = evaluate_regression_metrics(y_true, preds_biased)
    print(f"[MLP Metrics with Bias] RMSE: {rmse_biased:.2f} min")
    print(f"[MLP Metrics with Bias] MAE: {mae_biased:.2f} min")
    print(f"[MLP Metrics with Bias] R2: {r2_biased:.4f}")

    
    with open(os.path.join(CHECKPOINT_DIR, "mlp_bias.txt"), "w") as f:
        f.write(str(bias))

    
    plot_predictions(preds_biased, y_true, name="mlp", out_dir=CHECKPOINT_DIR)
    
    import shutil
    shutil.copyfile(
        os.path.join(CHECKPOINT_DIR, "pred_vs_actual_mlp.png"),
        os.path.join(CHECKPOINT_DIR, "pred_vs_actual.png")
    )

    
    with open(os.path.join(CHECKPOINT_DIR, "mlp_input_dim.txt"), "w") as f:
        f.write(str(X_train_stacked.shape[1]))

    # Generate prediction plots and metrics for Tree and LR
    train_sklearn_model(SklearnTreeWrapper, X_train, y_train, X_val, y_val, y_scaler, name="tree")
    train_sklearn_model(SklearnLRWrapper, X_train, y_train, X_val, y_val, y_scaler, name="lr")

    print("Training complete.")

def train_all_models():
    train_stacked_model() 
    print("[GA] Training started")
    train_ga_model(output_dir=CHECKPOINT_DIR)
    print("[GA] Training complete")

    
if __name__ == "__main__":
    train_all_models()

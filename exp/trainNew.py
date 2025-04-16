
import os
import numpy as np
import torch
import torch.nn as nn
from train import evaluate_mse_mpe, plot_predictions
from models.solar_time_model_tiny import TinyMLP
from models.solar_tree_model import SklearnTreeWrapper
from models.solar_lr_model import SklearnLRWrapper
from data.datasets.data_loader_new import load_data
import joblib

def train_stacked_model():
    # Load data
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    X_train, X_val, y_train, y_val, y_scaler = load_data()

    # Train Tree
    tree = SklearnTreeWrapper()
    tree.fit(X_train, y_train)
    joblib.dump(tree, os.path.join(checkpoint_dir, "tree_model.pkl")) 
    tree_train_pred = tree.predict(X_train).reshape(-1, 1)
    tree_val_pred = tree.predict(X_val).reshape(-1, 1)

    # Train LR
    lr = SklearnLRWrapper()
    lr.fit(X_train, y_train)
    joblib.dump(lr, os.path.join(checkpoint_dir, "lr_model.pkl"))
    lr_train_pred = lr.predict(X_train).reshape(-1, 1)
    lr_val_pred = lr.predict(X_val).reshape(-1, 1)

    # Stack original features + tree + lr
    X_train_stacked = np.hstack([X_train, tree_train_pred, lr_train_pred])
    X_val_stacked = np.hstack([X_val, tree_val_pred, lr_val_pred])

    # Train MLP
    model = TinyMLP(input_dim=X_train_stacked.shape[1])
    loss_fn = nn.MSELoss()
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
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Evaluation
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_model.pth")))
    mse, mpe = evaluate_mse_mpe(model, X_val_tensor, y_val_tensor, y_scaler)
    print(f"Final Validation Evaluation (before bias):\nMSE: {mse:.4f}\nMPE: {mpe:.2f}%")

    with torch.no_grad():
        pred_vals = model(X_val_tensor)
        y_true = y_scaler.inverse_transform(y_val_tensor.numpy()).flatten()
        preds = y_scaler.inverse_transform(pred_vals.numpy()).flatten()
        bias = np.mean(y_true - preds)
        preds_biased = preds + bias
        mse_biased = np.mean((preds_biased - y_true) ** 2)
        mpe_biased = np.mean(np.abs((preds_biased - y_true) / y_true)) * 100

    print(f"Final Validation Evaluation (with bias):\nMSE: {mse_biased:.4f}\nMPE: {mpe_biased:.2f}%")
    print(f"[MLP Bias] Bias applied: {bias:.2f} minutes")

    plot_predictions(preds_biased, y_true, name="mlp", out_dir=checkpoint_dir)
    with open(os.path.join(checkpoint_dir, "mlp_bias.txt"), "w") as f:
        f.write(str(bias))
    with open(os.path.join(checkpoint_dir, "metrics.txt"), "w") as f:
        f.write(f"MSE: {mse_biased:.4f}\nMPE: {mpe_biased:.2f}%\n")

    print("Training complete.")

    
    
if __name__ == "__main__":
    train_stacked_model()

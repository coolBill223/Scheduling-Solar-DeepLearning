import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.datasets.data_loader import load_data
from models.solar_time_model import MLPModel


def evaluate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def evaluate_mse_mpe(model, X, y, y_scaler):
    model.eval()
    with torch.no_grad():
        predictions = model(X)
        y = y.cpu().numpy().reshape(-1, 1)
        predictions = predictions.cpu().numpy().reshape(-1, 1)
        y = np.clip(y_scaler.inverse_transform(y).flatten(), 0, None)
        predictions = np.clip(y_scaler.inverse_transform(predictions).flatten(), 0, None)
        mse = np.mean((predictions - y) ** 2)
        nonzero_mask = y != 0
        mpe = np.mean(((predictions - y) / y)[nonzero_mask]) * 100
    return mse, mpe

def train_and_tune(model, train_loader, val_loader, optimizer, criterion, scheduler, num_epochs=100, early_stopping_patience=10):
    best_val_loss = float("inf")
    no_improve_count = 0
    train_losses, val_losses = [], []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        train_loss = epoch_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count >= early_stopping_patience:
            print("Early stopping triggered.")
            break
        
        scheduler.step(val_loss)
    
    return best_val_loss, train_losses, val_losses

def main():
    os.makedirs("checkpoints", exist_ok=True)
    X, y, X_scaler, y_scaler = load_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)
    
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    best_overall_loss = float("inf")
    best_hyperparams = {}
    best_train_losses, best_val_losses = [], []
    
    for lr in [1e-3, 5e-4, 1e-4]:
        for weight_decay in [1e-3, 1e-4, 1e-5]:
            print(f"Training with lr={lr}, weight_decay={weight_decay}")
            
            model = MLPModel(input_dim=X.shape[1])
            criterion = nn.MSELoss()
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)
            
            val_loss, train_losses, val_losses = train_and_tune(model, train_loader, val_loader, optimizer, criterion, scheduler)
            
            if val_loss < best_overall_loss:
                best_overall_loss = val_loss
                best_hyperparams = {"lr": lr, "weight_decay": weight_decay}
                best_train_losses, best_val_losses = train_losses, val_losses
    
    print(f"Best Hyperparameters: {best_hyperparams}, Best Validation Loss: {best_overall_loss:.4f}")
    
    model.load_state_dict(torch.load("checkpoints/best_model.pth"))
    mse, mpe = evaluate_mse_mpe(model, X_val, y_val, y_scaler)
    print(f"Final Evaluation: MSE = {mse:.4f}, MPE = {mpe:.2f}%")
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(best_train_losses)), best_train_losses, label="Train Loss", marker="o", linestyle="-")
    plt.plot(range(len(best_val_losses)), best_val_losses, label="Validation Loss", marker="s", linestyle="-")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig("checkpoints/loss_curve.png")
    plt.show()
    
    joblib.dump(y_scaler, "checkpoints/y_scaler.pkl")
    print("Training and hyperparameter tuning complete.")
    
if __name__ == "__main__":
    main()

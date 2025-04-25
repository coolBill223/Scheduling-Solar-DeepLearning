import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data.datasets.data_loader_old import load_data
from models.genetic_algorithm import genetic_algorithm

from sklearn.linear_model import LinearRegression

def generate_nonlinear_features(X):
    # Add nonlinear features: square, log, and square root
    X_squared = X ** 2
    X_log = np.log1p(np.abs(X))
    X_sqrt = np.sqrt(np.abs(X))
    return np.hstack([X, X_squared, X_log, X_sqrt])
USER_DIR = os.path.expanduser('~')
CHECKPOINT_DIR = os.path.join(USER_DIR, ".my_software_checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
def train_ga_model(output_dir=CHECKPOINT_DIR):
    for f in ["ga_scaler.pkl", "ga_bias.txt", "best_weights_ga.csv", "metrics_ga.txt", "pred_vs_actual_ga.png"]:
        path = os.path.join(output_dir, f)
        if os.path.exists(path):
            os.remove(path)
            print(f"[GA] Removed old file: {path}")
    
    X, y = load_data()
    if X is None or y is None:
        print("Data loading failed. Exiting.")
        exit()
    print(f"[GA] Raw X shape: {X.shape}, y shape: {y.shape}")
    X = np.hstack([X, np.ones((X.shape[0], 1))])  # Add bias term
    print(f"[GA] After adding bias: X shape = {X.shape}")
    # Add nonlinear features
    X = generate_nonlinear_features(X)
    print(f"[GA] After nonlinear expansion: X shape = {X.shape}")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_all_scaled = scaler_X.transform(X)  # Normalize all data

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(scaler_X, os.path.join(output_dir, "ga_scaler.pkl"))
    print(f"[GA] Saved scaler to {os.path.join(output_dir, 'ga_scaler.pkl')}")
    best_weights, best_fitness_per_gen = genetic_algorithm(
        X_train_scaled, y_train,
        population_size=200,
        generations=2500,
        elite_size=2,
        save_path=os.path.join(output_dir, "best_weights_ga.csv")
    )
    print(f"[GA] Saved best weights to {os.path.join(output_dir, 'best_weights_ga.csv')}")
    y_pred_all_ga = X_all_scaled @ best_weights 

    
    bias_constant = np.mean(y - y_pred_all_ga)
    
    with open(os.path.join(output_dir, "ga_bias.txt"), "w") as f:
        f.write(str(bias_constant))


    y_val_pred = X_val_scaled @ best_weights + bias_constant
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    mae = mean_absolute_error(y_val, y_val_pred)
    r2 = r2_score(y_val, y_val_pred)

    with open(os.path.join(output_dir, "metrics_ga.txt"), "w") as f:
        f.write(f"RMSE: {rmse:.2f} mins\n")
        f.write(f"MAE: {mae:.2f} mins\n")
        f.write(f"R2: {r2:.4f}\n")

    plt.figure()
    plt.scatter(y_val, y_val_pred, alpha=0.6)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
    plt.xlabel("Actual Duration")
    plt.ylabel("Predicted Duration")
    plt.title("GA Model Prediction vs Actual")
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "pred_vs_actual_ga.png")
    plt.savefig(plot_path)
    print(f"[ga] Prediction plot saved to {plot_path}")

    print(f"[ga] Done. RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}")
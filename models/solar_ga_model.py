import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from data.datasets.data_loader_new import load_data
from models.genetic_algorithm import genetic_algorithm

def generate_nonlinear_features(X):
    X_squared = X ** 2
    X_log = np.log1p(np.abs(X))
    X_sqrt = np.sqrt(np.abs(X))
    return np.hstack([X, X_squared, X_log, X_sqrt])

def train_ga_model(output_dir="checkpoints"):
    os.makedirs(output_dir, exist_ok=True)
    excel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw_data", "Data.xlsx")
    excel_path = os.path.normpath(excel_path)
    X_train, X_val, y_train, y_val, y_scaler, X_scaler = load_data(excel_path)

    X = np.vstack([X_train, X_val])
    y = np.concatenate([y_train, y_val])
    
    if X is None or y is None:
        print("Data loading failed. Exiting.")
        return

    X = np.hstack([X, np.ones((X.shape[0], 1))])  # Add bias term
    X = generate_nonlinear_features(X)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_all_scaled = scaler_X.transform(X)

    joblib.dump(scaler_X, os.path.join(output_dir, "ga_scaler.pkl"))

    best_weights, best_fitness_per_gen = genetic_algorithm(
        X_train_scaled, y_train,
        population_size=200,
        generations=20000,
        elite_size=2,
        save_path=os.path.join(output_dir, "best_weights_ga.csv")
    )

    y_pred_ga = X_val_scaled @ best_weights
    mse_ga = mean_squared_error(y_val, y_pred_ga)
    rmse_ga = np.sqrt(mse_ga)
    print(f"\n[GA-based Linear Model] Test MSE={mse_ga:.4f}, RMSE={rmse_ga:.4f}")

    # GA predictions on all data
    y_pred_all_ga = X_all_scaled @ best_weights

    # Calibrate GA prediction (fit bias to match y)
    bias_adjust = LinearRegression()
    bias_adjust.fit(y_pred_all_ga.reshape(-1, 1), y)
    y_pred_adjusted = bias_adjust.predict(y_pred_all_ga.reshape(-1, 1))

    # Plot: Bias-Corrected GA vs Actual + Linear Regression
    plt.figure()
    plt.scatter(y, y_pred_adjusted, alpha=0.6, s=20, label="GA + Bias Adjust")
    min_val = min(np.min(y), np.min(y_pred_adjusted))
    max_val = max(np.max(y), np.max(y_pred_adjusted))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="y = x")
    plt.xlabel("Actual Duration (minutes)")
    plt.ylabel("Predicted Duration (minutes)")
    plt.title("Bias-Corrected GA vs. Actual")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pred_vs_actual_ga.png"))

    # Output RMSE and mean bias after bias adjustment
    mse_adjusted = mean_squared_error(y, y_pred_adjusted)
    rmse_adjusted = np.sqrt(mse_adjusted)
    mean_bias = np.mean(y_pred_adjusted - y)
    max_bias = np.max(abs(y_pred_adjusted - y))

    bias_constant = np.mean(y - y_pred_all_ga)
    y_pred_const_adjusted = y_pred_all_ga + bias_constant

    rmse_const = np.sqrt(mean_squared_error(y, y_pred_const_adjusted))
    mean_bias_const = np.mean(y_pred_const_adjusted - y)

    with open(os.path.join(output_dir, "metrics_ga.txt"), "w") as f:
        f.write(f"[GA + Bias Adjust] RMSE: {rmse_adjusted:.4f}\n")
        f.write(f"[GA + Bias Adjust] Mean Bias: {mean_bias:.4f} minutes\n")
        f.write(f"[GA + Bias Adjust] Max Bias: {max_bias:.4f} minutes\n")
        f.write(f"[GA + Constant Bias Adjust] RMSE: {rmse_const:.4f}\n")
        f.write(f"[GA + Constant Bias Adjust] Mean Bias: {mean_bias_const:.4f}\n")
        f.write(f"Bias Term Added: {bias_constant:.4f}\n")
    with open(os.path.join(output_dir, "ga_bias.txt"), "w") as f:
      f.write(str(bias_constant))

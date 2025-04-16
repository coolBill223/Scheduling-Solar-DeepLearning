import os
import torch
import joblib
import numpy as np
from models.solar_time_model_tiny import TinyMLP
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def predict_with_model(model_name: str, feature_vector: list) -> float:
    """
    Predict installation time given a model and feature list.
    Parameters:
        model_name (str): one of 'mlp', 'tree', 'lr'
        feature_vector (list): length = 24~27 depending on input features
    Returns:
        float: predicted installation time (in hours)
    """
    print("[DEBUG] Running updated predict_engine with stacking")
    
    # Convert input to NumPy array
    X = np.array(feature_vector).reshape(1, -1)

    # Load scaler to inverse prediction
    y_scaler = joblib.load("checkpoints/y_scaler.pkl")

    if model_name == "mlp":
        # Load stacking base models
        tree_model = joblib.load("checkpoints/tree_model.pkl")
        lr_model = joblib.load("checkpoints/lr_model.pkl")

        # Predict tree & lr outputs
        X_torch = torch.from_numpy(X.astype(np.float32))
        tree_pred = tree_model.predict(X_torch).reshape(-1, 1)
        lr_pred = lr_model.predict(X_torch).reshape(-1, 1)

        # Concatenate original + tree + lr predictions
        X_stacked = np.hstack([X, tree_pred, lr_pred])
        print("[DEBUG] X_stacked type:", type(X_stacked))
        print("[DEBUG] X_stacked shape:", X_stacked.shape)


        # Load and predict with MLP
        model = TinyMLP(input_dim=X_stacked.shape[1])
        model.load_state_dict(torch.load("checkpoints/best_model.pth"))
        model.eval()
        with torch.no_grad():
            if isinstance(X_stacked, np.ndarray):
                X_tensor = torch.from_numpy(X_stacked.astype(np.float32))
            else:
                X_tensor = torch.tensor(X_stacked, dtype=torch.float32)
            
            y_pred_tensor = model(X_tensor).detach().cpu()
            y_pred = y_pred_tensor.numpy().flatten()


    elif model_name == "tree":
        model = joblib.load("checkpoints/tree_model.pkl")
        y_pred = model.predict(X)

    elif model_name == "lr":
        model = joblib.load("checkpoints/lr_model.pkl")
        y_pred = model.predict(X)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Inverse transform prediction
    y_pred = np.array(y_pred).reshape(-1, 1)
    y_pred_inv = y_scaler.inverse_transform(y_pred).flatten()
    y_pred_minutes = y_pred_inv[0] ** 2
    
    # Load bias (optional)
    bias_path = os.path.join("checkpoints", f"{model_name}_bias.txt")
    if os.path.exists(bias_path):
        with open(bias_path) as f:
            bias = float(f.read().strip())
            y_pred_minutes += bias
    return float(y_pred_minutes)

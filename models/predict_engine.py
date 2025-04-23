import os
import torch
import joblib
import numpy as np
from models.solar_time_model_tiny import TinyMLP
import sys
import os
import random
import json
BASE_DIR = getattr(sys, '_MEIPASS', os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

with open(os.path.join(BASE_DIR, "checkpoints", "category_mappings.json")) as f:
    category_mappings = json.load(f)

def map_category_value(col_name, val):
    try:
        return category_mappings[col_name].index(val) + 1
    except:
        return 0  # fallback if not found

def process_feature_vector(raw_vector: list, feature_names: list) -> list:
    categorical_fields = [
        "Inverter Manufacturer", "Array Type", "Squirrel Screen", "Consumption Monitoring",
        "Truss / Rafter", "Reinforcements", "Interconnection Type", "Roof Type",
        "Attachment Type", "Portrait / Landscape", "Install Season"
    ]
    
    processed = []
    for i, val in enumerate(raw_vector):
        col = feature_names[i]
        if col in categorical_fields:
            processed.append(map_category_value(col, val))
        else:
            try:
                processed.append(float(val))
            except:
                processed.append(0.0)
    return processed

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
    print("[DEBUG] feature_vector =", feature_vector)

    X_scaler = joblib.load(os.path.join(BASE_DIR, "checkpoints", "X_scaler.pkl")) 

    import json
    print("Scaler expects:", X_scaler.n_features_in_)          # 24
    print("Feature names :", X_scaler.feature_names_in_)
    
    feature_names = list(X_scaler.feature_names_in_)
    input_dict = dict(zip(feature_names, feature_vector))
    feature_vector_raw = [input_dict.get(k) for k in feature_names]
    feature_vector = process_feature_vector(feature_vector_raw, feature_names)

    print("[DEBUG] MAPPED feature_vector =", feature_vector)
    
    incoming = np.array(feature_vector).reshape(1, -1)
    print("You provided :", incoming.shape[1])
    
    # Convert input to NumPy array
    X_raw = np.array(feature_vector).reshape(1, -1)
    X_std = X_scaler.transform(incoming)
    # Load scaler to inverse prediction
    y_scaler = joblib.load(os.path.join(BASE_DIR, "checkpoints", "y_scaler.pkl"))

    if model_name == "mlp":
        # Load stacking base models
        tree_model = joblib.load(os.path.join(BASE_DIR, "checkpoints", "tree_model.pkl"))
        lr_model = joblib.load(os.path.join(BASE_DIR, "checkpoints", "lr_model.pkl"))

        # Predict tree & lr outputs
        X_torch = torch.from_numpy(X_std.astype(np.float32))
        tree_pred = tree_model.predict(X_torch).reshape(-1, 1)
        lr_pred = lr_model.predict(X_torch).reshape(-1, 1)

        # Concatenate original + tree + lr predictions
        X_stacked = np.hstack([X_std, tree_pred, lr_pred])
        print("[DEBUG] X_stacked type:", type(X_stacked))
        print("[DEBUG] X_stacked shape:", X_stacked.shape)


        # Load and predict with MLP
        with open(os.path.join(BASE_DIR, "checkpoints", "mlp_input_dim.txt")) as f:
            input_dim = int(f.read().strip())

        model = TinyMLP(input_dim=input_dim)
        state_dict = torch.load(os.path.join(BASE_DIR, "checkpoints", "best_model.pth"), map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval() 

        with torch.no_grad():
            if isinstance(X_stacked, np.ndarray):
                X_tensor = torch.from_numpy(X_stacked.astype(np.float32))
            else:
                X_tensor = torch.tensor(X_stacked, dtype=torch.float32)
            
            y_pred_tensor = model(X_tensor).detach().cpu()
            y_pred = y_pred_tensor.numpy().flatten()


    elif model_name == "tree":
        model = joblib.load(os.path.join(BASE_DIR, "checkpoints", "tree_model.pkl"))
        X_tensor = torch.from_numpy(X_std.astype(np.float32))
        y_pred = model.predict(X_tensor)

    elif model_name == "lr":
        model = joblib.load(os.path.join(BASE_DIR, "checkpoints", "lr_model.pkl"))
        X_tensor = torch.from_numpy(X_std.astype(np.float32))
        y_pred = model.predict(X_tensor)

    elif model_name == "ga":
        scaler = joblib.load(os.path.join(BASE_DIR, "checkpoints", "ga_scaler.pkl"))
        weights = np.loadtxt(os.path.join(BASE_DIR, "checkpoints", "best_weights_ga.csv"), delimiter=',', skiprows=1, usecols=1)
        
        incoming = np.array(feature_vector).reshape(1, -1)
        X_std = scaler.transform(incoming)

        y_pred = X_std @ weights.reshape(-1, 1)

        bias_path = os.path.join(BASE_DIR, "checkpoints", "ga_bias.txt")
        if os.path.exists(bias_path):
            with open(bias_path) as f:
                bias = float(f.read().strip())
                y_pred += bias

        return float(np.clip(y_pred.flatten()[0], 0, None))

    
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Inverse transform prediction
    y_pred = np.array(y_pred).reshape(-1, 1)
    y_pred_inv = y_scaler.inverse_transform(y_pred).flatten()
    y_pred_minutes = np.clip(np.expm1(y_pred_inv[0]), 0, None)


    
    # Load bias (optional)
    bias_path = os.path.join(BASE_DIR, "checkpoints", f"{model_name}_bias.txt")
    if os.path.exists(bias_path):
        with open(bias_path) as f:
            bias = float(f.read().strip())
            y_pred_minutes += bias
    
    print("y_pred_raw   :", y_pred)           
    print("y_inv_scaled:", y_pred_inv[0])     
    print("bias        :", bias if os.path.exists(bias_path) else 0)
    
    return float(y_pred_minutes)

    

# data_loader.py

import pandas as pd
import numpy as np
import os
import datetime
import re
import sys

USER_DIR = os.path.expanduser('~')
CHECKPOINT_DIR = os.path.join(USER_DIR, ".my_software_checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
def clean_angle(value):
    """
    Convert angle-related values (e.g., '25°' or '30/40') into numeric form.
    If there's a slash '/', take the average of split values.
    If parsing fails, returns NaN.
    """
    if pd.isna(value):
        return np.nan
    value = str(value).replace("°", "").strip()
    if "/" in value:  # handle cases like '30/40'
        try:
            values = list(map(float, value.split("/")))
            return sum(values) / len(values)
        except:
            return np.nan
    try:
        return float(value)
    except:
        return np.nan

def convert_dhm_to_minutes_strict(value):
    """
    Convert a datetime-like value (including strings like 'D:H:M') into total minutes.
    If parsing fails, returns None.
    """
    if pd.isna(value) or value in ["", "nan", "None"]:
        return None

    import datetime
    try:
        if isinstance(value, datetime.datetime):
            # If it's a datetime object, interpret as day + hour*60 + minute
            return 1440 + value.hour * 60 + value.minute + value.second / 60
        elif isinstance(value, datetime.time):
            return value.hour * 60 + value.minute + value.second / 60

        # If it's a string, possibly 'D:H:M' or 'D:H'
        parts = list(map(int, str(value).strip().split(":")))
        if len(parts) == 3:
            d, h, m = parts
        elif len(parts) == 2:
            d, h = parts
            m = 0
        elif len(parts) == 1:
            d = 0
            h = parts[0]
            m = 0
        else:
            return None
        return d * 1440 + h * 60 + m

    except:
        return None

def load_data(file_path=None):
    """
    Load and preprocess the Excel dataset, returning feature matrix X and target y (in minutes).
    Ensures all columns in X are numeric to avoid errors during scaling.
    """
    if file_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, "..", "raw_data", "Data.xlsx")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Excel file not found: {file_path}")
    # Read a preview (10 rows) to find the header row
    preview = pd.read_excel(file_path, engine="openpyxl", nrows=10, header=None)
    header_row = preview.apply(lambda row: row.notna().sum(), axis=1).idxmax()
    df = pd.read_excel(file_path, engine="openpyxl", header=header_row)

    # Drop empty rows/columns
    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)

    df.columns = df.columns.str.strip().str.replace("\n", " ", regex=False)
    unnamed_cols = [col for col in df.columns if col.lower().startswith("unnamed")]
    df.drop(columns=unnamed_cols, inplace=True)

    if df.empty:
        print("The dataset is empty. Please check the Excel file content.")
        return None, None

    # Define the target column
    target = "Total Direct Time for Project for Hourly Employees (Including Drive Time)"
    if target not in df.columns:
        print(f"Target column '{target}' not found.")
        return None, None

    # Convert target column to minutes
    if pd.api.types.is_timedelta64_dtype(df[target]):
        df[target] = df[target].dt.total_seconds() / 60
    else:
        df[target] = df[target].apply(convert_dhm_to_minutes_strict)
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df[target] = df[target].fillna(df[target].mean())

    # Example: if there's a "Drive Time" column we want to keep or remove, handle it
    # if "Drive Time" in df.columns:
    #     if pd.api.types.is_timedelta64_dtype(df["Drive Time"]):
    #         df["Drive Time"] = df["Drive Time"].dt.total_seconds() / 60
    #     else:
    #         df["Drive Time"] = df["Drive Time"].apply(convert_dhm_to_minutes_strict)
    #     df["Drive Time"] = pd.to_numeric(df["Drive Time"], errors="coerce").fillna(0)

    # Process angles if 'Tilt' or 'Azimuth' columns exist
    if "Tilt" in df.columns:
        df["Tilt"] = df["Tilt"].apply(clean_angle)
    if "Azimuth" in df.columns:
        df["Azimuth"] = df["Azimuth"].apply(clean_angle)

    # Convert yes/no to 1/0
    boolean_cols = [
        col for col in df.columns
        if df[col].dropna().astype(str).apply(lambda x: x.lower() in ["yes", "no"]).all()
    ]
    for col in boolean_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0, "yes": 1, "no": 0})

    # Factorize other categorical columns
    categorical_cols = df.select_dtypes(exclude=["number", "timedelta"]).columns.tolist()
    # Exclude the target itself and booleans from factorization
    categorical_cols = [c for c in categorical_cols if c not in boolean_cols + [target]]

    for col in categorical_cols:
        df[col] = df[col].astype(str).factorize()[0] + 1

    # Drop any rows where the target is NaN (if any remain)
    df = df.dropna(subset=[target])

    # --- Convert any Timedelta columns in the DataFrame to numeric minutes ---
    for col in df.columns:
        if pd.api.types.is_timedelta64_dtype(df[col]):
            # Convert Timedelta to total minutes
            df[col] = df[col].dt.total_seconds() / 60

    # Exclude columns not needed
    exclude_columns = [
        "Project ID", "Notes", "Total # of Days on Site", "Estimated # of Salaried Employees on Site",
        "Estimated Salary Hours", "Estimated Total Direct Time", "Estimated Total # of People on Site",
        "Drive Time"
    ]
    features = [col for col in df.columns if col != target and col not in exclude_columns]

    # Optional: If you want to ensure *only numeric* columns are used, filter them:
    numeric_features = []
    for c in features:
        # 'float64', 'int64', etc. means numeric
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_features.append(c)
        else:
            print(f"[WARN] Non-numeric column excluded automatically: {c}")

    # Finally, define X and y
    X = df[numeric_features].values  # all numeric
    y = df[target].values

    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    print(f"[INFO] Data loaded successfully. X.shape={X.shape}, y.shape={y.shape}")
    print(f"X shape = {X.shape}, type = {type(X)}")
    print(f"y shape = {y.shape}, type = {type(y)}")
    print("X dtype =", X.dtype)
    return X, y

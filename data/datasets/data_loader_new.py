import pandas as pd
import numpy as np
import os
import torch
import re
import datetime
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------

def clean_angle(value):
    """Convert tilt / azimuth values like "90°" or "30/15" to float.
    If parsing fails we propagate NaN so that later fill / drop rules apply."""
    if pd.isna(value):
        return np.nan

    value = str(value).replace("°", "").strip()

    if "/" in value:  # average e.g. "20/25"
        try:
            return np.mean([float(v) for v in value.split("/")])
        except ValueError:
            return np.nan

    try:
        return float(value)
    except ValueError:
        return np.nan

# ---------------------------------------------------------------------------
# Generic helpers to fallback‑parse human readable durations ( “2h 15m”, etc.)
# ---------------------------------------------------------------------------

def convert_time_to_minutes(text):
    """Fairly permissive fallback parser for durations that are *not* of the
    colon‑separated style handled later. Returns **minutes** or None."""
    if pd.isna(text) or text in ("", "nan", "None"):
        return None

    if isinstance(text, pd.Timedelta):
        return text.total_seconds() / 60

    if isinstance(text, (datetime.datetime, datetime.time)):
        return text.hour * 60 + text.minute + text.second / 60

    text = str(text).strip().lower()

    # hh:mm(:ss)
    m = re.match(r"^(\d+):(\d+)(?::(\d+))?$", text)
    if m:
        h, m_, s = (int(x) if x else 0 for x in m.groups())
        return h * 60 + m_ + s / 60

    # "2h 15m" / "3h"
    m = re.match(r"^(\d+)\s*h(?:\s*(\d+)\s*m)?$", text)
    if m:
        h = int(m.group(1)); m_ = int(m.group(2) or 0)
        return h * 60 + m_

    # "90mins"
    m = re.match(r"^(\d+)\s*mins?$", text)
    if m:
        return int(m.group(1))

    # Plain integer – assume minutes
    if text.isdigit():
        return int(text)

    return None  # give up

# ---------------------------------------------------------------------------
# Robust Excel target‑column parser
# ---------------------------------------------------------------------------

def _interpret_two_colon_parts(parts, total_days):
    """Handle ambiguous "a:b" cases.
    * If `total_days > 1`  ⇒ (total_days‑1) days + a h b m
    * Else if a ≥ 24       ⇒ treat a as *days*   (a d b h)
    * Otherwise            ⇒ treat as *hours:minutes*"""
    a, b = parts
    if total_days > 1:
        return (total_days - 1) * 1440 + a * 60 + b
    if a >= 24:
        return a * 1440 + b * 60
    return a * 60 + b


def parse_install_time(row, target_col, days_col):
    """Convert the raw *target* cell plus the sibling "Total # of Days on Site"
    to **minutes**.  This function tries, in order:
    1. native Timedelta / datetime objects
    2. "D:H:M:S", "H:M:S", "H:M" or "D:H"  colon formats
    3. Fallback regexes handled by `convert_time_to_minutes`.
    Returns None if parsing fails."""
    raw = row[target_col]
    total_days = row.get(days_col, np.nan)

    # 1. Missing value
    if pd.isna(raw) or raw in ("", "nan", "None"):
        return None

    # 2. Timedelta
    if isinstance(raw, pd.Timedelta):
        return raw.total_seconds() / 60

    # 3. datetime / time: Excel stores "1 day + hh:mm:ss" as datetime
    if isinstance(raw, (datetime.datetime, datetime.time)):
        base_min = 1440 if total_days > 1 else 0  # Excel never starts at day‑0
        return base_min + raw.hour * 60 + raw.minute + raw.second / 60

    # 4. String parsing
    text = str(raw).strip()
    parts = text.split(":")
    try:
        parts_f = [float(p) for p in parts]
    except ValueError:
        parts_f = []  # will fall back

    if len(parts_f) == 4:          # D:H:M:S
        d, h, m, s = parts_f
        return d * 1440 + h * 60 + m + s / 60
    if len(parts_f) == 3:          # H:M:S
        h, m, s = parts_f
        return h * 60 + m + s / 60
    if len(parts_f) == 2:          # ambiguous → helper
        return _interpret_two_colon_parts(parts_f, total_days)

    # 5. Fallback regex rules
    return convert_time_to_minutes(text)

# ----------------------------------------------------------
# Main loader
# ----------------------------------------------------------

def load_data(file_path=None):
    """Read Excel, clean + standardise features, *robustly* convert the target
    column to minutes, then apply √‑transform + scaling.
    Returns: X_train, X_val, y_train, y_val, y_scaler"""

    # ---------------- Resolve path ----------------
    if file_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, "..", "raw_data", "uploaded_data.xlsx")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Excel file not found: {file_path}")

    # --------------- Read header row heuristically ---------------
    preview = pd.read_excel(file_path, engine="openpyxl", nrows=10, header=None)
    header_row = preview.apply(lambda r: r.notna().sum(), axis=1).idxmax()
    df = pd.read_excel(file_path, engine="openpyxl", header=header_row)

    # --------------- Basic sanitising ---------------
    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    df.columns = df.columns.str.strip().str.replace("\n", " ")

    df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")], inplace=True)

    if df.empty:
        raise ValueError("Excel appears empty after cleaning header/blank rows.")

    # --------------- Target engineering ---------------
    target = "Total Direct Time for Project for Hourly Employees (Including Drive Time)"
    days_col = "Total # of Days on Site"

    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in sheet.")

    if days_col not in df.columns:
        df[days_col] = 1  # assume 1 day if the column is missing

    df[target] = df.apply(lambda r: parse_install_time(r, target, days_col), axis=1)

    # Process Drive Time if present
    if "Drive Time" in df.columns:
        drive = df["Drive Time"].apply(convert_time_to_minutes).fillna(0)
        df[target] = df[target] - drive

    # Clip negatives and apply √‑transform (stabilise variance)
    df[target] = df[target].clip(lower=0)
    df[target] = np.sqrt(df[target])

    # Drop rows where target still missing
    df.dropna(subset=[target], inplace=True)

    # --------------- Feature engineering ---------------
    if "Tilt" in df.columns:
        df["Tilt"] = df["Tilt"].apply(clean_angle)
    if "Azimuth" in df.columns:
        df["Azimuth"] = df["Azimuth"].apply(clean_angle)

    # yes/no → 1/0
    bool_cols = [c for c in df.columns if df[c].dropna().astype(str).str.lower().isin(["yes", "no"]).all()]
    for c in bool_cols:
        df[c] = df[c].map({"yes": 1, "no": 0, "Yes": 1, "No": 0})

    # Label‑encode remaining categoricals
    cat_cols = df.select_dtypes(exclude=["number"]).columns.difference(bool_cols + [target])
    for c in cat_cols:
        df[c] = df[c].astype(str).factorize()[0] + 1

    # Exclude obviously non‑predictive / leakage columns
    exclude = [
        "Project ID", "Notes", "Total # of Days on Site", "Estimated # of Salaried Employees on Site",
        "Estimated Salary Hours", "Estimated Total Direct Time", "Estimated Total # of People on Site",
        "Drive Time"
    ]
    features = [c for c in df.columns if c not in exclude + [target]]

    # Ensure numeric & fill NaNs with zero before scaling
    df[features] = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)

    # --------------- Scaling ---------------
    X_scaler = StandardScaler()
    df[features] = X_scaler.fit_transform(df[features])

    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(df[[target]])

    # --------------- Torch tensors + train/val split ---------------
    X_tensor = torch.tensor(df[features].values, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=42
    )

    return X_train, X_val, y_train, y_val, y_scaler


if __name__ == "__main__":
    X_train, X_val, y_train, y_val, y_scaler = load_data()
    print("✔ Data loader ran successfully. Train shape:", X_train.shape)

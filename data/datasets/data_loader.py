import pandas as pd
import os
import torch
import re
from sklearn.preprocessing import LabelEncoder

def convert_time_to_minutes(time_str):
    """ 将时间字符串转换为分钟数 """
    time_str = str(time_str).strip().lower()
    
    # 处理 "1 hour 32 mins" / "45 mins" / "2 hours"
    match = re.match(r'(?:(\d+)\s*hours?)?\s*(?:(\d+)\s*mins?)?', time_str)
    if match:
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2)) if match.group(2) else 0
        return hours * 60 + minutes
    return None  # 转换失败返回 None（稍后用 fillna 处理）

def load_data():
    """ 读取 Excel 数据并进行预处理 """
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "..", "raw_data", "Data.xlsx")  

    df = pd.read_excel(file_path, engine="openpyxl")

    # 目标变量
    target = "Total Direct Time for Project for Hourly Employees (Including Drive Time)"

    # 处理 Drive Time
    if "Drive Time" in df.columns:
        df["Drive Time"] = df["Drive Time"].apply(convert_time_to_minutes)

    # 处理 Yes/No 列：转换为 1/0
    boolean_cols = [col for col in df.columns if df[col].dropna().astype(str).apply(lambda x: x.lower() in ["yes", "no"]).all()]
    for col in boolean_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0, "yes": 1, "no": 0})  

    # 选择数值列
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # 选择类别列并进行 Label Encoding
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in boolean_cols + [target]]  # 排除布尔列和目标列

    # **使用 Label Encoding**
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str)  # 确保都是字符串
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # 保存编码器，方便以后解码

    # 确保目标列是数值型
    df[target] = pd.to_numeric(df[target], errors="coerce")

    # 删除 NaN 行
    df = df.dropna(subset=[target])

    # 重新获取所有特征列
    features = [col for col in df.columns if col != target]

    # **确保所有列都是 float 类型**
    df[features] = df[features].apply(pd.to_numeric, errors="coerce")  
    df = df.fillna(0)  # 填充 NaN 值，避免 PyTorch 报错

    print(f"✅ Selected features: {features}")

    # **转换为 PyTorch 张量**
    try:
        X = torch.tensor(df[features].values, dtype=torch.float32)
        y = torch.tensor(df[target].values, dtype=torch.float32).view(-1, 1)
    except Exception as e:
        print("❌ PyTorch Tensor 转换失败！检查数据是否仍包含非数值项。")
        print(e)
        return None, None

    print(f"✅ Data loaded successfully! X shape: {X.shape}, y shape: {y.shape}")
    return X, y, label_encoders  # 返回编码器，方便解码

if __name__ == "__main__":
    X, y, label_encoders = load_data()

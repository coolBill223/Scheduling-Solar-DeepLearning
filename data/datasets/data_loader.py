import pandas as pd
import os
import torch
import re
import datetime
import joblib  # 用于保存和加载标准化模型
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder  # ✅ 确保导入 StandardScaler

import re

def convert_time_to_minutes(value):
    """ 将时间数据转换为分钟数 """
    if pd.isna(value) or value in ["nan", "None", ""]:
        return None  # 让 fillna() 处理，而不是返回 0

    if isinstance(value, pd.Timedelta):  # 处理 timedelta64
        return value.total_seconds() / 60
    elif isinstance(value, datetime.datetime) or isinstance(value, datetime.time):  
        return value.hour * 60 + value.minute
    elif isinstance(value, str):  # 处理字符串格式的时间
        value = value.strip().lower()

        # **匹配 HH:MM:SS 或 HH:MM**
        match = re.match(r"(\d+):(\d+)(?::(\d+))?", value)
        if match:
            h, m, s = map(lambda x: int(x) if x else 0, match.groups())
            return h * 60 + m

        # **匹配 '2h 15m' 这种格式**
        match = re.match(r"(\d+)\s*h\s*(\d*)\s*m?", value, re.IGNORECASE)
        if match:
            h = int(match.group(1))
            m = int(match.group(2)) if match.group(2) else 0
            return h * 60 + m

        # **匹配 'XX mins' 格式**
        match = re.match(r"(\d+)\s*mins?", value)
        if match:
            return int(match.group(1))  # 直接返回分钟数

        # **纯数字格式，可能是分钟**
        if value.isnumeric():
            return int(value)

    return None  # 解析失败返回 None，让 fillna() 处理



def load_data():
    """ 读取 Excel 数据并进行预处理 """
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "..", "raw_data", "Data.xlsx")  

    if not os.path.exists(file_path):
        print(f"文件未找到: {file_path}")
        return None, None, None, None 

    df = pd.read_excel(file_path, engine="openpyxl")
    df.columns = df.columns.str.strip().str.replace("\n", " ")

    if df.empty:
        print("读取的数据为空，请检查数据文件！")
        return None, None, None, None  

    print(f"数据加载成功，形状: {df.shape}")

    
    # 目标变量
    target = "Total Direct Time for Project for Hourly Employees (Including Drive Time)"
    
    if target not in df.columns:
        print(f"目标列 {target} 不存在！")
        return None, None, None, None  

    print(f"🔍 目标列 {target} 的唯一值: {df[target].unique()}")

    if pd.api.types.is_timedelta64_dtype(df[target]):
        print(f"🔍 `{target}` 是 timedelta64 类型，转换为分钟数")
        df[target] = df[target].dt.total_seconds() / 60
        
    if df[target].apply(lambda x: isinstance(x, (datetime.datetime, datetime.time, str, pd.Timedelta))).any():
        print(f"🔍 `{target}` 包含时间格式数据，转换为分钟数")
        df[target] = df[target].apply(convert_time_to_minutes)

    # ✅ **确保 `target` 是数值**
    df[target] = pd.to_numeric(df[target], errors="coerce").astype("float64")
    
    if df[target].isnull().sum() > 0:
        df[target].fillna(df[target].mean(), inplace=True)

    print(f"🔍 目标列 {target} 为空的行数: {df[target].isnull().sum()}")
    print(df[target].dtype)
    print(df[target].head(10))

    # 处理 Drive Time
    print(f"🔍 原始 Drive Time 前 10 行:\n{df['Drive Time'].head(10)}")
    print(f"🔍 原始 Drive Time 的唯一值: {df['Drive Time'].unique()[:20]}")

    if "Drive Time" in df.columns:
        print(f"🔍 Drive Time 列数据类型: {df['Drive Time'].dtype}")

        # 如果是 timedelta64，直接转换为分钟
        if pd.api.types.is_timedelta64_dtype(df["Drive Time"]):
            print("🔍 `Drive Time` 是 timedelta64 类型，转换为分钟数")
            df["Drive Time"] = df["Drive Time"].dt.total_seconds() / 60
        else:
            print("🔍 `Drive Time` 可能是字符串或时间格式，尝试转换")
            df["Drive Time"] = df["Drive Time"].astype(str).str.strip()  # 先去掉空格
            df["Drive Time"] = df["Drive Time"].replace(["", "nan", "None"], pd.NA)  # 处理空字符串
            df["Drive Time"] = df["Drive Time"].apply(convert_time_to_minutes)

        # 确保转换成功
        print(f"🔍 处理后 Drive Time 前 10 行:\n{df['Drive Time'].head(10)}")

    # **填充 NaN 为 0，避免影响后续计算**
    df["Drive Time"].fillna(0, inplace=True)


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
    for col in categorical_cols:
        df[col] = df[col].astype(str)  # 确保都是字符串
        df[col] = df[col].factorize()[0] + 1  # 类别从 1 开始

    # 确保目标列是数值型
    df[target] = pd.to_numeric(df[target], errors="coerce")

    print(f"🔍 目标列 {target} 为空的行数: {df[target].isnull().sum()}")
    print(df[target].dtype)
    print(df[target].head(10))

    df = df.dropna(subset=[target])  # 删除 y 为空的行


    print(f"📊 数据集列名: {df.columns.tolist()}")

    # **检查空值**
    print(f"🔍 缺失值情况:\n{df.isnull().sum()}")


    # 重新获取所有特征列
    features = [col for col in df.columns if col != target and col != "Project ID"]
    missing_features = [col for col in features if col not in df.columns]

    if missing_features:
        print(f"❌ 缺失的特征列: {missing_features}")
        return None, None, None, None  

    # **检查特征列是否为空**
    print(f"🔍 选定的特征数据:\n{df[features].head()}")

   # **标准化 X（输入特征）**
    # **确保所有特征都是数值型**
    for col in features:
        if pd.api.types.is_timedelta64_dtype(df[col]):
            print(f"🔍 `{col}` 是 timedelta64 类型，转换为分钟数")
            df[col] = df[col].dt.total_seconds() / 60

# **标准化 X（输入特征）**
    X_scaler = StandardScaler()
    df[features] = df[features].fillna(0)  # 填充 NaN 为 0，避免标准化错误
    df[features] = X_scaler.fit_transform(df[features])


    # **归一化 y（目标变量）**
    y_scaler = MinMaxScaler()
    y = df[[target]].values
    y = pd.DataFrame(y).fillna(0).values  # ✅ 填充 NaN 为 0
    y = y_scaler.fit_transform(y)

    # **保存 scaler 以便之后使用**
    joblib.dump(X_scaler, "checkpoints/X_scaler.pkl")
    joblib.dump(y_scaler, "checkpoints/y_scaler.pkl")

    # **转换为 PyTorch 张量**
    X = torch.tensor(df[features].values, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    print(f"✅ Selected features: {features}")
    print(f"✅ Data loaded successfully! X shape: {X.shape}, y shape: {y.shape}")

    return X, y, X_scaler, y_scaler  # 返回 scaler 以便反归一化

if __name__ == "__main__":
    X, y, X_scaler, y_scaler = load_data()

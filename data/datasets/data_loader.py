import pandas as pd
import os
import torch
import re
import joblib  # 用于保存和加载标准化模型
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder  # ✅ 确保导入 StandardScaler

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
    df_raw = pd.read_excel(file_path, engine="openpyxl", dtype=str)  # 以字符串格式读取，防止数据丢失
    print(df_raw[target].head(10))
    
    if target not in df.columns:
        print(f"目标列 {target} 不存在！")
        return None, None, None, None  


    # 处理 Drive Time
    if "Drive Time" in df.columns:
        df["Drive Time"] = df["Drive Time"].apply(convert_time_to_minutes)

    # **将 timedelta64 转换为 float64（单位：分钟）**
    for col in df.select_dtypes(include=['timedelta64']).columns:
        df[col] = df[col].dt.total_seconds() / 60  # 转换为分钟
        print(f"转换列 {col} 为 float64（分钟）")

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
    X_scaler = StandardScaler()
    df[features] = df[features].fillna(0)  # ✅ 填充 NaN 为 0

    if df[features].shape[0] == 0:
        print("df[features] 为空，无法标准化！")
        return None, None, None, None 
    
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

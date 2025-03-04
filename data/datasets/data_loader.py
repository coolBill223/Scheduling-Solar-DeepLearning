import pandas as pd
import os
import torch

def time_to_minutes(time_str):
    """ 把时间字符串转换为分钟数 """
    if isinstance(time_str, str):
        time_str = time_str.lower().strip()  

        if "hour" in time_str or "h" in time_str:  
            time_str = time_str.replace("hours", "").replace("hour", "").replace("h", "").strip()
            parts = time_str.replace("mins", "").replace("min", "").replace("m", "").strip().split()

            if len(parts) == 2:  # "1 hour 32 mins" -> ["1", "32"]
                return int(parts[0]) * 60 + int(parts[1])  
            return int(parts[0]) * 60  # "1 hour" -> ["1"]

        elif "min" in time_str or "m" in time_str:  
            return int(time_str.replace("mins", "").replace("min", "").replace("m", "").strip())

        elif ":" in time_str:  
            h, m, *_ = map(int, time_str.split(":"))
            return h * 60 + m

    return None  

def load_data():
    """ 读取 Excel 数据并进行预处理 """
    
    # 获取当前文件的路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "../raw_data/Data.xlsx")  # 访问 Excel 文件
    
    # 读取 Excel 数据
    df = pd.read_excel(file_path, engine="openpyxl", skiprows=1)  # 跳过前1行（如果第一行是空的）

    # 处理列名
    df.columns = df.iloc[0]  # 设定新列名
    df = df[1:].reset_index(drop=True)  # 删除旧的列名行
    df.columns = df.columns.str.strip()  # 去掉列名中的空格
    df = df.loc[:, ~df.columns.duplicated()]  # 删除重复列

    # 处理时间格式
    if "Drive Time" in df.columns:
        df["Drive Time"] = df["Drive Time"].apply(time_to_minutes)
    if "Total Direct Time for Project for Hourly Employees (Including Drive Time)" in df.columns:
        df["Total Direct Time for Project for Hourly Employees (Including Drive Time)"] = df["Total Direct Time for Project for Hourly Employees (Including Drive Time)"].apply(time_to_minutes)

    # 选择模型所需的特征
    features = [col for col in df.columns if col != target]  # 选择除目标列以外的所有列
    target = "Total Direct Time for Project for Hourly Employees (Including Drive Time)"

    # 去除包含 NaN 的行
    df = df.dropna(subset=features + [target])

    # 转换为 PyTorch 张量
    X = torch.tensor(df[features].values, dtype=torch.float32)
    y = torch.tensor(df[target].values, dtype=torch.float32).view(-1, 1)

    return X, y

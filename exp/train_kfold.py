import sys
import os

# 获取当前文件的绝对路径，并找到项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))  # 上一级目录
sys.path.append(project_root)  # 添加到 sys.path

print(f"📌 已添加项目根目录到 sys.path: {project_root}")

import torch
import joblib
import numpy as np
import subprocess

num_runs = 10  # 训练次数
best_mse = float("inf")
best_model_path = "checkpoints/best_model.pth"
best_scaler_path = "checkpoints/y_scaler.pkl"

os.makedirs("checkpoints", exist_ok=True)  # 确保 checkpoint 文件夹存在

for i in range(num_runs):
    print(f"🔄 运行 {i+1}/{num_runs} ...")

    # 运行 train.py
    subprocess.run(["python", "exp/train.py"], check=True)

    # 加载当前训练的模型并计算 MSE
    model = torch.load(best_model_path)  # 载入最新训练的模型
    y_scaler = joblib.load(best_scaler_path)

    from data.datasets.data_loader import load_data
    from models.solar_time_model import MLPModel

    X, y, _, _ = load_data()  # 重新加载数据

    model = MLPModel(input_dim=X.shape[1])
    model.load_state_dict(torch.load(best_model_path))  # 加载权重
    model.eval()

    # 计算 MSE
    with torch.no_grad():
        predictions = model(X)
        y = y.cpu().numpy().reshape(-1, 1)
        predictions = predictions.cpu().numpy().reshape(-1, 1)
        y = np.clip(y_scaler.inverse_transform(y).flatten(), 0, None)
        predictions = np.clip(y_scaler.inverse_transform(predictions).flatten(), 0, None)
        mse = np.mean((predictions - y) ** 2)

    print(f"📉 运行 {i+1} - MSE: {mse:.4f}")

    # 记录最好的模型
    if mse < best_mse:
        best_mse = mse
        torch.save(model.state_dict(), "checkpoints/best_overall_model.pth")
        joblib.dump(y_scaler, "checkpoints/best_overall_scaler.pkl")

print(f"✅ 最佳模型已保存至 'checkpoints/best_overall_model.pth'，最低 MSE: {best_mse:.4f}")

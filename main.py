# 主程序
import torch
from models.solar_time_model import SolarTimeModel
from data.datasets.data_loader import load_data
from utils.metrics import mean_absolute_error, mean_squared_error

# 加载数据
X, y, scaler = load_data()

# 加载训练好的模型
model = SolarTimeModel(input_dim=X.shape[1])
model.load_state_dict(torch.load("checkpoints/model.pth"))
model.eval()

# 预测
with torch.no_grad():
    predictions = model(X)

# 计算误差
mae = mean_absolute_error(y, predictions)
mse = mean_squared_error(y, predictions)

print(f"MAE: {mae:.4f}, MSE: {mse:.4f}")

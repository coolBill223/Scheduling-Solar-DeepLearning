# 训练代码
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import joblib  # 用于保存标准化器

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.datasets.data_loader import load_data
from models.solar_time_model import SolarTimeModel

# 载入数据
X, y, X_scaler, y_scaler = load_data()  # ✅ 现在会返回 scaler

print(f"🔍 检查 X 是否含 NaN: {torch.isnan(X).sum().item()} 个 NaN")
print(f"🔍 检查 y 是否含 NaN: {torch.isnan(y).sum().item()} 个 NaN")


dataset = torch.utils.data.TensorDataset(X, y)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# 初始化模型
model = SolarTimeModel(input_dim=X.shape[1])

# 选择损失函数 & 优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-8)

# 训练
num_epochs = 50
best_loss = float("inf")

os.makedirs("checkpoints", exist_ok=True)  # ✅ 确保模型存储文件夹存在

for epoch in range(num_epochs):
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)  # ✅ 计算平均损失

    # 每 10 轮打印损失
    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # 如果损失更小，则保存模型
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), "checkpoints/best_model.pth")

# ✅ 训练完成后保存最终模型
torch.save(model.state_dict(), "checkpoints/final_model.pth")
joblib.dump(y_scaler, "checkpoints/y_scaler.pkl")  # ✅ 保存 y_scaler 以便预测时反归一化

print("✅ Training Complete! Model saved to 'checkpoints/final_model.pth'")
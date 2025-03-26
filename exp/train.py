# 训练代码
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import joblib  # 用于保存标准化器
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.datasets.data_loader import load_data
from models.solar_time_model import MLPModel

# 载入数据
X, y, X_scaler, y_scaler = load_data()  #现在会返回 scaler

print(f"检查 X 是否含 NaN: {torch.isnan(X).sum().item()} 个 NaN")
print(f"检查 y 是否含 NaN: {torch.isnan(y).sum().item()} 个 NaN")

# **划分 7:3 训练/验证集**
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42,shuffle=False)

# **创建 PyTorch DataLoader**
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)

# 初始化模型
model = MLPModel(input_dim=X.shape[1])

# 选择损失函数 & 优化器
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)  # MLP 一般可以用稍大的学习率
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)


# **存储损失值**
train_losses = []
val_losses = []
num_epochs = 300
best_val_loss = float("inf")
no_improve_count = 0  # 记录连续未改善的轮数
early_stopping_patience = 20  # 早停耐心值

os.makedirs("checkpoints", exist_ok=True)  # 确保模型存储文件夹存在

# **定义验证函数**
def evaluate(model, val_loader):
    model.eval()  # 设置为评估模式
    total_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            total_loss += loss.item()
    return total_loss / len(val_loader)  # 返回平均损失

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 防止梯度爆炸
        optimizer.step()
        
        epoch_loss += loss.item()

    train_loss = epoch_loss / len(train_loader)  # 计算训练集的平均损失
    val_loss = evaluate(model, val_loader)  # 计算验证集损失
    
    if np.isnan(train_loss) or np.isinf(train_loss):
        print(f"⚠️ Warning: NaN detected in losses at epoch {epoch}!")
        break  # 直接停止训练，防止无效训练

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # **保存最好的模型**
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "checkpoints/best_model.pth")
        no_improve_count = 0  # 重新计数
    else:
        no_improve_count += 1

    # **早停检查**
    if no_improve_count >= early_stopping_patience:
        print(f"🚨 Early stopping at epoch {epoch}")
        break
    
    # **这里是关键：让学习率动态调整**
    scheduler.step(val_loss)  # 🟢 每轮结束后检查 `val_loss`，如果连续 5 轮没有下降，就减少学习率

# 训练完成后保存最终模型
torch.save(model.state_dict(), "checkpoints/final_model.pth")
joblib.dump(y_scaler, "checkpoints/y_scaler.pkl")  # 保存 y_scaler 以便预测时反归一化


# ** 计算 MSE 和 MPE**
def evaluate_mse_mpe(model, X, y, y_scaler):
    model.eval()
    with torch.no_grad():
        predictions = model(X)

        # ✅ 确保 `y` 和 `predictions` 转换为 NumPy 数组
        y = y.cpu().numpy().reshape(-1, 1)  # ⚠️ 添加 reshape
        predictions = predictions.cpu().numpy().reshape(-1, 1)  # ⚠️ 添加 reshape

        # ✅ 反归一化 `y` 和 `predictions`
        y = np.clip(y_scaler.inverse_transform(y).flatten(), 0, None)  # 只允许非负数
        predictions = np.clip(y_scaler.inverse_transform(predictions).flatten(), 0, None)

        # 计算 MSE
        mse = np.mean((predictions - y) ** 2)

        # 计算 MPE，避免除零错误
        nonzero_mask = y != 0
        mpe = np.mean(((predictions - y) / y)[nonzero_mask]) * 100  # 百分比误差

    return mse, mpe


mse, mpe = evaluate_mse_mpe(model, X_val, y_val, y_scaler)

print(f"Final Evaluation on Validation Set:")
print(f"MSE: {mse:.4f}")
print(f"MPE: {mpe:.2f}%")

# **绘制训练曲线**
plt.figure(figsize=(8, 6))
plt.plot(range(len(train_losses)), train_losses, label="Train Loss", marker="o", linestyle="-")
plt.plot(range(len(val_losses)), val_losses, label="Validation Loss", marker="s", linestyle="-")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.grid()
plt.savefig("checkpoints/loss_curve.png")  # **保存图表**
plt.show()

print(" Training Complete! Best model saved to 'checkpoints/best_model.pth'")

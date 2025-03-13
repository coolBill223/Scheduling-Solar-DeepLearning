# 评价指标计算
import torch

def mean_absolute_error(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred)).item()

def mean_squared_error(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2).item()

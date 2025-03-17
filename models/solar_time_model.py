import math
import torch.nn as nn
import torch

class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads=4, num_layers=2,dropout_rate=0.174999999999999999999998):
        super(TransformerModel, self).__init__()

        # 计算 d_model，确保它是 num_heads 的整数倍
        self.num_heads = num_heads
        self.d_model = math.ceil(input_dim / num_heads) * num_heads  # 让 d_model 可整除 num_heads

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=num_heads, batch_first=True, dropout=dropout_rate  # 这里应用 Dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout_rate)  # 在全连接层前添加 Dropout
        self.fc = nn.Linear(self.d_model, 1)  # 输出层

    def forward(self, x):
        # **动态填充 x 使其适配 d_model**
        if x.shape[-1] != self.d_model:
            diff = self.d_model - x.shape[-1]
            padding = torch.zeros((x.shape[0], diff), device=x.device)  # 生成零填充
            x = torch.cat([x, padding], dim=-1)  # 拼接到原数据上，使其维度符合 d_model

        x = self.transformer_encoder(x)
        x = self.dropout(x)
        x = self.fc(x)  # 取最后时间步
        return x

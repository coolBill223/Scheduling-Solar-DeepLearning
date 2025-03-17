import math
import torch.nn as nn
import torch

class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads=4, num_layers=2, dropout_rate=0.3):  # 适当增大 Dropout
        super(TransformerModel, self).__init__()

        # **确保 input_dim 能被 num_heads 整除**
        self.num_heads = num_heads
        self.d_model = math.ceil(input_dim / num_heads) * num_heads  # 让 d_model 可整除 num_heads

        # **使用 Linear 层转换 input_dim 而不是填充 0**
        self.input_projection = nn.Linear(input_dim, self.d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=num_heads, batch_first=True, dropout=dropout_rate
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout_rate)  # 在全连接层前添加 Dropout
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, 512),  # ⚡ 增加神经元，提升表达能力
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # **使用 `Linear` 映射 input_dim -> d_model**
        x = self.input_projection(x)

        x = self.transformer_encoder(x)
        x = self.dropout(x)

        # **取 Transformer 的最后一个时间步**
        if x.dim() == 2:
            x = self.fc(x)
        else:
            x = self.fc(x[:, -1, :]) 
        return x

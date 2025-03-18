import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, input_dim):
        super(MLPModel, self).__init__()

        # 🔹 让输入 x 变成 64 维，方便残差连接
        self.shortcut1 = nn.Linear(input_dim, 64, bias=False)

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 🔹 让 64 维输入到 32 维时，也有 shortcut
        self.shortcut2 = nn.Linear(64, 32, bias=False)

        self.fc2 = nn.Sequential(
            nn.Linear(64, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(32, 16, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.fc4 = nn.Sequential(
            nn.Linear(16, 8, bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.05)
        )

        self.out = nn.Linear(8, 1)  # 输出层

    def forward(self, x):
        # ✅ 第一层加 shortcut
        x = self.fc1(x) + self.shortcut1(x)  

        # ✅ 第二层加 shortcut
        x = self.fc2(x) + self.shortcut2(x)

        x = self.fc3(x)
        x = self.fc4(x)
        x = self.out(x)
        return x

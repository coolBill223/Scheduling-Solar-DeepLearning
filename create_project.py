import os

project_name = "Scheduling-Solar-DeepLearning"

# 目录结构
folders = [
    "data/datasets",
    "layers",
    "models",
    "exp",
    "configs",
    "utils",
    "logs",
    "scripts",
    "checkpoints",
    "model_hub"
]

# 主要 Python 文件
files = {
    "data/__init__.py": "",
    "data/datasets/__init__.py": "",
    "data/datasets/data_loader.py": "# 数据加载代码",
    
    "models/__init__.py": "",
    "models/solar_time_model.py": "# 太阳能安装时间预测模型",
    
    "exp/__init__.py": "",
    "exp/train.py": "# 训练代码",
    
    "configs/__init__.py": "",
    "configs/config.py": "# 配置文件",
    
    "utils/__init__.py": "",
    "utils/utils.py": "# 工具函数",
    "utils/metrics.py": "# 评价指标计算",
    
    "logs/__init__.py": "",
    "scripts/__init__.py": "",
    
    "checkpoints/__init__.py": "",
    
    "model_hub/__init__.py": "",
    "model_hub/pretrained_model.pth": "",  # 预训练模型文件
    
    "main.py": "# 主程序",
    "requirements.txt": "torch\npandas\nopenpyxl\nnumpy",
    "environment.yml": "name: solar-deep-learning\ndependencies:\n  - python=3.9\n  - pytorch\n  - pandas\n  - openpyxl",
    "readme.md": "# 太阳能安装时间预测项目"
}

# 创建文件夹
for folder in folders:
    os.makedirs(os.path.join(project_name, folder), exist_ok=True)

# 创建文件
for file_path, content in files.items():
    full_path = os.path.join(project_name, file_path)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)

print(f"项目 {project_name} 目录结构创建完成！")

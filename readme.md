# Solar Panel Installation Time Prediction (Deep Learning)

## Project Overview

This project is a web-based application that allows users to upload Excel datasets and train a deep learning model to predict solar panel installation time based on various features. The backend is implemented with Flask, and the model is trained using PyTorch.

## Features

- Upload `.xlsx` files through a user-friendly web interface
- One-click training of a prediction model
- Visualization of training results (loss curve and prediction vs actual plot)
- Modular structure separating backend, model, and frontend

## Setup Instructions

### 1. Install dependencies
pip install flask pandas numpy torch matplotlib scikit-learn

### 2. Run the server
python main.py

### 3. Open in browser
Visit: https://127.0.0.1:5000

## How to Use:
1. Drag and drop an Excel file into the upload area
2. Click the "Train" button to start training
3. After training, click "View Result" to see visualizations and evaluation metrics

## Notes:
- The Excel file must include a target column, such as Install Time (min)
- Training outputs are saved in the checkpoints/ directory
- You can modify the model architecture in models/solar_time_model_tiny.py


## Author
Developed by Zhirui (Bill) Zhou. 
University of Virginia. 
School of Engineering and Applied Science

# 基于深度学习的太阳能面板安装时间预测系统

## 项目简介

本项目是一个基于网页的深度学习应用，用户可以上传 Excel 数据文件，训练模型预测太阳能面板的安装时间。后端使用 Flask 实现，模型使用 PyTorch 训练。

## 功能特点

- 网页支持拖拽上传 `.xlsx` 格式的文件
- 一键启动模型训练流程
- 显示训练效果图，包括损失曲线和预测对比图
- 模块化设计，前后端分离，代码结构清晰

## 使用说明

### 1. 安装依赖
pip install flask pandas numpy torch matplotlib scikit-learn

### 2. 启动服务器
python main.py

### 3. 打开网页
浏览器访问: https://127.0.0.1:5000

## 使用流程:
1. 拖拽 Excel 文件上传（文件中需包含目标列）
2. 点击 “Train” 按钮开始训练
3. 点击 “View Result” 查看训练图像和误差指标

## 注意事项:
- Excel 文件中必须包含如 Install Time (min) 的目标列
- 所有训练结果保存在 checkpoints/ 文件夹中
- 可在 models/solar_time_model_tiny.py 中自由修改模型结构


## 作者
作者：周知睿. 
弗吉尼亚大学. 
工程与应用科学学院



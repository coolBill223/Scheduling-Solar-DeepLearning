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
1. Upload an Excel file into the upload area
2. Click the "Train" button to start training
3. After training, click "View Result" to see visualizations and evaluation metrics
4. Click the estimated result page, and entring all the information to predict, then hit the Submit button to submit the form.

## Notes:
- The Excel file must include a target column, such as Install Time (min)
- Training outputs are saved in the checkpoints/ directory
- You can modify the model architecture in models/solar_time_model_tiny.py


## Author
Developed by Aden Kim, Heena Parekh, Yi (Eason) Ping, and Zhirui (Bill) Zhou. 
University of Virginia. 
School of Engineering and Applied Science




## 作者
作者：周知睿. 
弗吉尼亚大学. 
工程与应用科学学院



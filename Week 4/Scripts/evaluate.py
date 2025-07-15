# scripts/evaluate.py
import numpy as np  # 添加这行导入 NumPy
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def get_regression_model_performance(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2
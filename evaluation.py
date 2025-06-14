import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_error_metrics(actual, predicted):
    """Calculate MAE, RMSE, and MAPE."""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual.clip(min=1e-10))) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_forecast(actual, predicted):
    """
    Evaluate forecast using MAE and RMSE metrics.

    Parameters:
        actual (array-like): The ground truth values.
        predicted (array-like): The predicted values.

    Returns:
        dict: A dictionary containing 'MAE' and 'RMSE' scores.
    """
    # Validate inputs for common issues
    if len(actual) != len(predicted):
        raise ValueError("The 'actual' and 'predicted' arrays must have the same length.")
    
    # Compute metrics
    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted, squared=False)  # Corrected RMSE calculation
    
    # Return results in a dictionary
    return {'MAE': mae, 'RMSE': rmse}

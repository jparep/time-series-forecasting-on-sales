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

    Raises:
        ValueError: If the input arrays are empty or have different lengths.
        TypeError: If the inputs are not array-like.
    """
    # Validate inputs
    if not isinstance(actual, (list, np.ndarray, pd.Series)) or not isinstance(predicted, (list, np.ndarray, pd.Series)):
        raise TypeError("Inputs must be array-like (e.g., list, numpy array, or pandas Series).")
    
    if len(actual) == 0 or len(predicted) == 0:
        raise ValueError("Input arrays must not be empty.")
    
    if len(actual) != len(predicted):
        raise ValueError("The 'actual' and 'predicted' arrays must have the same length.")
    
    # Compute metrics
    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted, squared=False)  # RMSE calculation
    
    # Return results in a dictionary
    return {'MAE': mae, 'RMSE': rmse}
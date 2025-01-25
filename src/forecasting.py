from prophet import Prophet
import pandas as pd

def train_forecast_model(data: pd.DataFrame, target_col: str = 'sale') -> Prophet:
    """
    Train a Prophet model on time series data.
    
    Args:
        data (pd.DataFrame): Input data containing a datetime column and target column.
        target_col (str): The name of the column containing the target values.
    
    Returns:
        Prophet: A trained Prophet model.
    """
    if 'data' not in data.columns:
        raise ValueError("The input data must have a 'data' column for datetime values.")
    
    # Rename columns for Prophet requirements
    data = data.rename(columns={'data': 'ds', target_col: 'y'})
    
    # Initialize and fit the model
    model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
    model.fit(data)
    
    return model

def make_forecast(model: Prophet, periods: int = 12, freq: str = 'M') -> pd.DataFrame:
    """
    Make future predictions using a trained Prophet model.
    
    Args:
        model (Prophet): A trained Prophet model.
        periods (int): Number of future periods to predict.
        freq (str): Frequency of predictions (e.g., 'M' for monthly).
    
    Returns:
        pd.DataFrame: A DataFrame containing the forecasted values.
    """
    # Generate future dates
    future = model.make_future_dataframe(periods=periods, freq=freq)
    
    # Make predictions
    forecast = model.predict(future)
    
    return forecast

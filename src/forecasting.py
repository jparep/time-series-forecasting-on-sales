from prophet import Prophet
import pandas as pd

def train_forecast_model(
    data: pd.DataFrame, 
    target_col: str = 'sale', 
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = False,
    daily_seasonality: bool = False
) -> Prophet:
    """
    Train a Prophet model on time series data.
    
    Args:
        data (pd.DataFrame): Input data containing a datetime column ('date') and a target column.
        target_col (str): The name of the column containing the target values.
        yearly_seasonality (bool): Whether to enable yearly seasonality. Default is True.
        weekly_seasonality (bool): Whether to enable weekly seasonality. Default is False.
        daily_seasonality (bool): Whether to enable daily seasonality. Default is False.
    
    Returns:
        Prophet: A trained Prophet model.
    """
    # Validate required columns
    if 'date' not in data.columns or target_col not in data.columns:
        raise ValueError("The input data must have 'date' and target column.")

    # Convert 'date' to datetime format
    data['date'] = pd.to_datetime(data['date'])

    # Drop missing values
    data = data.dropna(subset=[target_col])

    # Rename columns for Prophet
    data = data.rename(columns={'date': 'ds', target_col: 'y'})
    
    # Initialize Prophet model
    model = Prophet(
        yearly_seasonality=yearly_seasonality, 
        weekly_seasonality=weekly_seasonality, 
        daily_seasonality=daily_seasonality
    )
    
    # Fit the model
    model.fit(data)
    
    return model

def make_forecast(model: Prophet, periods: int = 12, freq: str = 'M') -> pd.DataFrame:
    """
    Make future predictions using a trained Prophet model.
    
    Args:
        model (Prophet): A trained Prophet model.
        periods (int): Number of future periods to predict.
        freq (str): Frequency of predictions (e.g., 'M' for monthly, 'D' for daily).
    
    Returns:
        pd.DataFrame: A DataFrame containing the forecasted values.
    """
    # Generate future dates
    future = model.make_future_dataframe(periods=periods, freq=freq)
    
    # Make predictions
    forecast = model.predict(future)
    
    # Return selected columns
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

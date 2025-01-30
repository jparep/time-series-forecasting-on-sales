import matplotlib.pyplot as plt
import pandas as pd

def plot_forecast(
    data: pd.DataFrame, 
    forecast: pd.DataFrame, 
    target_col: str = 'sale',
    figsize: tuple = (12, 6),
    forecast_line_style: str = '--'
) -> None:
    """
    Plot historical data with forecast.

    Args:
        data (pd.DataFrame): Historical data with columns 'date' and the target column.
        forecast (pd.DataFrame): Forecast data from Prophet with 'ds', 'yhat', 'yhat_lower', and 'yhat_upper'.
        target_col (str): Name of the target column in the historical data. Default is 'sale'.
        figsize (tuple): Figure size for the plot. Default is (12, 6).
        forecast_line_style (str): Line style for the forecast line. Default is '--'.
    
    Returns:
        None: Displays the plot.
    """
    # Validate input columns
    required_data_cols = {'date', target_col}
    required_forecast_cols = {'ds', 'yhat', 'yhat_lower', 'yhat_upper'}
    
    if not required_data_cols.issubset(data.columns):
        raise ValueError(f"Historical data must contain {required_data_cols}.")
    if not required_forecast_cols.issubset(forecast.columns):
        raise ValueError(f"Forecast data must contain {required_forecast_cols}.")

    # Ensure 'date' and 'ds' columns are datetime
    data['date'] = pd.to_datetime(data['date'])
    forecast['ds'] = pd.to_datetime(forecast['ds'])

    # Plotting
    plt.figure(figsize=figsize)
    plt.plot(data['date'], data[target_col], label='Actual Sales', color='navy', linewidth=2)
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', linestyle=forecast_line_style, color='#FF8C00', linewidth=2)
    plt.fill_between(
        forecast['ds'], 
        forecast['yhat_lower'], 
        forecast['yhat_upper'], 
        color='#FF8C00', 
        alpha=0.2, 
        label='Confidence Interval'
    )

    # Enhancements
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(target_col.replace('_', ' ').title(), fontsize=12)
    plt.title(f"{target_col.replace('_', ' ').title()} Forecast", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

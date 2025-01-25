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
    if 'date' not in data.columns or target_col not in data.columns:
        raise ValueError("Historical data must contain 'date' and the specified target column.")
    if not all(col in forecast.columns for col in ['ds', 'yhat', 'yhat_lower', 'yhat_upper']):
        raise ValueError("Forecast data must contain 'ds', 'yhat', 'yhat_lower', and 'yhat_upper'.")

    # Plotting
    plt.figure(figsize=figsize)
    plt.plot(data['date'], data[target_col], label='Actual Sales', color='blue')
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', linestyle=forecast_line_style, color='orange')
    plt.fill_between(
        forecast['ds'], 
        forecast['yhat_lower'], 
        forecast['yhat_upper'], 
        color='orange', 
        alpha=0.2, 
        label='Confidence Interval'
    )

    # Enhancements
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Sales', fontsize=12)
    plt.title(f'{target_col.capitalize()} Forecast', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt

def plot_forecast(data, forecast, target_col='sale'):
    """Plot historical data with forecast."""
    plt.figure(figsize=(10,6))
    plt.plot(data['date'], data['target_col'], label='Acutal Sales')
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', linestyle='--')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2)
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title('Retial Sales Forecast')
    plt.legend()
    plt.show()
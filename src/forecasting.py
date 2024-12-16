import prophet as Prophet

def train_forecast_model(data, target_col='sale'):
    """Train Prophet model on time series data."""
    data = data.rename(columns={'data': 'ds', target_col: 'y'}) # Prophet requires ds and y
    model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
    model.fit(data)
    return model
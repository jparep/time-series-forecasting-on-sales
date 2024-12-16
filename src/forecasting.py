import prophet as Prophet

def train_forecast_model(data, target_col='sale'):
    """Train Prophet model on time series data."""
    data = data.rename(columns={'data': 'ds', target_col: 'y'}) # Prophet requires ds and y
    model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
    model.fit(data)
    return model

def make_forecast(model, periods=12, freq='M'):
    "make future prediction using Prophet model"
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast
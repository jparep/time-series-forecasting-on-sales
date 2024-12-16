from src.data_loader import load_sales_data
from src.forecasting import train_forecast_model, make_forecast
from src.evaluation import evaluate_forecast
from src.visualization import plot_forecast

# Load Data
FILE_PATH = 'data/retail_sale.csv'
sales_data = load_sales_data(FILE_PATH)

# Train Prophet Model
model = train_forecast_model(sales_data)

# Make Forecast
forecast = make_forecast(model, periods=12)

# Evaluate Model
evaluation_results = evaluate_forecast(
    actual=sales_data['sales'][-12:].values,
    predicted=forecast['yhat'][-12:].values
)
print(evaluation_results)
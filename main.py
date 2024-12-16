from src.data_loader import load_sales_data
from src.forecasting import train_forecast_model, make_forecast
from src.evaluation import evaluate_forecast
from src.visualization import plot_forecast

# Load Data
FILE_PATH = 'data/retail_sale.csv'
sales_data = load_sales_data(FILE_PATH)

# Train Prophet Model
model = train_forecast_model(sales_data)

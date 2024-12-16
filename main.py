from src.data_loader import load_sales_data
from src.forecasting import train_forecast_model, make_forecast
from src.evaluation import evaluate_forecast
from src.visualization import plot_forecast

# Load Data
FILE_PATH = 'data/retail_sale.csv'
df = load_sales_data(FILE_PATH)

# Train
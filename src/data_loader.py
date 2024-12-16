import pandas as pd

def load_sales_data(file_path):
    """Load sale data from a CSV file"""
    data = pd.read_csv(file_path, parse_dates=["data"])
    data.sort_values(by='data', inplace=True)
    return data
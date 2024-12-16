# Retail Sales Forecasting

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Enhancements](#enhancements)
- [Future Work](#future-work)
- [License](#license)

---

## Project Overview
This project uses **time series forecasting** techniques to predict monthly sales for a retail chain. The goal is to provide business insights and assist in inventory management, budgeting, and sales trend analysis.

We use **Meta’s Prophet** library for forecasting and include professional enhancements like:
- Hyperparameter tuning
- Cross-validation
- Feature engineering
- Interactive visualizations

---

## Features
- **Data Preprocessing**: Load and clean retail sales data.
- **Feature Engineering**: Add date-based features and aggregate sales to monthly levels.
- **Forecasting**: Predict future sales using the Prophet library.
- **Evaluation**: Evaluate forecasts using metrics like MAE and RMSE.
- **Visualization**: Generate plots to visualize historical data, predictions, and confidence intervals.

---

## Project Structure
```text
        retail-sales-forecasting/ ├── data/ │ ├── retail_sales.csv # Historical sales data ├── notebooks/ │ ├── eda.ipynb # Exploratory Data Analysis ├── src/ │ ├── data_loader.py # Data loading utilities │ ├── forecasting.py # Forecasting utilities │ ├── evaluation.py # Evaluation utilities │ ├── feature_engineering.py # Feature engineering utilities │ └── visualization.py # Visualization utilities ├── forecasts/ │ ├── sales_forecast.csv # Saved forecasts ├── requirements.txt # Project dependencies └── main.py # Main script for running the project
```

---

## Installation

### 1. Clone the Repository
```bash
    git clone https://github.com/japrep/time-series-forecasting-on-sales.git
    cd time-series-forecasting-on-sales
```

### 2. Install Dependencies

Create a virtual environment and install the required packages:
```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
```

## Usage

### 1. Load Data

Place the dataset (retail_sales.csv) in the data/ folder. Ensure the file contains columns Date and Sales.

### 2. Run the Project

Run the main script:
```bash
    python main.py
```

#### 3. Outputs

    Forecast results are saved in the forecasts/ folder as sales_forecast.csv.
    Visualizations are displayed during runtime.

## Dataset

Retail Sales Dataset: Monthly sales data for a retail chain.

    - Columns: Date, Sales
    - Source: Retail Sales Dataset on Kaggle
    - Path: data/retail_sales.csv

## Enhancements
Professional Enhancements

    - Hyperparameter Tuning:
        Tune Prophet parameters such as seasonality_mode and changepoint_prior_scale using grid search.
    - Cross-Validation:
        Evaluate model accuracy using Prophet’s built-in cross-validation tools.
    - Feature Engineering:
        Add date-based features such as year, month, day of the week, and weekend indicators.
    - Interactive Dashboard:
        Deploy the forecasting tool using Streamlit or Flask for real-time interaction.

## Future Work

    Integrate external factors (e.g., promotions, holidays) as regressors in the Prophet model.
    Extend forecasts to include predictions for individual store-level sales.
    Automate model retraining with new data.

## License

This project is licensed under the [MIT License](https://opensource.org/license/mit).
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
from datetime import datetime, timedelta
import os

def generate_data():
    print("Generating sample data...")
    np.random.seed(42)
    n_days = 365
    regions = ['North', 'South', 'East', 'West']
    products = [f'SKU{i:03d}' for i in range(1, 11)]
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    data = []
    for date in dates:
        for region in regions:
            for product in products:
                base_demand = np.random.normal(100, 20)
                seasonality = 20 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
                is_promotion = np.random.random() < 0.1
                promotion_effect = 30 if is_promotion else 0
                demand = max(0, int(base_demand + seasonality + promotion_effect))
                base_price = np.random.normal(50, 10)
                price = base_price * 0.8 if is_promotion else base_price
                data.append({
                    'date': date,
                    'region': region,
                    'product_id': product,
                    'sales_quantity': demand,
                    'price': round(price, 2),
                    'promotion_flag': is_promotion
                })
    df = pd.DataFrame(data)
    print("Data generation complete.")
    return df

def train_model(df):
    print("Training model...")
    df['date'] = pd.to_datetime(df['date'])
    
    # Feature Engineering
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['lag_7'] = df.groupby(['region', 'product_id'])['sales_quantity'].shift(7)
    df['lag_14'] = df.groupby(['region', 'product_id'])['sales_quantity'].shift(14)
    df['lag_30'] = df.groupby(['region', 'product_id'])['sales_quantity'].shift(30)
    df['rolling_mean_7'] = df.groupby(['region', 'product_id'])['sales_quantity'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df['rolling_mean_14'] = df.groupby(['region', 'product_id'])['sales_quantity'].transform(lambda x: x.rolling(14, min_periods=1).mean())
    df = df.dropna()

    feature_columns = [
        'month', 'day_of_week', 'quarter', 'year',
        'lag_7', 'lag_14', 'lag_30',
        'rolling_mean_7', 'rolling_mean_14',
        'price', 'promotion_flag'
    ]
    
    X = pd.get_dummies(df[feature_columns], columns=['month', 'day_of_week', 'quarter'])
    y = df['sales_quantity']
    
    # Align columns for prediction script
    train_size = int(len(df) * 0.8)
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # Save model and feature names
    model_data = {
        'model': model,
        'feature_names': X.columns.tolist(),
    }
    joblib.dump(model_data, 'models/demand_model.pkl')
    print("Model training complete. Model saved to models/demand_model.pkl")

def map_and_fill_columns(df):
    # Normalize all columns
    df.columns = [normalize_column(col) for col in df.columns]
    col_map = {}
    for req, aliases in COLUMN_ALIASES.items():
        found = None
        for alias in aliases:
            if alias in df.columns:
                found = alias
                break
        if found:
            col_map[req] = found
    # Rename columns to required names
    df = df.rename(columns=col_map)
    # Special handling: always ensure sales_quantity exists
    if 'sales_quantity' not in df.columns:
        # Try to create from any alias
        for alias in COLUMN_ALIASES['sales_quantity']:
            if alias in df.columns:
                df['sales_quantity'] = df[alias]
                break
        else:
            df['sales_quantity'] = 0  # Default if not found
    # Fill missing columns with defaults
    for col, default in DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
    return df

if __name__ == "__main__":
    sales_df = generate_data()
    train_model(sales_df) 
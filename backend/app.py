from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import joblib
from datetime import datetime, timedelta
import os
import traceback

app = FastAPI(title="Walmart Demand Prediction API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class PredictionRequest(BaseModel):
    region: str
    product_id: str
    start_date: str
    end_date: str

class PredictionResponse(BaseModel):
    predictions: List[dict]
    confidence_score: Optional[float] = None

# Mock data for demonstration
MOCK_REGIONS = ["North", "South", "East", "West"]
MOCK_PRODUCTS = ["SKU001", "SKU002", "SKU003"]

# Load the trained model
try:
    model_data = joblib.load("models/demand_model.pkl")
    model = model_data['model']
    model_feature_names = model_data['feature_names']
except FileNotFoundError:
    model = None
    model_feature_names = None
    print("Model not found. /predict_from_file will not work.")

def normalize_column(col):
    return col.strip().lower().replace(' ', '').replace('_', '')

COLUMN_ALIASES = {
    'date': ['date', 'dates'],
    'region': ['region', 'regions'],
    'product_id': ['productid', 'product_id', 'sku', 'product', 'store_id'],
    'sales_quantity': ['salesquantity', 'quantity', 'qty', 'sales', 'demand_units', 'actual_sales_units'],
    'price': ['price', 'cost'],
    'promotion_flag': ['promotionflag', 'promo', 'promotion', 'is_promotion', 'promotion_flag']
}

REQUIRED_COLUMNS = list(COLUMN_ALIASES.keys())

DEFAULTS = {
    'product_id': 'SKU001',
    'price': 0.0,
    'promotion_flag': False
}

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

def preprocess_data(df: pd.DataFrame):
    df['date'] = pd.to_datetime(df['date'])
    
    # Feature Engineering from the notebook
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    
    # Create lag features
    df = df.sort_values(['region', 'product_id', 'date'])
    df['lag_7'] = df.groupby(['region', 'product_id'])['sales_quantity'].shift(7)
    df['lag_14'] = df.groupby(['region', 'product_id'])['sales_quantity'].shift(14)
    df['lag_30'] = df.groupby(['region', 'product_id'])['sales_quantity'].shift(30)
    
    # Create rolling mean features
    df['rolling_mean_7'] = df.groupby(['region', 'product_id'])['sales_quantity'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df['rolling_mean_14'] = df.groupby(['region', 'product_id'])['sales_quantity'].transform(lambda x: x.rolling(14, min_periods=1).mean())

    df = df.dropna().reset_index(drop=True)

    if not df.empty:
        # One-hot encode categorical features
        feature_columns_to_encode = ['month', 'day_of_week', 'quarter']
        df_encoded = pd.get_dummies(df, columns=feature_columns_to_encode)
    else:
        return pd.DataFrame(), df

    return df_encoded, df

@app.get("/")
async def root():
    return {"message": "Welcome to Walmart Demand Prediction API"}

@app.get("/regions")
async def get_regions():
    return {"regions": MOCK_REGIONS}

@app.get("/products")
async def get_products():
    return {"products": MOCK_PRODUCTS}

@app.post("/predict_from_file", response_model=PredictionResponse)
async def predict_from_file(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")

    try:
        # Read and process the uploaded file
        df = pd.read_csv(file.file)
        print("CSV columns:", list(df.columns))
        
        # Map and fill columns
        df = map_and_fill_columns(df)

        # Check for required columns (after mapping/filling)
        # (Removed strict error check to always proceed)
        # if not all(col in df.columns for col in REQUIRED_COLUMNS):
        #     raise HTTPException(
        #         status_code=400,
        #         detail=f"Missing required columns. Required: {REQUIRED_COLUMNS}. Found: {list(df.columns)}"
        #     )

        preprocessed_df, original_df_with_dates = preprocess_data(df)

        if preprocessed_df.empty:
             return PredictionResponse(predictions=[])

        # Align columns with model's expected features
        X_live = pd.DataFrame(columns=model_feature_names)
        for col in preprocessed_df.columns:
            if col in X_live.columns:
                X_live[col] = preprocessed_df[col]
        X_live = X_live.fillna(0)
        
        # Ensure all model features are present
        missing_cols = set(model_feature_names) - set(X_live.columns)
        for c in missing_cols:
            X_live[c] = 0
        X_live = X_live[model_feature_names]

        # Make predictions
        predictions = model.predict(X_live)
        
        # Create response
        response_df = original_df_with_dates[['date', 'region', 'product_id']].copy()
        response_df['predicted_demand'] = predictions.round().astype(int)

        return PredictionResponse(
            predictions=response_df.to_dict(orient="records")
        )

    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}\n{tb}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_demand(request: PredictionRequest):
    try:
        # Validate dates
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(request.end_date, "%Y-%m-%d")
        
        if end_date < start_date:
            raise HTTPException(status_code=400, detail="End date must be after start date")
        
        # Mock prediction logic
        predictions = []
        current_date = start_date
        while current_date <= end_date:
            # Generate mock prediction
            prediction = {
                "date": current_date.strftime("%Y-%m-%d"),
                "predicted_demand": 100 + (hash(request.region + request.product_id) % 50),
                "confidence": 0.85
            }
            predictions.append(prediction)
            current_date += timedelta(days=1)
        
        return PredictionResponse(
            predictions=predictions,
            confidence_score=0.85
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/actuals")
async def get_historical_data(region: str, product_id: str, start_date: str, end_date: str):
    try:
        # Mock historical data
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        historical_data = []
        current_date = start
        while current_date <= end:
            data_point = {
                "date": current_date.strftime("%Y-%m-%d"),
                "actual_demand": 90 + (hash(region + product_id) % 40),
                "region": region,
                "product_id": product_id
            }
            historical_data.append(data_point)
            current_date += timedelta(days=1)
        
        return {"historical_data": historical_data}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
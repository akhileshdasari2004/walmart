# Walmart Predictive Demand Sensing System

A real-time demand sensing system that optimizes regional stock allocation through predictive analytics.

## 🎯 Problem Statement

This project addresses Walmart's need for real-time demand insights to prevent regional stock mismatches and optimize inventory allocation across physical stores and distribution centers.

## 🚀 Features

- Real-time demand forecasting by region and SKU
- Integration of multiple data sources (historical sales, weather, events)
- Interactive dashboard for visualization
- API endpoints for demand predictions
- Machine learning models for time-series forecasting

## 📁 Project Structure

```
predictive_demand_demo/
├── data/               # Data storage
├── notebooks/          # Jupyter notebooks for analysis
├── models/            # Trained ML models
├── backend/           # FastAPI backend
├── frontend/          # Streamlit dashboard
└── requirements.txt   # Python dependencies
```

## 🛠️ Setup & Installation

1. Create and activate virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the backend server:

```bash
cd backend
uvicorn app:app --reload
```

4. Run the frontend dashboard:

```bash
cd frontend
streamlit run dashboard.py
```

## 📊 Data Flow

1. Historical sales data is processed and cleaned
2. ML models are trained on processed data
3. Real-time data is ingested for predictions
4. Predictions are served via API
5. Dashboard visualizes results

## 🔄 API Endpoints

- `/predict`: Get demand predictions
- `/actuals`: Get historical data
- `/regions`: List available regions
- `/products`: List available products

## 📈 Business Impact

- Reduced stockouts
- Optimized inventory allocation
- Lower transportation costs
- Improved customer satisfaction

## 🤝 Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## 📝 License

This project is licensed under the MIT License - see the LICENSE.md file for details
# walmart

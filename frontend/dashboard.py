import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import json

# API Configuration
API_BASE_URL = "http://localhost:8005"

# Page config
st.set_page_config(
    page_title="Walmart Demand Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("Walmart Demand Prediction Dashboard")
st.markdown("""
This dashboard provides real-time demand predictions and historical data visualization
for Walmart's regional inventory management.
""")

# Sidebar controls
st.sidebar.header("Controls")

# File uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload Sales Data (CSV)",
    type="csv"
)

if uploaded_file is not None:
    if st.sidebar.button("Generate Predictions from Uploaded Data"):
        try:
            # Send file to backend for prediction
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            response = requests.post(f"{API_BASE_URL}/predict_from_file", files=files)
            
            if response.status_code == 200:
                predictions_data = response.json()
                predictions_df = pd.DataFrame(predictions_data["predictions"])
                
                # Display predictions
                st.subheader("Demand Predictions from File")
                fig = px.line(
                    predictions_df,
                    x="date",
                    y="predicted_demand",
                    title="Predicted Demand"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Download button
                csv = predictions_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions",
                    data=csv,
                    file_name="predictions_from_file.csv",
                    mime="text/csv"
                )
            else:
                st.error(f"Error from API: {response.json().get('detail')}")

        except Exception as e:
            st.error(f"Error processing file: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("### OR")
st.sidebar.markdown("---")

# Fetch available regions and products
try:
    regions_response = requests.get(f"{API_BASE_URL}/regions")
    products_response = requests.get(f"{API_BASE_URL}/products")
    
    regions = regions_response.json()["regions"]
    products = products_response.json()["products"]
except:
    st.error("Could not connect to the API. Please ensure the backend server is running.")
    st.stop()

# Selection controls
selected_region = st.sidebar.selectbox("Select Region", regions)
selected_product = st.sidebar.selectbox("Select Product", products)

# Date range selection
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        datetime.now() - timedelta(days=30)
    )
with col2:
    end_date = st.date_input(
        "End Date",
        datetime.now()
    )

# Prediction button
if st.sidebar.button("Generate Predictions"):
    try:
        # Prepare request data
        request_data = {
            "region": selected_region,
            "product_id": selected_product,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d")
        }
        
        # Get predictions
        predictions_response = requests.post(
            f"{API_BASE_URL}/predict",
            json=request_data
        )
        predictions_data = predictions_response.json()
        
        # Get historical data
        historical_response = requests.get(
            f"{API_BASE_URL}/actuals",
            params=request_data
        )
        historical_data = historical_response.json()
        
        # Convert to DataFrames
        predictions_df = pd.DataFrame(predictions_data["predictions"])
        historical_df = pd.DataFrame(historical_data["historical_data"])
        
        # Create visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Demand Predictions")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=historical_df["date"],
                y=historical_df["actual_demand"],
                name="Historical Demand",
                line=dict(color="blue")
            ))
            fig.add_trace(go.Scatter(
                x=predictions_df["date"],
                y=predictions_df["predicted_demand"],
                name="Predicted Demand",
                line=dict(color="red", dash="dash")
            ))
            fig.update_layout(
                title="Historical vs Predicted Demand",
                xaxis_title="Date",
                yaxis_title="Demand",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Prediction Confidence")
            fig = px.bar(
                predictions_df,
                x="date",
                y="confidence",
                title="Prediction Confidence by Date"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Average Predicted Demand",
                f"{predictions_df['predicted_demand'].mean():.0f}"
            )
        with col2:
            st.metric(
                "Average Historical Demand",
                f"{historical_df['actual_demand'].mean():.0f}"
            )
        with col3:
            st.metric(
                "Prediction Confidence",
                f"{predictions_data['confidence_score']*100:.1f}%"
            )
        
        # Download button for predictions
        csv = predictions_df.to_csv(index=False)
        st.download_button(
            label="Download Predictions",
            data=csv,
            file_name=f"predictions_{selected_region}_{selected_product}.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Error generating predictions: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ for Walmart's Predictive Demand Sensing System") 
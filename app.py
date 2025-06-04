from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
import tensorflow as tf

app = Flask(__name__)

# Load data
data = pd.read_csv(r'C:\Users\dell\Desktop\electric Load Forecasting Clustering Web App\data\preprocessed_and_cleaned_data.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Get unique cities
cities = data['city'].unique().tolist()

# Model paths
MODEL_DIR = r'C:\Users\dell\Desktop\electric Load Forecasting Clustering Web App\model'
LINEAR_MODEL_PATH = os.path.join(MODEL_DIR, 'best_linear_regression.pkl')
XGBOOST_MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_model.pkl')
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, 'LSTM_model.h5')

# Load LSTM model
def load_lstm_model():
    try:
        import tensorflow as tf
        # Define custom_objects to handle the 'mse' function
        custom_objects = {
            'mse': tf.keras.losses.MeanSquaredError(),
            'mean_squared_error': tf.keras.losses.MeanSquaredError()
        }
        return tf.keras.models.load_model(LSTM_MODEL_PATH, custom_objects=custom_objects)
    except Exception as e:
        print(f"Error loading LSTM model: {e}")
        return None

# Load models
def load_model(model_path):
    try:
        return joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None

# Load SARIMA model for a specific city
def load_sarima_model(city):
    city_name = city.lower().replace(' ', '_')
    model_path = os.path.join(MODEL_DIR, f'sarima_{city_name}_daily.pkl')
    return load_model(model_path)

# Perform clustering
def perform_clustering(k=3):
    # Features for clustering
    features = ['temperature', 'humidity', 'windSpeed', 'pressure', 'dewPoint', 'demand_mwh']
    X = data[features]
    
    # Apply RobustScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-means
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.5)
    plt.title(f"Cluster Visualization (k={k})")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(label='Cluster')
    plt.tight_layout()
    
    # Convert plot to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    encoded = base64.b64encode(image_png).decode('utf-8')
    return encoded

# Generate forecast
def generate_forecast(city, start_date, end_date, model_type):
    # Filter data for the selected city
    city_data = data[data['city'] == city]
    
    if model_type == 'sarima':
        # Load SARIMA model for the selected city
        sarima_model = load_sarima_model(city)
        if sarima_model is None:
            return None, "SARIMA model not found for this city"
        
        # Convert to daily data
        daily_data = city_data.set_index('timestamp')['demand_mwh'].resample('D').mean()
        
        # Generate forecast
        forecast_steps = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
        forecast = sarima_model.get_forecast(steps=forecast_steps)
        forecast_mean = forecast.predicted_mean
        conf_int = forecast.conf_int()
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(daily_data.index, daily_data, label='Historical Data')
        plt.plot(forecast_mean.index, forecast_mean, label='Forecast', color='red')
        plt.fill_between(
            forecast_mean.index,
            conf_int.iloc[:, 0],
            conf_int.iloc[:, 1],
            color='red',
            alpha=0.2
        )
        plt.title(f'SARIMA Forecast for {city}')
        plt.xlabel('Date')
        plt.ylabel('Demand (MWh)')
        plt.legend()
        plt.grid(True)
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        encoded = base64.b64encode(image_png).decode('utf-8')
        return encoded, None
    
    elif model_type in ['linear', 'xgboost']:
        try:
            # Load the appropriate model
            if model_type == 'linear':
                model = load_model(LINEAR_MODEL_PATH)
                model_name = "Linear Regression"
            else:  # xgboost
                model = load_model(XGBOOST_MODEL_PATH)
                model_name = "XGBoost"
            
            if model is None:
                return None, f"{model_name} model not found"
            
            # Filter data for the date range
            mask = (city_data['timestamp'] >= start_date) & (city_data['timestamp'] <= end_date)
            filtered_data = city_data[mask]
            
            if filtered_data.empty:
                return None, "No data available for the selected date range"
            
            # Create plot
            plt.figure(figsize=(12, 6))
            plt.plot(filtered_data['timestamp'], filtered_data['demand_mwh'], label='Predicted Demand', color='red')
            plt.title(f'{model_name} Forecast for {city}', fontsize=14, fontweight='bold')
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Demand (MWh)', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Convert plot to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            plt.close()
            
            encoded = base64.b64encode(image_png).decode('utf-8')
            return encoded, None
        except Exception as e:
            return None, f"Error generating forecast with {model_type}: {str(e)}"
    
    elif model_type == 'lstm':
        try:
            # Load LSTM model
            lstm_model = load_lstm_model()
            if lstm_model is None:
                return None, "LSTM model could not be loaded"
            
            # Filter data for the date range
            mask = (city_data['timestamp'] >= start_date) & (city_data['timestamp'] <= end_date)
            filtered_data = city_data[mask]
            
            if filtered_data.empty:
                return None, "No data available for the selected date range"
            
            # Create plot
            plt.figure(figsize=(12, 6))
            plt.plot(filtered_data['timestamp'], filtered_data['demand_mwh'], label='Predicted Demand', color='red')
            plt.title(f'LSTM Forecast for {city}', fontsize=14, fontweight='bold')
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Demand (MWh)', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Convert plot to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            plt.close()
            
            encoded = base64.b64encode(image_png).decode('utf-8')
            return encoded, None
        except Exception as e:
            return None, f"Error generating forecast with LSTM: {str(e)}"
    
    else:
        return None, "Unsupported model type"

@app.route('/')
def index():
    # Get min and max dates from the dataset
    min_date = data['timestamp'].min().strftime('%Y-%m-%d')
    max_date = data['timestamp'].max().strftime('%Y-%m-%d')
    
    return render_template('index.html', 
                           cities=cities,
                           min_date=min_date,
                           max_date=max_date)

@app.route('/cluster', methods=['POST'])
def cluster():
    k = int(request.form.get('k', 3))
    cluster_image = perform_clustering(k)
    return jsonify({'image': cluster_image})

@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        city = request.form.get('city')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        model_type = request.form.get('model_type')
        
        forecast_image, error = generate_forecast(city, start_date, end_date, model_type)
        
        if error:
            return jsonify({'error': error})
        
        return jsonify({'image': forecast_image})
    except Exception as e:
        return jsonify({'error': f"Server error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)

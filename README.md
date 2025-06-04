# Electric-Load-Forecasting-Clustering-Web-App

## Overview

- Flask-based application
- Performs clustering on combined electricity‐demand and weather data
- Generates time‐series forecasts using:
  - Linear Regression
  - XGBoost
  - LSTM
  - SARIMA

## Demo Video

Watch a short demonstration of the app in action.

https://github.com/user-attachments/assets/e77c7800-3422-4979-aa4a-27ffe72c51a9


## Repository Structure

```
electric-load-forecasting/
│
├── README.md                       
├── app.py                         
├── tempCodeRunnerFile.py           
│
├── data_creation.ipynb             ← Notebook to fetch/merge raw JSON/CSV data
├── Preprocessing.ipynb             ← Notebook for data cleaning & feature engineering
├── model.ipynb                     ← Notebook for training models (Linear, XGBoost, LSTM, SARIMA)
│
├── data/                           ← small sample datasets or merged CSV
│   └── preprocessed_and_cleaned_data.csv
│
├── model/                          ← Trained model artifacts
│   ├── best_linear_regression.pkl
│   ├── linear_regression_model.pkl   
│   ├── xgb_encoder.pkl
│   ├── xgboost_model.pkl
│   ├── LSTM_model.h5
│   ├── sarima_dallas_daily.pkl
│   ├── sarima_houston_daily.pkl
│   ├── sarima_la_daily.pkl
│   ├── sarima_nyc_daily.pkl
│   ├── sarima_philadelphia_daily.pkl
│   ├── sarima_phoenix_daily.pkl
│   ├── sarima_san_antonio_daily.pkl
│   ├── sarima_san_diego_daily.pkl
│   ├── sarima_san_jose_daily.pkl
│   └── sarima_seattle_daily.pkl
│
├── static/                       
│   └── css/
│       └── style.css
│
├── templates/                      
│   └── index.html
              
```


## Prerequisites

- **Python 3.8+**  
- (Optional) **Node.js 12+** if you plan to extend the frontend build  
- Recommended: a virtual environment (e.g., `python3 -m venv venv`)


## Setup

1. **Install Python dependencies**  
   > Make sure you have a `requirements.txt` that includes at least:
   > ```
   > flask
   > pandas
   > numpy
   > scikit-learn
   > xgboost
   > tensorflow      # for LSTM
   > statsmodels     # for SARIMA
   > joblib
   > matplotlib
   > ```
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   > If `requirements.txt` is missing, you can generate it after installing packages:
   > ```bash
   > pip freeze > requirements.txt
   > ```

2. **Prepare data & models**  
   - If you already have `data/preprocessed_and_cleaned_data.csv`, skip to the next step. Otherwise, open and run the notebooks in order:
     1. **`data_creation.ipynb`**  
        - load raw weather JSONs and demand CSVs  
        - Merge them into one CSV → `data/preprocessed_and_cleaned_data.csv`
     2. **`Preprocessing.ipynb`**  
        - Clean missing values, engineer time features, detect anomalies  
        - Verify that `data/preprocessed_and_cleaned_data.csv` is properly formatted
     3. **`model.ipynb`**  
        - Train and save models:  
          - Linear Regression → `model/best_linear_regression.pkl`  
          - XGBoost → `model/xgboost_model.pkl` (+ `model/xgb_encoder.pkl` if needed)  
          - LSTM → `model/LSTM_model.h5`  
          - City-specific SARIMA → `model/sarima_<city>_daily.pkl`
   - Copy or download your trained `.pkl` and `.h5` files into the `model/` folder.  
     

## Running the Application

1. **Activate your virtual environment** (if not already active):
   ```bash
   # macOS/Linux
   source venv/bin/activate
   # Windows PowerShell
   .\venv\Scripts\Activate.ps1
   ```

2. **Set Flask environment variables**  
   - macOS/Linux:
     ```bash
     export FLASK_APP=app.py
     export FLASK_ENV=development  
     ```
   - Windows (PowerShell):
     ```powershell
     $env:FLASK_APP = "app.py"
     $env:FLASK_ENV = "development"
     ```

3. **Start the Flask server**  
   ```bash
   flask run --host=0.0.0.0 --port=5000
   ```
   > You should see:
   > ```
   >  * Serving Flask app "app.py"
   >  * Environment: development
   >  * Debug mode: on
   >  * Running on http://127.0.0.1:5000/
   > ```

4. **Open your browser** at [http://localhost:5000] 
   - **Generate Forecast**:  
     1. Select a **City** (populated from `data/preprocessed_and_cleaned_data.csv`)  
     2. Choose **Start Date** / **End Date** (min/max derived from the dataset)  
     3. Pick a **Model Type**:  
        - `linear` → Linear Regression (hourly)  
        - `xgboost` → XGBoost (hourly)  
        - `lstm` → LSTM (hourly)  
        - `sarima` → SARIMA (daily)  
     4. Click **Generate Forecast** → the resulting plot appears below the form.  
   - **Generate Clusters**:  
     1. Enter **k** (number of clusters, e.g., 3)  
     2. Click **Generate Clusters** → a PCA scatter plot colored by cluster labels appears.



## Application Details

### 1. `app.py`

- **Loads**  
  - `data/preprocessed_and_cleaned_data.csv` → `data` (pandas DataFrame)  
  - Trained models from `model/`:
    - `best_linear_regression.pkl`
    - `xgboost_model.pkl`
    - `LSTM_model.h5`
    - `sarima_<city>_daily.pkl` (one per city)

- **Routes & Endpoints**  
  - **`GET /`** → Renders `templates/index.html`  
    - Passes:  
      - `cities` – list of unique city names  
      - `min_date`, `max_date` – for date picker bounds  
  - **`POST /cluster`**  
    - Form data: `{ k: <int> }`  
    - Steps:  
      1. Extract features: `temperature, humidity, windSpeed, pressure, dewPoint, demand_mwh`  
      2. Scale with `RobustScaler`  
      3. Apply `KMeans(n_clusters=k)` → cluster labels  
      4. PCA → 2 components for visualization  
      5. Generate matplotlib scatter plot → base64 PNG → return JSON `{ "image": "<base64>" }`  
  - **`POST /forecast`**  
    - Form data:  
      ```json
      {
        "city": "<city name>",
        "start_date": "YYYY-MM-DD",
        "end_date": "YYYY-MM-DD",
        "model_type": "linear" | "xgboost" | "lstm" | "sarima"
      }
      ```
    - Steps (per `model_type`):  
      - **`sarima`**:  
        1. Load `model/sarima_<city>_daily.pkl`  
        2. Resample city’s `demand_mwh` to daily means  
        3. Forecast `n_days = (end_date – start_date) + 1`  
        4. Plot historical daily vs. forecast with confidence intervals → base64 PNG  
      - **`linear` / `xgboost`**:  
        1. Load `best_linear_regression.pkl` or `xgboost_model.pkl`  
        2. Filter hourly data for `city` & `[start_date, end_date]`  
        3. Plot predicted `demand_mwh` (red line) over time → base64 PNG  
      - **`lstm`**:  
        1. Load `LSTM_model.h5` (with custom_objects for `mse`)  
        2. Filter hourly data for `city` & `[start_date, end_date]`  
        3. Plot predicted `demand_mwh` (red line) over time → base64 PNG  
  

### 2. `templates/index.html`

- Electric Load Forecast & Clustering lets you quickly visualize demand patterns and generate future load forecasts: simply choose a city, date range, and model (Linear, XGBoost, LSTM, or SARIMA) to see a demand forecast, or enter a “k” value to view hourly demand clusters. Just pick your options and click “Generate.”

### 3. `static/css/style.css`

- Custom CSS defines the visual look and feel of the entire application. It includes styling for form elements (inputs, buttons, labels), layout containers (margins, paddings, grid structure), and typography (font families, sizes, colors) to ensure a clean, consistent user interface.


## Notebooks & Data Preparation

1. **`data_creation.ipynb`**  
   - load raw weather JSON files and demand CSV files.  
   - Merge them on `timestamp` & `city`.  
   - Output: `data/preprocessed_and_cleaned_data.csv`  

2. **`Preprocessing.ipynb`**  
   - Load `data/preprocessed_and_cleaned_data.csv`  
   - Clean missing values, engineer time features (`hour_of_day`, `day_of_week`, etc.), detect/outlier removal via `IsolationForest`.  
   - Save final cleaned CSV back to `data/preprocessed_and_cleaned_data.csv`.  

3. **`model.ipynb`**  
   - Load cleaned data, split per city/time subset.  
   - Train & tune models:  
     - **Linear Regression** → save `model/best_linear_regression.pkl`  
     - **XGBoost** → save `model/xgboost_model.pkl` (+ `xgb_encoder.pkl`)  
     - **LSTM** → Keras/TensorFlow, save `model/LSTM_model.h5`  
     - **SARIMA** → one model per city, save `model/sarima_<city>_daily.pkl`  


## Future Enhancements

- **Model Improvements**  
  - Hyperparameter tuning (GridSearch, Bayesian optimization)  
  - Ensemble averaging (e.g., averaging XGBoost + LSTM predictions)  
  - Experiment with Transformer-based time-series models  

- **Data Enrichment**  
  - Add additional weather variables (e.g., precipitation, solar radiation)  
  - Incorporate holiday/socioeconomic indicators  

- **Frontend Upgrades**  
  - Replace static `<img>` with interactive Chart.js or Plotly visualizations  
  - Add a real-time data feed (e.g., Kafka → Spark Streaming)  
  - Improve mobile responsiveness and UI styling  


## Acknowledgments
 
- Built using: Flask, pandas, scikit-learn, XGBoost, TensorFlow/Keras, statsmodels, and matplotlib.


## Contributors
 - Rafia Khan
 - M.Tashfeen Abbasi
 - Laiba Mazhar


## Contact
  For queries or contributions, please contact: Rafia Khan                                                                                                   
  Email: rafiah.khan18@gmail.com


Thank you!

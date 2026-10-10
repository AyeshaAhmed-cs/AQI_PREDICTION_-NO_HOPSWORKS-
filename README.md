# 🌍 Karachi AQI Intelligence System
### **Comparative AI Forecasting: Statistical vs. Deep Learning Approaches**

**Developed by:** ✨ Ayesha Ahmed ✨  
**Status:** 🟢 System Online | **Architecture:** Local Offline-First Pipeline  
**Live Dashboard:** [🚀 View Karachi AQI Dashboard](https://aqi-prediction-nohopsworks-kdvkbnnitpeqihagrpz98k.streamlit.app/)  

---

## 📌 Project Overview
This repository hosts a production-grade machine learning pipeline designed to predict the Air Quality Index (AQI) for Karachi, Pakistan. The system leverages automated local data engineering, local model artifact persistence, and an interactive Streamlit dashboard to provide actionable environmental insights for the general public, fully decoupled from external cloud platforms for maximum stability and portability.

---

## 🔬 Advanced Analytics Features
This system implements a robust data science framework designed for multi-step atmospheric forecasting:

* **Exploratory Data Analysis (EDA):** Interactive multivariate correlation studies analyzing the interplay between humidity, atmospheric pressure, temperature, and AQI levels.
* **Model Explainability:** Integrated feature importance tracking and SHAP integration to explain AI decision-making (e.g., how wind speed and particulate concentrations affect forecasts).
* **Public Health Alerts:** An automated hazard detection system that triggers critical warnings for AQI spikes exceeding health safety thresholds.
* **Multi-Model Support:** Architecture designed to evaluate and support multiple regression algorithms (Ridge, Random Forest, Gradient Boosting, and ANNs) via a decoupled local inference pipeline.

---

## 🛠️ System Architecture
The project is organized into four primary local pipelines:

### 1. 🔄 Backfill Pipeline (`backfill_pipeline/`)
* **Historical Grounding:** Ingests and cleanses historical AQI and weather data to create a robust baseline, storing records locally in `karachi_aqi_data.csv`.
* **Cold Start:** Ensures sufficient history is available to calculate rolling averages and trends from day one.
* **Data Integrity:** Implements outlier detection and handling to remove faulty sensor readings from the dataset.

### 2. 🏗️ Feature Pipeline (`feature_pipeline/`)
* **Source:** Real-time atmospheric data ingestion from Open-Meteo & Air Quality APIs.
* **Logic:** Fetches and prepares atmospheric data hourly, computing rolling lags and temporal features.
* **Persistence:** Appends processed features directly to the local offline dataset (`karachi_aqi_data.csv`).

### 3. 🧠 Training Pipeline (`training_pipeline/`)
* **Training:** Automatically trains and benchmarks ML models locally on the dataset.
* **Persistence:** Saves optimized model binaries, scalers, imputers, and metadata directly to the `model_artifact/` folder.
* **Optimization:** Implements feature scaling and missing value imputation for high-accuracy forecasting.

### 4. 📈 Inference & Dashboard (`inference_pipeline/app.py`)
* **Real-time Prediction:** Loads local artifacts to run recursive multi-step forecasting for the next 72 hours.
* **UX/UI:** A premium Streamlit dashboard featuring:
    * **Health Advisory:** Plain-English clinical recommendations and hazard analysis.
    * **Expert EDA:** Heatmaps and multivariate statistical analysis.
    * **Hazard Monitor:** A dedicated visual tracker for identifying dangerous pollution windows.

---

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone [https://github.com/AyeshaAhmed-cs/aqi_prediction_-no_hopsworks](https://github.com/AyeshaAhmed-cs/aqi_prediction_-no_hopsworks)
cd aqi_prediction_-no_hopsworks
2. Install dependenciesBashpip install -r requirements.txt
3. Run the Local Pipeline SequenceExecute the scripts in order to populate data, train models locally, and launch the UI:Bash# Step 1: Generate historical baseline data
python backfill_pipeline/backfill_historical.py

# Step 2: Train models and save artifacts locally
python training_pipeline/train_model.py

# Step 3: Launch the Streamlit dashboard
streamlit run inference_pipeline/app.py
🤖 Model Benchmark & SelectionThe system automatically compares multiple algorithms during training and selects the top-performing model based on $R^2$ Score and MAE.Evaluated models include:Ridge RegressionRandom Forest RegressionGradient BoostingArtificial Neural Networks (ANNs)"Providing data-driven clarity for a cleaner, safer Karachi."

# ================= training_pipeline/train_model.py =================
import os
import shutil
import time
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# TensorFlow for ANN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# --- 1. SETUP ---
load_dotenv()
LOCAL_DATA_PATH = "karachi_aqi_data.csv"

# --- 2. FETCH DATA LOCALLY ---
print("📡 Loading local dataset...")
if not os.path.exists(LOCAL_DATA_PATH):
    raise FileNotFoundError(f"❌ Could not find {LOCAL_DATA_PATH}. Please run your backfill script first!")

df = pd.read_csv(LOCAL_DATA_PATH)

# --- 3. DATA PREPARATION (The NaN-Proof Version) ---
training_df = df.select_dtypes(include=[np.number]).dropna(subset=["aqi"])

X = training_df.drop(columns=["aqi"])
y = training_df["aqi"]

# 🛠️ Fill missing values in Features (X) BEFORE scaling
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Scaling for Ridge and ANN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 4. EXPERIMENT WITH VARIOUS ML MODELS ---

# A. Ridge Regression
print("📈 Training Ridge Regression...")
ridge = Ridge(alpha=1.0).fit(X_train_scaled, y_train)

# B. Ensemble Models
print("🌲 Training Ensemble Models...")
rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
gb = GradientBoostingRegressor(random_state=42).fit(X_train, y_train)

# C. Artificial Neural Network (ANN)
print("🧠 Training ANN (TensorFlow)...")
def build_ann(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1) 
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

ann = build_ann(X_train.shape[1])
ann.fit(X_train_scaled, y_train, epochs=50, batch_size=8, verbose=0)

# --- 5. EVALUATE PERFORMANCE ---
models = {
    "Ridge": (ridge.predict(X_test_scaled), ridge),
    "RandomForest": (rf.predict(X_test), rf),
    "GradientBoosting": (gb.predict(X_test), gb),
    "ANN": (ann.predict(X_test_scaled).flatten(), ann)
}

results = {}
for name, (preds, model_obj) in models.items():
    results[name] = {
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "MAE": mean_absolute_error(y_test, preds),
        "R2": r2_score(y_test, preds),
        "object": model_obj
    }
    print(f"📊 {name} -> RMSE: {results[name]['RMSE']:.2f}, R2: {results[name]['R2']:.2f}")

# --- 6. STORE TRAINED MODEL LOCALLY ---
best_name = min(results, key=lambda k: results[k]["RMSE"])
print(f"🏆 Best Performing Model: {best_name}")

artifact_path = "model_artifact"

# 1. SMARTER CLEANUP: Avoids Permission Errors if app.py is running
if not os.path.exists(artifact_path):
    os.makedirs(artifact_path)
else:
    for filename in os.listdir(artifact_path):
        file_path = os.path.join(artifact_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"⚠️ Note: {filename} is busy, overwriting instead of deleting.")

# 2. SAVE THE WINNER'S NAME
with open(os.path.join(artifact_path, "model_name.txt"), "w") as f:
    f.write(best_name)

# 3. SAVE THE BEST MODEL OBJECT
if best_name == "ANN":
    results[best_name]["object"].save(os.path.join(artifact_path, "saved_model"))
else:
    joblib.dump(results[best_name]["object"], os.path.join(artifact_path, "model.pkl"))

# 4. SAVE PREPROCESSING TOOLS
joblib.dump(scaler, os.path.join(artifact_path, "scaler.pkl"))
joblib.dump(imputer, os.path.join(artifact_path, "imputer.pkl"))

print(f"🔥 SUCCESS: {best_name} trained and saved locally to {artifact_path}/!")
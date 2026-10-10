# ================= backfill_pipeline/backfill_historical.py =================
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import requests

load_dotenv()

LOCAL_DATA_PATH = "karachi_aqi_data.csv"

# --- 1. FETCH REAL DATA ---
def fetch_historical_data():
    LAT, LON = 24.8607, 67.0011
    
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=31)).strftime('%Y-%m-%d')
    
    weather_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={LAT}&longitude={LON}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl"
    aqi_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={LAT}&longitude={LON}&start_date={start_date}&end_date={end_date}&hourly=pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone"
    print(f"📅 Fetching from {start_date} to {end_date}...")

    weather_resp = requests.get(weather_url).json()
    aqi_resp = requests.get(aqi_url).json()

    if 'hourly' not in weather_resp:
        print(f"❌ Weather API Error: {weather_resp.get('reason', 'Unknown error')}")
        return pd.DataFrame() 
        
    df = pd.merge(pd.DataFrame(weather_resp['hourly']), pd.DataFrame(aqi_resp['hourly']), on="time")
    df['time'] = pd.to_datetime(df['time'])
    
    # Match schema columns
    df = df.rename(columns={
        "time": "timestamp", 
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "wind_speed_10m": "wind_speed",
        "pressure_msl": "pressure",
        "pm2_5": "pm25",
        "nitrogen_dioxide": "no2", 
        "sulphur_dioxide": "so2",
        "carbon_monoxide": "co", 
        "ozone": "o3"
    })
    
    df['aqi'] = (df['pm25'] * 1.5 + df['pm10'] * 0.5)
    df['city'] = "Karachi"
    return df

# --- 2. ENGINEER MISSING FEATURES ---
def engineer_backfill(df):
    df = df.sort_values("timestamp")
    df = df.ffill().bfill() 
    
    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    
    df["aqi_change"] = df["aqi"].diff().fillna(0.0)
    df["pm25_change_rate"] = df["pm25"].diff().fillna(0.0)
    df["rolling_avg_aqi_24h"] = df["aqi"].rolling(24, min_periods=1).mean().fillna(0.0)
    df["rolling_avg_pm25_24h"] = df["pm25"].rolling(24, min_periods=1).mean().fillna(0.0)
    
    cols_to_fix = ["aqi", "pm25", "temperature", "humidity", "pressure", 
                   "hour", "day", "month", "day_of_week", "is_weekend"]
    
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = df[col].round().astype('int64')

    return df

# --- 3. SAVE LOCALLY ---
if __name__ == "__main__":
    print("🚀 Fetching and Engineering REAL data locally...")
    df = fetch_historical_data()
    if not df.empty:
        df = engineer_backfill(df)

        if 'precipitation' in df.columns:
            df = df.drop(columns=['precipitation'])
        
        # Save or update local CSV file
        df.to_csv(LOCAL_DATA_PATH, index=False)
        print(f"✅ MISSION SUCCESS! 30 days of historical data saved locally to {LOCAL_DATA_PATH}.")
    else:
        print("❌ Backfill failed because no data was returned.")
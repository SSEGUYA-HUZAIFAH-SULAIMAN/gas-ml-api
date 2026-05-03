from fastapi import FastAPI
import joblib
import numpy as np


app = FastAPI()

# Load trained model
model = joblib.load("gas_model.pkl")

# Constants for gas cylinder
TARE_WEIGHT = 9.0

@app.get("/")
def home():
    return {"message": "Gas ML API is running"}

@app.post("/predict")
def predict(data: dict):
    try:
       
        weight = data["weight"]
        day_of_week = data["day_of_week"]
        hour_of_day = data["hour_of_day"]
        days_since_last_refill = data["days_since_last_refill"]
        weight_change = data["weight_change"]

        # Calculate net gas weight (current weight - empty cylinder weight)
        net_weight = max(0, weight - TARE_WEIGHT)
        
        # Use net_weight as the first feature for the model
        X = np.array([[net_weight, day_of_week, hour_of_day,
                       days_since_last_refill, weight_change]])

        predicted_usage = model.predict(X)[0]

        days_remaining = net_weight / predicted_usage if predicted_usage > 0 and net_weight > 0 else 0

        return {
            "predicted_usage_per_day": float(predicted_usage),
            "days_until_refill": round(float(days_remaining), 1)
        }

    except Exception as e:
        return {"error": str(e)}
    
import os
port = int(os.environ.get("PORT", 8000))
from fastapi import FastAPI
import joblib
import numpy as np


app = FastAPI()

# Load trained model
model = joblib.load("gas_model.pkl")

@app.get("/")
def home():
    return {"message": "Gas ML API is running"}

@app.post("/predict")
def predict(data: dict):
    try:
        # Extract features
        weight = data["weight"]
        day_of_week = data["day_of_week"]
        hour_of_day = data["hour_of_day"]
        days_since_last_refill = data["days_since_last_refill"]
        weight_change = data["weight_change"]

        # Prepare input
        X = np.array([[weight, day_of_week, hour_of_day,
                       days_since_last_refill, weight_change]])

        # Predict usage
        predicted_usage = model.predict(X)[0]

        return {
            "predicted_usage_per_day": float(predicted_usage)
        }

    except Exception as e:
        return {"error": str(e)}
    
import os
port = int(os.environ.get("PORT", 8000))
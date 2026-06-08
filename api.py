from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

# --------------------------------------------------
# FastAPI App
# --------------------------------------------------

app = FastAPI()

# --------------------------------------------------
# Load Model
# --------------------------------------------------

model = joblib.load("gas_model.pkl")

# --------------------------------------------------
# Request Model
# --------------------------------------------------

class PredictionRequest(BaseModel):
    weight: float
    day_of_week: int
    days_since_last_refill: int
    weight_change: float
    rolling_avg_usage: float
    month: int
    is_weekend: int

# --------------------------------------------------
# Home Route
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Gas ML API is running"
    }

# --------------------------------------------------
# Prediction Route
# --------------------------------------------------

@app.post("/predict")
def predict(data: PredictionRequest):

    try:

        X = np.array([[
            data.weight,
            data.day_of_week,
            data.days_since_last_refill,
            data.weight_change,
            data.rolling_avg_usage,
            data.month,
            data.is_weekend
        ]])

        prediction = model.predict(X)[0]

        days_remaining = max(0, round(float(prediction), 1))

        return {
            "days_remaining": days_remaining
        }

    except Exception as e:

        return {
            "error": str(e)
        }

# --------------------------------------------------
# Render Port
# --------------------------------------------------

port = int(os.environ.get("PORT", 8000))
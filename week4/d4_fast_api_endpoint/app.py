from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "kmeans.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "scaler.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = joblib.load(MODEL_PATH)

    scaler = joblib.load(SCALER_PATH)

    print("Model loaded successfully.")

except Exception as e:

    raise RuntimeError(
        f"Could not load model or scaler: {e}"
    )


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Airtel-Inspired Customer Segmentation API",
    description=(
        "K-Means clustering API for telecom "
        "customer segmentation."
    ),
    version="1.0.0"
)


# ============================================================
# INPUT SCHEMA
# ============================================================

class CustomerInput(BaseModel):

    tenure: float = Field(
        ...,
        ge=0,
        le=100,
        description="Customer tenure in months"
    )

    MonthlyCharges: float = Field(
        ...,
        ge=0,
        description="Monthly customer charges"
    )

    TotalCharges: float = Field(
        ...,
        ge=0,
        description="Total customer charges"
    )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Airtel-inspired Customer Segmentation API",
        "status": "running",
        "docs": "/docs"
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(customer: CustomerInput):

    try:

        # Create feature array
        input_data = np.array([
            [
                customer.tenure,
                customer.MonthlyCharges,
                customer.TotalCharges
            ]
        ])

        # Apply SAME scaler used during training
        scaled_data = scaler.transform(
            input_data
        )

        # Predict cluster
        cluster = model.predict(
            scaled_data
        )[0]

        # Distance from cluster centers
        distances = model.transform(
            scaled_data
        )[0]

        return {
            "cluster": int(cluster),
            "distances_to_clusters": [
                round(float(distance), 4)
                for distance in distances
            ]
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
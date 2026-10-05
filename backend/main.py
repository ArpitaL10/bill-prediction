from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

app = FastAPI(
    title="Household Electricity Bill Prediction API",
    description="Predicts a household's estimated monthly electricity bill.",
    version="1.0.0",
)

# Streamlit runs in a separate local process. Restrict origins further before deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

FEATURES = [
    "num_rooms", "num_people", "housearea", "is_ac", "is_tv",
    "is_flat", "ave_monthly_income", "num_children", "is_urban"
]

class HouseholdInput(BaseModel):
    num_rooms: int = Field(ge=1, le=100, description="Number of rooms")
    num_people: int = Field(ge=1, le=100, description="People living in the home")
    housearea: float = Field(gt=0, le=1_000_000, description="House area in the same unit used for training")
    is_ac: int = Field(ge=0, le=1, description="1 if an air conditioner is present, otherwise 0")
    is_tv: int = Field(ge=0, le=1, description="1 if a television is present, otherwise 0")
    is_flat: int = Field(ge=0, le=1, description="1 if the home is a flat, otherwise 0")
    ave_monthly_income: float = Field(ge=0, le=1_000_000_000, description="Average monthly household income")
    num_children: int = Field(ge=0, le=100, description="Number of children")
    is_urban: int = Field(ge=0, le=1, description="1 if urban, otherwise 0")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}

@app.post("/predict")
def predict(payload: HouseholdInput):
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Prediction model not found. Save your trained model as backend/model.pkl first."
        )
    try:
        model = joblib.load(MODEL_PATH)
        row = [[getattr(payload, feature) for feature in FEATURES]]
        prediction = float(model.predict(row)[0])
        if not np.isfinite(prediction):
            raise ValueError("The model returned a non-finite value.")
        return {
            "predicted_amount": round(max(0.0, prediction), 2),
            "currency": "INR",
            "unit": "per month",
            "note": "This is an estimate from the trained model, not an official utility bill."
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

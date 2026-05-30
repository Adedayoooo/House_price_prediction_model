from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import logging
import numpy as np

from .schemas import HouseFeatures, PredictionResponse
from .utils import load_model, preprocess_input

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="House Price Prediction API",
    description="Professional API for predicting house prices",
    version="1.0.0"
)

app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"],)

model = None
scaler = None
encoders = None

def get_model():
    global model, scaler, encoders
    if model is None:
        model, scaler, encoders = load_model()
    return model, scaler, encoders

@app.get("/")
async def health_check():
    return {"status": "healthy", "service": "House Price Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(
    features: HouseFeatures,
    artifacts = Depends(get_model)
):
    try:
        model, scaler, encoders = artifacts
        input_data = features.model_dump()
        X_input = preprocess_input(input_data, encoders)
        
        if scaler:
            X_input = scaler.transform(X_input)
        
        prediction_log = model.predict(X_input)[0]
        predicted_price = np.expm1(prediction_log)
        
        return PredictionResponse(predicted_price=float(predicted_price),formatted_price=f"${predicted_price:,.2f}")
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error")

@app.get("/health")
async def detailed_health(artifacts = Depends(get_model)):
    return {
        "status": "ok",
        "model_type": type(artifacts[0]).__name__,
        "model_loaded": True
      }

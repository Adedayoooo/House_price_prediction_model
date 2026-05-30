import pickle
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/model.pkl")
REPO_ID = "Adedayoooo/house-price-prediction-model"

def load_model():
    try:
        if not MODEL_PATH.exists():
            logger.info("Downloading model from Hugging Face...")
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            
            hf_hub_download(repo_id=REPO_ID,filename="model.pkl",local_dir=MODEL_PATH.parent)
        
        with open(MODEL_PATH, "rb") as f:
            artifacts = pickle.load(f)
        
        logger.info("Model loaded successfully")
        return (artifacts["model"],artifacts.get("scaler"),artifacts.get("encoders", {}))
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def preprocess_input(features: dict, encoders: dict):
    df = pd.DataFrame([features])
    
    if "Neighborhood" in df.columns and "Neighborhood" in encoders:
        try:
            df["Neighborhood"] =encoders["Neighborhood"].transform(df["Neighborhood"].astype(str))
        except:
            pass
    
    return df.values

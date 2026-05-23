import numpy as np 
import pandas as pd
import logging 
from pathlib import Path
import pickle
import json
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from src.preprocessing import main as run_preprocessing
from src.config import DATA_PATH

logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger=logging.getLogger(__name__)

CONFIG=json.loads(Path("src/CONFIG.json").read_text())

house_features,target=run_preprocessing(DATA_PATH)
    
def training(X:pd.DataFrame,y:pd.Series):
  try:
    logger.info("Setting up KFold with 5 splits...")
    cv=KFold(n_splits=CONFIG["validation"]["n_splits"],shuffle=True,random_state=CONFIG["model_params"]["random_state"])
        
    model=KNeighborsRegressor(**model_params)
        
    logger.info("Running cross-validation...")
    cv_scores=cross_val_score(model,X,y,cv=cv,scoring='r2')
    logger.info(f"CV R2 scores: {cv_scores}")
    logger.info(f"Mean CV R2: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
    logger.info("Training final model on full data...")
    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(X)
    model.fit(X_scaled,y)
    logger.info("Model trained")
    return model,scaler
  except Exception as e:
    logger.error(f"An error occurred:{e}")
    raise
  
def save_model(model):
  try:
    logger.info("Saving best model...")
    with open("knn_based_house_price_model.pkl","wb") as f:
      pickle.dump(model,f)
    logger.info("Model saved as knn_model.pkl")
  except Exception as e:
    logger.error(f"An error occurred:{e}")
    raise

if __name__=="__main__":
  best_model,scaler=training(house_features,target)
  save_model(best_model)
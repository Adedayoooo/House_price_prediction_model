import logging 
from pathlib import Path
import pickle
import numpy as np 

from src.train import main as run_train
run_train()

logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger=logging.getLogger(__name__)

def load_model():
  try:
    root_directory=Path(__file__).parent.parent
    model_path=root_directory/"model"/"rf_house_price_model.pkl"
    logger.info(f"Loading model from {model_path}")
    if model_path.exists():
      with model_path.open('rb') as f:
        model=pickle.load(f)
    else:
      raise FileNotFoundError(f"Model not found at {model_path}")
    return model
  except Exception as e:
    logger.error(f"An error occurred:{e}")
    raise

def predict(model:knn_based_house_price_model.pkl,new_data:np.ndarray)->tuple[int,float]:
  try:
    logger.info("Predicting...")
    new_data=np.ndarray()
    X_scaled=scaler.fit(new_data)
    X_reshaped=X_scaled.reshape(1,-1)
    prediction=model.predict(X_reshaped)[0]
    probabilities=model.predict_proba(X_reshaped)[0]
    logger.info(f"Predicted house price is: {prediction}")
    logger.info(f"Confidence level: {probabilities[0]*100:.2f}%")
    return prediction,probabilities
    
  except Exception as e:
    logger.error(f"An error occurred:{e}")
    raise

if __name__=="__main__":
  model=load_model()
  new_data=np.ndarray()
  house_price,confidence_level=predict(model,new_data)
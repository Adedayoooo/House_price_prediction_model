import numpy as np 
import pandas as pd
import logging 
from pathlib import Path
import pickle
import json
from sklearn.model_selection import RandomizedSearchCV,cross_val_score,train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint, uniform

from src.preprocessing import main as run_preprocessing
from src.config import DATA_PATH
from src.metrics import regression_metrics  

logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger=logging.getLogger(__name__)

CONFIG=json.loads(Path("src/CONFIG.json").read_text())

house_features,target=run_preprocessing(DATA_PATH)
    
def training(X:pd.DataFrame,y:pd.Series):
  try:
    param_dist = {
        "n_estimators":randint(50,500),
        "max_depth":[None,10,20,30],
        "min_samples_split":randint(2,12),
        "max_features":uniform(0.5,0.5)
    }
        
    rf=RandomForestRegressor(random_state=CONFIG["random_search_cv_params"]["random_state"])
    
    logger.info("Carrying out a train_test_split...")
    X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.2,random_state=CONFIG["random_search_cv_params"]["random_state"]
    )
    
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    
    logger.info("Setting up RandomizedSearchCV with 5 splits...")
    random_search=RandomizedSearchCV(
      estimator=rf,
      param_distributions=param_dist,
      n_iter=CONFIG["random_search_cv_params"]["n_iter"],
      cv=CONFIG["random_search_cv_params"]["cv"],
      scoring=CONFIG["random_search_cv_params"]["scoring"],
      n_jobs=CONFIG["random_search_cv_params"]["n_jobs"],
      random_state=CONFIG["random_search_cv_params"]["random_state"],
      verbose=CONFIG["random_search_cv_params"]["verbose"]
    )
    
    logger.info("Training model...")
    random_search.fit(X_train_scaled,y_train)
    logger.info("Model trained")
    
    best_params=random_search.best_params_
    logger.info(f"Best hyper parameters combination: {best_params}")
    
    best_score=random_search.best_score_ 
    logger.info(f"Mean cross-validated score of the best estimator: {best_score}")
    
    best_model=random_search.best_estimator_
    
    y_pred = best_model.predict(X_test_scaled)
    logger.info("Evaluating on test set...")
    mse, mae, rmse, r2 = regression_metrics(y_test, y_pred)
    
    return best_model, scaler
    
  except Exception as e:
    logger.error(f"An error occurred:{e}")
    raise
  
def save_model(best_model):
  try:
    logger.info("Saving best model...")
    with open("rf_house_price_model.pkl","wb") as f:
      pickle.dump(best_model,f)
    logger.info("Model saved as rf_house_price_model.pkl")
  except Exception as e:
    logger.error(f"An error occurred:{e}")
    raise

if __name__=="__main__":
  best_model,scaler=training(house_features,target)
  save_model(best_model)
import pandas as pd
import logging 
from pathlib import Path
import pickle
import json
from sklearn.model_selection import RandomizedSearchCV,train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint,uniform

from src.preprocessing import main as run_preprocessing
from src.config import DATA_PATH
from src.metrics import regression_metrics  

logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger=logging.getLogger(__name__)

CONFIG=json.loads(Path("src/CONFIG.json").read_text())

house_features,target=run_preprocessing(DATA_PATH)
    
def initial_training(house_features:pd.DataFrame,target:pd.Series):
  try:
    param_dist = {
        "n_estimators":randint(50,500),
        "max_depth":[None,10,20,30],
        "min_samples_split":randint(2,12),
        "max_features":uniform(0.5,0.5)
    }
        
    rf=RandomForestRegressor(random_state=CONFIG["random_search_cv_params"]["random_state"])
    
    X,y=house_features,target
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
    return scaler,random_search,X_train,X_test_scaled,y_test
  except Exception as e:
    logger.error(f"An error occurred:{e}")
    raise

def feature_importance(random_search,X_train:pd.DataFrame):
  try:
    logger.info("Checking feature importance...")
    training_best_model=random_search.best_estimator_
    importance=training_best_model.feature_importances_
    feature_names = X_train.columns if hasattr(X_train, 'columns') else [f"feature_{i}" for i in range(len(importance))]
    
    importance_df=pd.DataFrame({
      'feature': feature_names,
      'importance': importance
    }).sort_values('importance', ascending=False)
    
    logger.info("\nMost important features in descending order")
    for idx, row in importance_df.head(10).iterrows():
        logger.info(f"{row['feature']:25} : {row['importance']:.5f}")
    
    irrelevant_features=importance_df.loc[importance_df['importance']==0,'feature'].to_list()
    if irrelevant_features:
      logger.info(f"The following features act as noise to the model, and are hence removed:\n{irrelevant_features}")
    else:
      logger.info("No zero importance feature found")
    
    relevant_features=importance_df.loc[importance_df['importance']>0,'feature'].to_list()
    logger.info(f"The following features are the relevant features: {relevant_features}")
    return relevant_features
  except Exception as e:
    logger.error(f"An error occurred:{e}")
    raise

def retrain(house_features:pd.DataFrame,target:pd.Series,relevant_features:list): 
  try:
    logger.info("Retraining model based using the relevant features...")
    X=house_features[relevant_features]
    y=target
    
    X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.2,random_state=CONFIG["random_search_cv_params"]["random_state"])
    
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    
    best_model=RandomForestRegressor(**random_search.best_params_,random_state=CONFIG["random_search_cv_params"]["random_state"])
    
    best_model.fit(X_train_scaled,y_train)
    best_params=random_search.best_params_
    logger.info(f"Best hyper parameters combination: {best_params}")
    
    best_score=random_search.best_score_ 
    logger.info(f"Mean cross-validated score of the best estimator: {best_score}")
    
    y_pred = best_model.predict(X_test_scaled)
    logger.info("Evaluating on test set...")
    mse,mae,rmse,r2=regression_metrics(y_test, y_pred)
    logger.info(f"""MSE:{mse:.4f}
    MAE:{mae:.4f}
    RMSE:{rmse:.4f}
    R2:{r2:.4f}""")
    return scaler,best_model
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
  scaler,random_search,X_train,X_test_scaled,y_test=initial_training(house_features,target)
  relevant_features=feature_importance(random_search,X_train)
  retraining_scaler,retraining_best_model=retrain(house_features,target,relevant_features)
  save_model(retraining_best_model)

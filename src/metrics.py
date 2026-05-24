import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import logging 

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def regression_metrics(y_true,y_pred):
    try:
        y_true=np.asarray(y_true)
        y_pred=np.asarray(y_pred)
        mse=mean_squared_error(y_true,y_pred)
        mae=mean_absolute_error(y_true,y_pred)
        rmse=np.sqrt(mse)
        r2=r2_score(y_true,y_pred)
        
        logger.info(f"""
        The mean absolute error is: {mae:.4f}
        The mean square error is: {mse:.4f}
        The root mean square error is: {rmse:.4f}
        The R2 score is: {r2:.4f}""")  
        return mse,mae,rmse,r2 
        
    except Exception as e:
        logger.error(f"An error occurred:{e}")
        raise

#if __name__=="__main__":
 #   mse,mae,rmse,r2=regression_metrics(y_true,y_pred)
 # Uncomment to test with dummy data
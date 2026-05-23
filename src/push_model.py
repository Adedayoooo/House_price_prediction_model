from huggingface_hub import HfApi,login
import logging 
import joblib

from src.config import DATA_PATH,HF_TOKEN_PATH,MODEL_PATH
from src.train import main as run_train
from src.preprocessing import main as run_preprocessing

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

house_features,target=run_preprocessing(DATA_PATH)
best_model=run_train(house_features,target)

token=HF_TOKEN_PATH.read_text().strip() if HF_TOKEN_PATH.exists() else None
   
def push_model_to_huggingface(best_model,repo_name:str,token:str=None):
  try:
    logger.info("Logging into Hugging Face Hub...")
    if token:
      login(token=token)
    else:
      login()
    model_filename=MODEL_PATH
    logger.info(f"Saving best model to {model_filename}...")
    joblib.dump(best_model,model_filename)
    logger.info(f"Model saved locally as {model_filename}")
    api = HfApi()
    api.upload_file(path_or_fileobj=model_filename,path_in_repo=model_filename.name,repo_id=repo_name,repo_type="model",commit_message="Upload trained KNN-based house price prediction model")
    logger.info(f"Model successfully pushed to Hugging Face Hub: https://huggingface.co/{repo_name}")
  except Exception as e:
    logger.error(f"Failed to push model to HuggingFace due to the following: {e}")
    raise
  
if __name__=="__main__": 
  push_model_to_huggingface(best_model,"KNN Based House Price Prediction Model")
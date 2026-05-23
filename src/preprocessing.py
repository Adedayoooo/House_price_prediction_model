import pandas as pd
import logging
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from config import DATA_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def load_data(DATA_PATH)->pd.DataFrame:
    try:
        logger.info("Loading house price prediction data...")
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Data file not found at: {DATA_PATH}")
        house_data=pd.read_csv(DATA_PATH)
        logger.info(f"Data successfully loaded. Data has {house_data.shape[0]} rows and {house_data.shape[1]} columns")
        logger.info(f"Columns: {list(house_data.columns)}")
        return house_data
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

def values_check(house_data)->pd.DataFrame:
    try:
        logger.info("Checking for missing values...")
        counts=house_data.isnull().sum()
        if counts.sum()>0:
            logger.info("Missing values detected")
            missing=counts[counts>0]
            logger.info(f"Columns with missing values are: {missing}")
            for col in missing.index:
                dtype=house_data[col].dtype
                if pd.api.types.is_numeric_dtype(dtype):
                    fill_value=house_data[col].mean()
                    house_data[col]=house_data[col].fillna(fill_value)
                    logger.info(f"Filled numeric column '{col}' with mean ({fill_value:.2f})")
                else:
                    fill_value=house_data[col].mode()[0]
                    house_data[col]=house_data[col].fillna(fill_value)
                    logger.info(f"Filled categorical column '{col}' with mode ('{fill_value}')")
        else:
            logger.info("There are no missing values")
        return house_data
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise 

def label_encoding(house_data:pd.DataFrame)->pd.DataFrame:
    try:
        logger.info("Encoding categorical variables using LabelEncoder...")
        categorical_cols=house_data.select_dtypes(include=['object','category']).columns
        if len(categorical_cols)==0:
            logger.info("No categorical columns found for encoding.")
            return house_data
        encoder=LabelEncoder()
        for col in categorical_cols:
            house_data[col]=encoder.fit_transform(house_data[col].astype(str))
            logger.info(f"Label encoded column: {col}")
        return house_data
    except Exception as e:
        logger.error(f"An error occurred during: {e}")
        raise
        
def separate_features_target(house_data)->tuple [pd.DataFrame,pd.Series]:
    try:
        logger.info("Separating features from targets...")
        house_data=house_data.drop(house_data.columns[0],axis=1)
        logger.info("Dropped the 'id' column")
        house_features=house_data.drop(['Price'],axis=1)
        target=house_data['Price']
        logger.info(f"The features are: {house_features}, and the target is {target.name}")
        return house_features,target
    except Exception as e:
        logger.error(f"An error occurred:{e}")
        raise 
    
def main(DATA_PATH):
    try:
        logger.info("Starting preprocessing pipeline...")
        house_data=load_data(DATA_PATH)
        house_data=values_check(house_data)
        house_data=label_encoding(house_data)
        house_features,target=separate_features_target(house_data)
        logger.info("Preprocessing pipeline completed")
        return house_features,target 
    except Exception as e:
        logger.error(f"An error occurred:{e}")
        raise 
if __name__=="__main__":
    main(DATA_PATH)
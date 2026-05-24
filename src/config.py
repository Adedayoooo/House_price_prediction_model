from pathlib import Path

ROOT_DIR=Path(__file__).parent.parent

APP_DIR=ROOT_DIR/"app"
DATA_DIR=ROOT_DIR/"data"
MODEL_DIR=ROOT_DIR/"model"
REQUIREMENTS_PATH=ROOT_DIR/"requirements.txt"
HF_TOKEN_PATH=ROOT_DIR/"hf_token.txt"
SRC_DIR=ROOT_DIR/"src"

#Data file
DATA_PATH=DATA_DIR/"house_price_data.csv"

#App files
APP_PATH=APP_DIR/"main.py"
SCHEMAS_PATH=APP_DIR/"schemas.py"
UTILS_PATH=APP_DIR/"utils.py"

#Model file
MODEL_PATH=MODEL_DIR/"rf_house_price_model.pkl"

#Test code files
TRAIN_PATH=SRC_DIR/"train.py"
TEST_PATH=SRC_DIR/"predictor.py"

# Config files
KNN_CONFIG_PATH=SRC_DIR/"CONFIG.json"

#EDA file
EDA_PATH=SRC_DIR/"preprocessing.py"

#Metric file
METRICS_PATH=SRC_DIR/"metrics.py"
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# Final feature set decided after EDA: dropped PURCHASES, 
# PURCHASES_INSTALLMENTS_FREQUENCY, and CASH_ADVANCE_TRX because each was
# >0.8 correlated with another column already in this list (see EDA.ipynb).
FEATURE_COLUMNS = [
    "BALANCE",
    "BALANCE_FREQUENCY",
    "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE",
    "PURCHASES_FREQUENCY",
    "ONEOFF_PURCHASES_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY",
    "PURCHASES_TRX",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "MINIMUM_PAYMENTS",
    "PRC_FULL_PAYMENT",
    "TENURE",
]

# Columns with skew > 1 in EDA — these get a log1p transform in
# data_transformation.py before scaling.
SKEWED_COLUMNS = [
    "BALANCE",
    "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE",
    "PAYMENTS",
    "MINIMUM_PAYMENTS",
    "PURCHASES_TRX",
    "PRC_FULL_PAYMENT",
    "CASH_ADVANCE_FREQUENCY",
    "ONEOFF_PURCHASES_FREQUENCY",
    "CREDIT_LIMIT",
]

N_CLUSTERS_RANGE = range(3, 9)   # k values to try when selecting cluster count
RANDOM_STATE = 42


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.getenv("RAW_DATA_PATH", os.path.join("data", "raw", "CC_GENERAL.csv"))
    ingested_data_path: str = os.path.join("artifacts", "data", "data.csv")
    train_data_path: str = os.path.join("artifacts", "data", "train.csv")
    test_data_path: str = os.path.join("artifacts", "data", "test.csv")


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "models", "preprocessor.pkl")
    transformed_train_path: str = os.path.join("artifacts", "data", "train_features.csv")
    transformed_test_path: str = os.path.join("artifacts", "data", "test_features.csv")


@dataclass
class ModelTrainerConfig:
    cluster_model_file_path: str = os.path.join("artifacts", "models", "cluster_model.pkl")
    classifier_file_path: str = os.path.join("artifacts", "models", "model.pkl")
    cluster_profiles_path: str = os.path.join("artifacts", "models", "cluster_profiles.json")
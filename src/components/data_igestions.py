import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DataIngestionConfig, FEATURE_COLUMNS, RANDOM_STATE
from src.exception import CustomException
from src.logger import logging


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")
        try:
            df = pd.read_csv(self.ingestion_config.raw_data_path)
            logging.info(f"Read raw dataset with shape {df.shape}")

            # Drop CUST_ID — it's an identifier, not a behavioral feature
            if "CUST_ID" in df.columns:
                df = df.drop(columns=["CUST_ID"])

            # Keep only the columns we decided on during EDA
            df = df[FEATURE_COLUMNS]

            # All ingestion outputs (data.csv, train.csv, test.csv) live in
            # artifacts/data/ — create that one folder once, upfront.
            os.makedirs(os.path.dirname(self.ingestion_config.ingested_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.ingested_data_path, index=False, header=True)
            logging.info(f"Saved ingested data to {self.ingestion_config.ingested_data_path}")

            # Train/test split — held out here so model_trainer.py can later
            # report the segment classifier's accuracy on unseen data
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=RANDOM_STATE
            )

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train/test split complete")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


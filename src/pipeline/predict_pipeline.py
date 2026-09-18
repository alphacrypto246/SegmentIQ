import json
import sys

import pandas as pd

from src.config import DataTransformationConfig, FEATURE_COLUMNS, ModelTrainerConfig
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        """
        Loads all three artifacts once, at object creation — not per
        prediction. Reloading from disk on every request would be a
        needless latency hit. FastAPI will create ONE PredictPipeline
        instance at startup and reuse it for every incoming request.
        """
        try:
            transformation_config = DataTransformationConfig()
            trainer_config = ModelTrainerConfig()

            self.preprocessor = load_object(transformation_config.preprocessor_obj_file_path)
            self.classifier = load_object(trainer_config.classifier_file_path)

            with open(trainer_config.cluster_profiles_path) as f:
                self.cluster_profiles = json.load(f)

            logging.info("PredictPipeline: all artifacts loaded successfully")

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, customer_data: dict) -> dict:
        """
        Takes a single customer's raw (unscaled) feature values as a dict,
        applies the exact same preprocessing used at training time, and
        returns which segment they belong to plus a plain-English profile.
        """
        try:
            # Build a single-row DataFrame in the exact column order the
            # preprocessor was fit on — order matters for sklearn Pipelines.
            input_df = pd.DataFrame([customer_data])[FEATURE_COLUMNS]

            # IMPORTANT: transform only, never fit, at inference time.
            # Reusing the training-time imputer medians and scaler
            # parameters is what keeps predictions consistent with training.
            transformed = self.preprocessor.transform(input_df)
            transformed_df = pd.DataFrame(transformed, columns=FEATURE_COLUMNS)

            segment = int(self.classifier.predict(transformed_df)[0])
            profile = self.cluster_profiles.get(str(segment), {})

            return {
                "segment": segment,
                "description": profile.get("description", "unknown segment"),
                "segment_size_in_training_data": profile.get("size", 0),
            }

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    sample_customer = {
        "BALANCE": 1500.50,
        "BALANCE_FREQUENCY": 0.9,
        "ONEOFF_PURCHASES": 300.0,
        "INSTALLMENTS_PURCHASES": 500.0,
        "CASH_ADVANCE": 0.0,
        "PURCHASES_FREQUENCY": 0.6,
        "ONEOFF_PURCHASES_FREQUENCY": 0.3,
        "CASH_ADVANCE_FREQUENCY": 0.0,
        "PURCHASES_TRX": 12,
        "CREDIT_LIMIT": 5000.0,
        "PAYMENTS": 1200.0,
        "MINIMUM_PAYMENTS": 200.0,
        "PRC_FULL_PAYMENT": 0.3,
        "TENURE": 12,
    }

    pipeline = PredictPipeline()
    result = pipeline.predict(sample_customer)
    print(result)
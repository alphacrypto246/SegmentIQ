import os
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer

from src.config import DataTransformationConfig, FEATURE_COLUMNS, SKEWED_COLUMNS
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_preprocessor_object(self):
        """
        Builds a single sklearn Pipeline: median-impute -> log1p the
        skewed columns -> standard-scale everything. Returned as one
        fitted object so predict_pipeline.py only ever calls
        .transform() once, guaranteed identical to training.
        """
        try:
            preprocessor = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("log_transform", FunctionTransformer(self._log_transform_subset, validate=False)),
                    ("scaler", StandardScaler()),
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def _log_transform_subset(self, X):
        """
        X arrives as a numpy array (imputer output, columns in
        FEATURE_COLUMNS order). Only log-transform the columns flagged
        as skewed in EDA — log1p on a bounded [0,1] frequency column
        would distort it rather than help.
        """
        X = pd.DataFrame(X, columns=FEATURE_COLUMNS)
        for col in SKEWED_COLUMNS:
            X[col] = np.log1p(X[col])
        return X

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info(f"Loaded train {train_df.shape} and test {test_df.shape} data")

            preprocessor = self.get_preprocessor_object()

            # Fit ONLY on training data — test data must never influence
            # the imputation medians or scaling parameters (data leakage).
            train_arr = preprocessor.fit_transform(train_df)
            test_arr = preprocessor.transform(test_df)

            train_features = pd.DataFrame(train_arr, columns=FEATURE_COLUMNS)
            test_features = pd.DataFrame(test_arr, columns=FEATURE_COLUMNS)

            # Make sure artifacts/data/ exists, independent of whether
            # ingestion already ran and created it.
            os.makedirs(os.path.dirname(self.transformation_config.transformed_train_path), exist_ok=True)

            train_features.to_csv(self.transformation_config.transformed_train_path, index=False)
            test_features.to_csv(self.transformation_config.transformed_test_path, index=False)
            logging.info("Saved transformed train/test features")

            # save_object() already creates artifacts/models/ internally
            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessor,
            )
            logging.info(f"Saved preprocessor object to {self.transformation_config.preprocessor_obj_file_path}")

            return (
                self.transformation_config.transformed_train_path,
                self.transformation_config.transformed_test_path,
                self.transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


import sys

from src.components.data_igestions import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


class TrainPipeline:
    def run(self):
        """
        Runs the full offline pipeline: ingest -> transform -> train.
        This is the single entry point for retraining the model, replacing
        the need to manually chain the three components together.
        """
        try:
            logging.info("=== Training pipeline started ===")

            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()

            transformation = DataTransformation()
            train_feat_path, test_feat_path, preprocessor_path = (
                transformation.initiate_data_transformation(train_path, test_path)
            )

            trainer = ModelTrainer()
            model_path = trainer.initiate_model_training(train_feat_path)

            logging.info("=== Training pipeline completed successfully ===")
            return model_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    model_path = pipeline.run()
    print(f"Training complete. Model saved to: {model_path}")
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add the project root to the path if it's not there already
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.exception import CustomException
from src.logger import logging
import pandas as pd

from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', "data.csv")
    train_data_path: str = os.path.join('artifacts', "train.csv")  # Clustering can use train path as the single dataset

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Load your dataset (adjust the path to the actual location)
            df = pd.read_csv('Notebook\data\data.csv')

            logging.info('Read the dataset as dataframe')

            # Create the necessary directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the entire dataframe as 'train_data' for clustering purposes
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            logging.info("Data ingestion completed")

            return self.ingestion_config.train_data_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, _ = data_transformation.initiate_data_transformation(train_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr))

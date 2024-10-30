import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.components.data_transformation import DataTransformation


class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", 'kmeans_model.pkl')  # Path for KMeans model
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        self.model = None
        self.preprocessor = None
        self.data_transformation = DataTransformation()

    def _load_resources(self):
        try:
            if not os.path.isfile(self.model_path):
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            if not os.path.isfile(self.preprocessor_path):
                raise FileNotFoundError(f"Preprocessor file not found at {self.preprocessor_path}")
            
            self.model = load_object(file_path=self.model_path)
            self.preprocessor = load_object(file_path=self.preprocessor_path)
        
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features: pd.DataFrame):
        try:
            if self.model is None or self.preprocessor is None:
                self._load_resources()

            # Rename columns to match those used during training
            features = self.data_transformation.col_rename(features)

            # Ensure the columns in features match the preprocessor's expected columns
            expected_columns = self.preprocessor.feature_names_in_
            missing_columns = set(expected_columns) - set(features.columns)
            if missing_columns:
                raise KeyError(f"Missing columns in input features: {missing_columns}")

            # Reorder columns to match preprocessor expectation
            features = features[expected_columns]

            # Transform features and perform clustering
            data_scaled = self.preprocessor.transform(features)
            cluster = self.model.predict(data_scaled)  # Perform KMeans clustering to predict cluster
            
            # Map cluster to devtype (this mapping should be based on your training data)
            devtype_mapping = {0: "Frontend Developer", 1: "Backend Developer", 2: "Full Stack Developer", 3: "Data Scientist"}
            devtype = devtype_mapping.get(cluster[0], "Unknown Developer Type")  # Retrieve devtype from the cluster
            
            return devtype
        
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        language_have_worked_with: str,
        webframe_have_worked_with: str,
        tools_tech_have_worked_with: str,
        misc_tech_have_worked_with: str
    ):
        # Initialize the attributes with input values
        self.language_have_worked_with = language_have_worked_with
        self.webframe_have_worked_with = webframe_have_worked_with
        self.tools_tech_have_worked_with = tools_tech_have_worked_with
        self.misc_tech_have_worked_with = misc_tech_have_worked_with

    def get_data_as_data_frame(self):
        try:
            # Create a dictionary with feature names and values
            custom_data_input_dict = {
                "language_have_worked_with": [self.language_have_worked_with],
                "webframe_have_worked_with": [self.webframe_have_worked_with],
                "tools_tech_have_worked_with": [self.tools_tech_have_worked_with],
                "misc_tech_have_worked_with": [self.misc_tech_have_worked_with]
            }

            # Convert the dictionary into a pandas DataFrame
            df = pd.DataFrame(custom_data_input_dict)

            return df
        
        except Exception as e:
            raise CustomException(e, sys)

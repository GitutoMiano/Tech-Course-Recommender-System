from dataclasses import dataclass
import numpy as np
import pandas as pd
import re
import sys
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def col_rename(self, df):
        """
        Renames columns by replacing spaces and hyphens with underscores and converting to lowercase.
        """
        df.columns = [col.replace(" ", "_").replace("-", "_").lower() for col in df.columns]
        return df

    def combine_columns(self, df):
        """
        Creates a new column 'all_info' containing concatenated text from all columns.
        """
        df['all_info'] = df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        return df

    def remove_non_word_characters(self, df):
        """
        Removes non-word characters from 'all_info' column.
        """
        df['all_info'] = df['all_info'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
        return df

    def tfidf_vectorize(self, df):
        """
        Converts 'all_info' column into TF-IDF features.
        """
        vectorizer = TfidfVectorizer(stop_words='english')
        features = vectorizer.fit_transform(df['all_info'])
        return features, vectorizer

    def get_data_transformer_object(self):
        """
        Sets up the preprocessor for categorical columns.
        """
        try:
            categorical_columns = ['languagehaveworkedwith', 'webframehaveworkedwith', 'cluster',
                                   'toolstechhaveworkedwith', 'misctechhaveworkedwith', 'devtype']
            logging.info(f"Categorical columns: {categorical_columns}")

            preprocessor = ColumnTransformer(
                [("cat_pipeline", 'passthrough', categorical_columns)]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Apply column renaming
            train_df = self.col_rename(train_df)
            test_df = self.col_rename(test_df)

            # Print columns for debugging
            logging.info(f"Train DataFrame columns: {train_df.columns}")
            logging.info(f"Test DataFrame columns: {test_df.columns}")

            # Obtain preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            # Define target column name and numerical columns
            target_column_name = "cluster"
            categorical_columns = ['languagehaveworkedwith', 'webframehaveworkedwith',
                                   'toolstechhaveworkedwith', 'misctechhaveworkedwith', 'devtype']

            # Check if target column and numerical columns are present in the DataFrame
            if target_column_name not in train_df.columns:
                raise KeyError(f"Target column '{target_column_name}' not found in training data.")
            if not set(categorical_columns).issubset(train_df.columns):
                missing_columns = set(categorical_columns) - set(train_df.columns)
                raise KeyError(f"Numerical columns missing from training data: {missing_columns}")

            # Prepare features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Apply preprocessing
            logging.info("Applying preprocessing object on training and testing dataframes.")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine features and target into final arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save preprocessing object
            logging.info("Saving preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)

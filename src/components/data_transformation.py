from dataclasses import dataclass
import numpy as np
import pandas as pd
import sys
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import re
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    model_obj_file_path = os.path.join('artifacts', "kmeans_model.pkl")
    cluster_devtype_map_path = os.path.join('artifacts', "cluster_devtype_map.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def col_rename(self, df):
        """
        Renames columns by replacing spaces and hyphens with underscores and converting to lowercase.
        """
        for column in df.columns:
            df.rename(columns={column: column.replace(" ", "_").replace("-", "_").lower()}, inplace=True)
        return df
    
    def get_data_transformer_object(self):
        '''
        Function to create data transformer pipeline for text features
        '''
        try:
            text_columns = ['languagehaveworkedwith', 'webframehaveworkedwith', 'toolstechhaveworkedwith',
                'misctechhaveworkedwith', 'devtype']

            # Text preprocessing pipeline
            text_pipeline = Pipeline(
                steps=[("tfidf", TfidfVectorizer(stop_words='english'))]
            )

            logging.info(f"Text columns: {text_columns}")

            preprocessor = ColumnTransformer(
                [("text_pipeline", text_pipeline, text_columns)],
                remainder='passthrough'
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path):
        try:
            # Read data
            train_df = pd.read_csv(train_path)

            # Apply column renaming
            train_df = self.col_rename(train_df)

            # Obtain preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            # Prepare text features for transformation
            logging.info("Applying preprocessing object on training dataframe.")
            input_feature_train_arr = preprocessing_obj.fit_transform(train_df)

            # K-Means clustering
            k = 7  # Number of clusters
            kmeans_model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
            train_df['cluster'] = kmeans_model.fit_predict(input_feature_train_arr)

            # Create cluster-DevType mapping
            logging.info("Creating cluster to DevType mapping.")
            cluster_devtype_map = train_df.groupby('cluster')['devtype'].unique().reset_index()
            cluster_devtype_map = dict(zip(cluster_devtype_map['cluster'], cluster_devtype_map['devtype']))

            # Save preprocessing object, model, and cluster-DevType map
            logging.info("Saving preprocessing, model, and cluster to DevType map objects.")
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessing_obj)
            save_object(file_path=self.data_transformation_config.model_obj_file_path, obj=kmeans_model)
            save_object(file_path=self.data_transformation_config.cluster_devtype_map_path, obj=cluster_devtype_map)

            return input_feature_train_arr, self.data_transformation_config.preprocessor_obj_file_path, self.data_transformation_config.model_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)

    def recommend_devtypes(self, language, webframe, tools, tech):
        """
        Recommend DevTypes based on user inputs using the pre-trained KMeans model.
        """
        try:
            # Load preprocessing object, model, and cluster-to-DevType map
            preprocessor = load_object(self.data_transformation_config.preprocessor_obj_file_path)
            kmeans_model = load_object(self.data_transformation_config.model_obj_file_path)
            cluster_devtype_map = load_object(self.data_transformation_config.cluster_devtype_map_path)

            # Combine input into 'all_info' format
            combined_info = ' '.join([language, webframe, tools, tech])

            # Preprocess the input data
            transformed_input = preprocessor.transform([combined_info])

            # Predict the cluster
            predicted_cluster = kmeans_model.predict(transformed_input)[0]

            # Get recommended DevTypes for the predicted cluster
            recommended_devtypes = cluster_devtype_map.get(predicted_cluster, [])

            return predicted_cluster, recommended_devtypes

        except Exception as e:
            raise CustomException(e, sys)

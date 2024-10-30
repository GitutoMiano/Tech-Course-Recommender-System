import os
import sys
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from src.exception import CustomException
from src.utils import save_object

# Define the path to the artifacts directory
artifacts_dir = "artifacts"

# Create the directory if it does not exist
if not os.path.exists(artifacts_dir):
    os.makedirs(artifacts_dir)

class ModelTrainerConfig:
    def __init__(self):
        self.trained_models_dir = os.path.join(artifacts_dir, "models")
        if not os.path.exists(self.trained_models_dir):
            os.makedirs(self.trained_models_dir)

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, data_array):
        try:
            X = data_array  # No splitting, we use the entire dataset for clustering

            # Define clustering models
            models = {
                'KMeans': KMeans(n_clusters=7, random_state=7215),
                'DBSCAN': DBSCAN(eps=0.5, min_samples=7),
                'AgglomerativeClustering': AgglomerativeClustering(n_clusters=7)
            }

            for model_name, model in models.items():
                print(f"Training {model_name}...")
                model.fit(X)
                
                # Obtain cluster labels
                labels = model.labels_

                # Evaluate the clustering performance
                silhouette_avg = silhouette_score(X, labels)
                db_score = davies_bouldin_score(X, labels)
                ch_score = calinski_harabasz_score(X, labels)

                print(f"\n{model_name} Clustering Performance:\n")
                print(f"Silhouette Score: {silhouette_avg}")
                print(f"Davies-Bouldin Score: {db_score}")
                print(f"Calinski-Harabasz Score: {ch_score}")

                # Save the trained model
                model_path = os.path.join(self.model_trainer_config.trained_models_dir, f"{model_name.lower().replace(' ', '_')}_model.pkl")
                save_object(file_path=model_path, obj=model)

        except Exception as e:
            raise CustomException(e, sys)

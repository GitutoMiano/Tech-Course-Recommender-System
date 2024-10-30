import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_clustering_models(X, models):
    """
    Evaluate multiple clustering models and select the best based on Silhouette Score.
    
    Parameters:
    X : numpy array or pandas DataFrame
        Data to be clustered.
    models : dict
        A dictionary of clustering models to evaluate.

    Returns:
    dict : A report containing the metrics for each model.
    str : The name of the best model based on Silhouette Score.
    """
    try:
        report = {}
        best_model_name = None
        best_silhouette_score = -1  # Higher is better for silhouette score

        for model_name, model in models.items():
            print(f"Training {model_name}...")
            model.fit(X)

            # Get cluster labels
            labels = model.labels_

            # Calculate clustering metrics
            silhouette_avg = silhouette_score(X, labels)
            db_score = davies_bouldin_score(X, labels)
            ch_score = calinski_harabasz_score(X, labels)

            # Store the results in the report
            report[model_name] = {
                'Silhouette Score': silhouette_avg,
                'Davies-Bouldin Score': db_score,
                'Calinski-Harabasz Score': ch_score
            }

            # Select the model with the best silhouette score
            if silhouette_avg > best_silhouette_score:
                best_silhouette_score = silhouette_avg
                best_model_name = model_name

            print(f"\n{model_name} Clustering Performance:")
            print(f"Silhouette Score: {silhouette_avg}")
            print(f"Davies-Bouldin Score: {db_score}")
            print(f"Calinski-Harabasz Score: {ch_score}")

        print("\nComparison of models complete.")
        return report, best_model_name

    except Exception as e:
        raise CustomException(e, sys)

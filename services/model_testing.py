import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tabulate import tabulate
import joblib

from config import CLASSIFIERS, TESTING_DATASET_PATH


class ModelTesting:
    epi_models_accuracy_df = []

    def __init__(self) -> None:
        logging.info("Model Testing service invoked.")

    def start_model_testing(self, X_test, y_test):
        """Starts the model testing after picking up the stored models and testing payload."""
        for model_name in CLASSIFIERS:
            model_name = model_name[0]
            test_data = pd.read_csv(TESTING_DATASET_PATH)

            # Split the testing dataset into features (X_test) and target variable (y_test)
            X_test = test_data.drop(columns=["target"])
            y_test = test_data["target"]
            # Accuracy: TP + TN / Total
            accuracy, cm, precision, recall, f1 = self.epi_model_accuracy(
                model_name, X_test, y_test
            )
            cm = np.array(cm)
            df_data_row = {
                "Model_Name": model_name,
                "Accuracy": accuracy,
                "Confusion_Matrix": cm,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
            ModelTesting.epi_models_accuracy_df.append(df_data_row)
            # logging.info(df_data_row)

    def epi_model_accuracy(self, selected_model_name, X_test, y_test):
        """Calculates the model accuracies"""
        selected_model_name = (
            "data_models/"
            + selected_model_name
            + "_breast_cancer_ml_model.joblib"
        )
        logging.info("Piciking up model:")
        logging.info(selected_model_name)
        selected_model = joblib.load(selected_model_name)
        predictions = selected_model.predict(X_test)

        # Calculate the confusion matrix
        cm = confusion_matrix(y_test, predictions)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions) * 100

        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, predictions
        )

        return accuracy, cm, precision, recall, f1

    @classmethod
    def display_accuracy_data(cls):
        """Displays the testing accuracies and other metrices of the models."""
        headers = [
            "Model Name",
            "Accuracy",
            "Confusion Matrix",
            "Precision",
            "Recall",
            "F1",
        ]
        data = []
        for row in ModelTesting.epi_models_accuracy_df:
            model_name = row["Model_Name"]
            accuracy = row["Accuracy"]
            confusion_matrix = row["Confusion_Matrix"]
            precision = row["precision"]
            recall = row["recall"]
            f1 = row["f1"]
            data.append(
                [model_name, accuracy, confusion_matrix, precision, recall, f1]
            )

        logging.info(tabulate(data, headers=headers, tablefmt="grid"))

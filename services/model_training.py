import os
import logging

import joblib
import pandas as pd

from config import MODEL_DIRECTORY, TRAINING_DATASET_PATH


class ModelTraining:
    def __init__(self) -> None:
        """Model Training object initializer."""
        logging.info("Model Training service intitated: ")

    @classmethod
    def start_model_training(cls, clf_name, classifier):
        """Helps in training the models and storing them."""
        logging.info("Started training models.")
        # Load: the training dataset from CSV
        train_data = pd.read_csv(TRAINING_DATASET_PATH)

        # Divide: the training data into features (X_train) and target variable (y_train)
        X_train = train_data.drop(columns=["target"])
        y_train = train_data["target"]
        trained_models = {}

        logging.info(f"Training Model: {clf_name}")

        classifier.fit(X_train, y_train)

        if not os.path.exists(MODEL_DIRECTORY):
            os.makedirs(MODEL_DIRECTORY)

        model_filename = joblib.os.path.join(
            MODEL_DIRECTORY, f"{clf_name}_breast_cancer_ml_model.joblib"
        )

        joblib.dump(classifier, model_filename)

        trained_models[clf_name] = model_filename

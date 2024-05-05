import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from config import (
    BS_CANCER_DATASET,
    TRAINING_DATASET_PATH,
    TESTING_DATASET_PATH,
)


class DataPreprocessing:
    def __init__(self) -> None:
        self.cancer_df = None
        logging.info("Data Preparation Service has been invoked:")

    @classmethod
    def load_initial_data(self):
        """Helps in loading original dataset."""
        logging.info("Started loading intial data")
        # Import Cancer data drom the Sklearn library
        from sklearn.datasets import load_breast_cancer

        cancer = load_breast_cancer()
        logging.info(cancer["DESCR"])
        logging.info(cancer["target_names"])
        logging.info(cancer["target"])
        # logging.info(cancer["feature_names"])
        # logging.info(cancer["data"])
        self.cancer_df = pd.DataFrame(
            np.c_[cancer["data"], cancer["target"]],
            columns=np.append(cancer["feature_names"], ["target"]),
        )
        self.cancer_df.to_csv(BS_CANCER_DATASET)

    @classmethod
    def vizualize_data(cls):
        print(
            sns.pairplot(
                cls.cancer_df,
                hue="target",
                vars=[
                    "mean radius",
                    "mean texture",
                    "mean area",
                    "mean perimeter",
                    "mean smoothness",
                ],
            )
        )
        print(sns.countplot(cls.cancer_df["target"], label="Count"))
        print(
            sns.scatterplot(
                x="mean area",
                y="mean smoothness",
                hue="target",
                data=cls.cancer_df,
            )
        )
        print(plt.figure(figsize=(20, 10)))
        print(sns.heatmap(cls.cancer_df.corr(), annot=True))

    @classmethod
    def split_data(cls):
        logging.info("Splitting the data into training and testing:")
        X = cls.cancer_df.drop(["target"], axis=1)
        y = cls.cancer_df["target"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=5
        )
        training_data = pd.concat([X_train, y_train], axis=1)
        testing_data = pd.concat([X_test, y_test], axis=1)
        training_data.to_csv(TRAINING_DATASET_PATH)
        testing_data.to_csv(TESTING_DATASET_PATH)
        return X_train, X_test, y_train, y_test

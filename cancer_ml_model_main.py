import time
import logging
from multiprocessing import Pool, cpu_count

import pandas as pd

from config import CLASSIFIERS
from services.data_processing import DataPreprocessing
from services.model_testing import ModelTesting
from services.model_training import ModelTraining

# Configure: logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Main function to run the model training and testing process.
    """
    # Load initial data and visualize it
    DataPreprocessing().load_initial_data()
    DataPreprocessing.vizualize_data()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = DataPreprocessing.split_data()

    # Determine the number of CPUs to use
    num_cpu_to_use = max(1, cpu_count() - 2)
    start_time = time.time()
    # Note: I have used only 3 cores of CPU
    try:
        with Pool(3) as mp_pool:
            results = mp_pool.starmap(
                ModelTraining.start_model_training,
                [
                    (clf_name, classifier)
                    for clf_name, classifier in CLASSIFIERS
                ],
            )

        logger.info(
            f"CPU count:{num_cpu_to_use}; Time taken: {time.time() - start_time}"
        )

    except Exception as exception:
        logger.error(exception)

    # Start model testing
    ModelTesting().start_model_testing(X_test, y_test)
    ModelTesting.display_accuracy_data()


if __name__ == "__main__":
    main()

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

CLASSIFIERS = [
    ("Logistic Regression", LogisticRegression(max_iter=5000)),
    ("Random Forest", RandomForestClassifier()),
    ("XGBoost", XGBClassifier()),
    ("GradientBoost", GradientBoostingClassifier()),
    ("KNN", KNeighborsClassifier(n_neighbors=5)),
    ("MultinomialNB", MultinomialNB()),
    ("SVC", SVC()),
]

BS_CANCER_DATASET = "data_files/breast_cancer_dataset_whole.csv"
TRAINING_DATASET_PATH = "data_files/breast_cancer_training_data.csv"
TESTING_DATASET_PATH = "data_files/breast_cancer_testing_data.csv"

MODEL_DIRECTORY = "data_models/"

## Parallel training ML Algos: Project Overview

This project trains multiple machine learning algorithms simultaneously to reduce training time, especially for large datasets, utilizing multiprocessing. Additionally, Ruff is integrated for code quality.

### Key Features

- **Parallel Training:** Enables training multiple algorithms concurrently, leveraging multi-core processors.
  
- **Time Efficiency:** Distributes workload across processes for faster model training, beneficial for large datasets.
  
- **Ruff Integration:** Ensures consistent code quality and adherence to standards.

## Breast Cancer Classification

Breast cancer classification predicts cancer diagnosis based on observations. Here's a summary:

#### Features

The dataset includes 30 features providing crucial cell characteristic information.

#### Linear Separability

All 30 input features allow effective discrimination between benign and malignant tumors.

#### Dataset Details

- **Instances:** 569
- **Class Distribution:** Malignant (212), Benign (357)

#### Target Classes

- **Malignant:** Cancerous tumors
- **Benign:** Non-cancerous tumors

Various classifier algorithms were explored:

- Logistic Regression
- K-Nearest Neighbor (K-NN)
- Naive Bayes
- Decision Tree
- Random Forest
- Adaboost
- XGBoost

# Comment Category Prediction

This repository contains a machine learning project dedicated to classifying comments into distinct categories. Using various Natural Language Processing (NLP) techniques and classification algorithms, this project builds and evaluates multiple models to achieve accurate predictions across four specific comment categories.

## Overview

The primary goal of this project is to automatically categorize text comments. The workflow involves text preprocessing, building scikit-learn pipelines, and extensively tuning hyperparameters for different machine learning models to identify the best performing classifier based on the `f1_macro` scoring metric.

## Models Explored

The notebook trains and evaluates a wide variety of machine learning classifiers:

- **Logistic Regression**
- **Linear Support Vector Classifier (LinearSVC)**
- **Stochastic Gradient Descent (SGDClassifier)**
- **LightGBM Classifier (LGBMClassifier)**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **AdaBoost Classifier**
- **XGBoost Classifier (XGBClassifier)**

## Methodology

- **Pipeline Construction:** Models are implemented using `sklearn.pipeline.Pipeline` to ensure a clean and reproducible workflow without data leakage.
- **Handling Class Imbalance:** Class weights (e.g., `{0: 1, 1: 1.5, 2: 1.2, 3: 2}`) are utilized across various models to handle unbalanced category distributions.
- **Hyperparameter Tuning:** Models are optimized using `RandomizedSearchCV` (with 3-fold cross-validation) to find the best combination of parameters efficiently.

## Technologies Used

- **Python**
- **Jupyter Notebook**
- **scikit-learn** (Pipelines, RandomizedSearchCV, ML algorithms)
- **LightGBM**
- **XGBoost**

## Getting Started

### Prerequisites

Ensure you have Python installed. Install the required libraries using pip:

```bash
pip install numpy pandas scikit-learn lightgbm xgboost jupyter
```

### Running the Notebook

1. Clone this repository:
   ```bash
   git clone https://github.com/NandaKumar-10/Comment-Category-Prediction.git
   cd Comment-Category-Prediction
   ```
2. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open `23f3001984-notebook-t12026.ipynb` and run the cells sequentially.

## Evaluation

Model performance is heavily reliant on the **F1-Macro** score, ensuring that all categories, regardless of their support size, are treated equally during evaluation. The notebook logs the training process and visually compares the scores to declare the optimal configuration.

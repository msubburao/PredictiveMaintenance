
import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

# for creating a folder
import os

# for data preprocessing and pipeline creation
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score

from huggingface_hub import login, HfApi

# for model serialization
import joblib

api = HfApi(token=os.getenv("HF_TOKEN"))

BASE_PATH = "hf://datasets/msubburao/predictivemaintenanceds/"
LXtrain = pd.read_csv(BASE_PATH + "Xtrain.csv")
LXtest = pd.read_csv(BASE_PATH + "Xtest.csv")
Lytrain = pd.read_csv(BASE_PATH + "ytrain.csv")
Lytest = pd.read_csv(BASE_PATH + "ytest.csv")

# Set the clas weight to handle class imbalance
class_weight = Lytrain.value_counts()[0] / Lytrain.value_counts()[1]
class_weight

numeric_features = LXtrain.columns.tolist()

mlflow.set_experiment("pm_xgb_exp_f")

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

param_grid_xgb = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__subsample": [0.8],
    "xgbclassifier__colsample_bytree": [0.7],
    "xgbclassifier__min_child_weight": [1, 3]
}

# Model pipeline
model_pipeline_xgb = make_pipeline(xgb_model)

with mlflow.start_run():
    # Hyperparameter tuning
    grid_search_xgb = GridSearchCV(model_pipeline_xgb, param_grid_xgb, cv=3, n_jobs=-1)
    grid_search_xgb.fit(LXtrain, Lytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search_xgb.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search_xgb.best_params_)

    # Store and evaluate the best model
    best_model_xgb = grid_search_xgb.best_estimator_

    classification_threshold = 0.45

    y_pred_train_proba = best_model_xgb.predict_proba(LXtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model_xgb.predict_proba(LXtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(Lytrain, y_pred_train, output_dict=True)
    test_report = classification_report(Lytest, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Save the model locally
    model_path = "predmaintain_model_v1.joblib"
    joblib.dump(best_model_xgb, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "msubburao/predictivemaintenancemodel"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    api.upload_file(
        path_or_fileobj="predmaintain_model_v1.joblib",
        path_in_repo="predmaintain_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )

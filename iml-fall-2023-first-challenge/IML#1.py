import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score


# Function to preprocess data
def preprocess_data(train_df, test_df):
    X = train_df.drop(columns=["hospital_death", "RecordID"])
    y = train_df["hospital_death"]
    test_df = test_df.drop(columns=["RecordID"])

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Label encode
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    test_df = pd.DataFrame(imputer.transform(test_df), columns=test_df.columns)

    return X, y, test_df


# Function to train XGBoost model
def train_xgboost(X, y):
    params = {
        "objective": "binary:logistic",
        "n_estimators": 200,
        "max_depth": 4,
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "auc",
    }

    xgb_model = xgb.XGBClassifier(**params)
    xgb_model.fit(X, y)

    return xgb_model


# Function to train the meta-model (logistic regression)
def train_meta_model(X_val, y_val, stacked_val_predictions):
    meta_model = LogisticRegression()
    meta_model.fit(stacked_val_predictions, y_val)
    return meta_model


# Function to perform Platt scaling
def apply_platt_scaling(
    meta_model, stacked_test_predictions, stacked_val_predictions, y_val
):
    calibrator = CalibratedClassifierCV(meta_model, method="sigmoid", cv="prefit")
    calibrator.fit(stacked_val_predictions, y_val)
    smoothed_predictions = calibrator.predict_proba(stacked_test_predictions)[:, 1]
    return smoothed_predictions


# Main function
def main():
    # Load data
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    sample_submission_df = pd.read_csv("sample.csv")

    # Preprocessing
    X, y, test_df = preprocess_data(train_df, test_df)

    # Train XGBoost model
    xgb_model = train_xgboost(X, y)

    # Split data for meta-model training
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=70
    )
    xgb_val_probs = xgb_model.predict_proba(X_val)[:, 1]

    # Stack predictions for meta-model training
    stacked_val_predictions = xgb_val_probs.reshape(-1, 1)

    # Train meta-model
    meta_model = train_meta_model(X_val, y_val, stacked_val_predictions)

    # Get model predictions on test set
    xgb_test_probs = xgb_model.predict_proba(test_df)[:, 1]

    # Stack predictions for final predictions
    stacked_test_predictions = xgb_test_probs.reshape(-1, 1)

    # Apply Platt scaling
    smoothed_predictions = apply_platt_scaling(
        meta_model, stacked_test_predictions, stacked_val_predictions, y_val
    )

    # Create submission file
    submission_df = sample_submission_df.copy()
    submission_df["hospital_death"] = smoothed_predictions
    submission_file_path = "test_predict_file.csv"
    submission_df.to_csv(submission_file_path, index=False)


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score


def gradient_boost(filename: str):
    print("XGBoost: ")
    df = pd.read_csv(filename)
    features = df.loc[:, df.columns != "TenYearCHD"]
    labels = df["TenYearCHD"]
    oversampler = RandomOverSampler(random_state=42)
    features_resampled, labels_resampled = oversampler.fit_resample(features,
                                                                    labels)
    (
        features_train,
        features_test,
        labels_train,
        labels_test,
    ) = train_test_split(
        features_resampled, labels_resampled, test_size=0.2, random_state=42
    )
    model = xgb.XGBClassifier()
    model.fit(features_train, labels_train)
    labels_pred = model.predict(features_test)
    labels_train_pred = model.predict(features_train)
    return (labels_train, labels_train_pred, labels_test, labels_pred)

def grid_search(filename):
    df = pd.read_csv(filename)
    features = df.loc[:, df.columns != "TenYearCHD"]
    labels = df["TenYearCHD"]
    oversampler = RandomOverSampler(random_state=42)
    features_resampled, labels_resampled = oversampler.fit_resample(features,
                                                                    labels)
    (
        features_train,
        features_test,
        labels_train,
        labels_test,
    ) = train_test_split(
        features_resampled, labels_resampled, test_size=0.2, random_state=42
    )
    param_grid = {
        'max_depth': range(3,10),
        'learning_rate': [0.1, 0.01, 0.001],
        'n_estimators': [100, 500, 1000],
        'subsample': [0.5, 0.8, 1.0],
        'colsample_bytree': [0.5, 0.8, 1.0],
        'gamma': [0, 0.3, 0.5],
        'reg_alpha': [0, 0.5, 1.0],
        'reg_lambda': [0, 0.5, 1.0]
    }
    model = xgb.XGBClassifier()
    grid_search = GridSearchCV(model, param_grid, scoring='f1', cv=5)
    grid_search.fit(features_train, labels_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    labels_pred = best_model.predict(features_test)
    f1 = f1_score(labels_test, labels_pred)
    print("Best Hyperparameters:", best_params)
    print("F1:", f1)
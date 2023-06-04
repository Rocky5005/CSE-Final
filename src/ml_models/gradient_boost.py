import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score
from imblearn.pipeline import Pipeline


def gradient_boost(filename: str):
    print("XGBoost: ")
    df = pd.read_csv(filename)
    features = df.loc[:, df.columns != "TenYearCHD"]
    labels = df["TenYearCHD"]
    (
        features_train,
        features_test,
        labels_train,
        labels_test,
    ) = train_test_split(
        features, labels, test_size=0.2,
        stratify=labels, random_state=42
    )  # stratified data
    oversampler = RandomOverSampler(random_state=42)
    (
        features_resampled, labels_resampled
    ) = oversampler.fit_resample(features_train,
                                 labels_train)  # Resample training data
    model = xgb.XGBClassifier(booster='gblinear', learning_rate = 0.1, n_estimators = 1000)
    model.fit(features_resampled, labels_resampled)
    labels_pred = model.predict(features_test)
    labels_train_pred = model.predict(features_resampled)
    return (labels_resampled, labels_train_pred, labels_test, labels_pred)


def grid_search(filename):
    df = pd.read_csv(filename)
    features = df.loc[:, df.columns != "TenYearCHD"]
    labels = df["TenYearCHD"]
    (
        features_train,
        features_test,
        labels_train,
        labels_test,
    ) = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42
    )
    param_grid = {
        'classification__booster': ['gbtree', 'gblinear', 'dart'],
        'classification__learning_rate': [0.1, 0.01, 0.001],
        'classification__n_estimators': [100, 500, 1000],
    }
    model = Pipeline([
        ('sampling', SMOTE()),
        ('classification', xgb.XGBClassifier())
    ])
    grid_search = GridSearchCV(model, param_grid, scoring='f1',
                               n_jobs=-1, cv=5, verbose=1)
    grid_search.fit(features_train, labels_train)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print("Best Hyperparameters:", best_params)
    print("F1:", best_score)

import pandas as pd
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
        features, labels, test_size=0.2, stratify=labels, random_state=42
    )  # stratified data
    oversampler = RandomOverSampler(random_state=42)
    (
        features_resampled, labels_resampled
    ) = oversampler.fit_resample(features_train,
                                 labels_train)  # Resample training data
    model = xgb.XGBClassifier()
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
    )  # stratified data
    oversampler = RandomOverSampler(random_state=42)
    (
        features_resampled, labels_resampled
    ) = oversampler.fit_resample(features_train,
                                 labels_train)
    param_grid = {
        'booster': ['gbtree', 'gblinear', 'dart'],
        'learning_rate': [0.1, 0.01, 0.001],
        'n_estimators': [100, 500, 1000],
    }
    model = xgb.XGBClassifier()
    grid_search = GridSearchCV(model, param_grid, scoring='f1',
                               cv=5, verbose=1)
    grid_search.fit(features_resampled, labels_resampled)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    labels_pred = best_model.predict(features_test)
    f1 = f1_score(labels_test, labels_pred)
    print("Best Hyperparameters:", best_params)
    print("F1:", f1)

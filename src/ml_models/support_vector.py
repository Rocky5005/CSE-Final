from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler


def support_vector(filename: str):
    print("SVM: ")
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
    # Perform standardization on training data
    scaler = StandardScaler()
    scaled_features_train = scaler.fit_transform(features_resampled)
    # Apply the same standardization to testing data
    scaled_features_test = scaler.transform(features_test)
    model = SVC()
    model.fit(scaled_features_train, labels_resampled)
    labels_pred = model.predict(scaled_features_test)
    labels_train_pred = model.predict(scaled_features_train)
    return (labels_resampled, labels_train_pred, labels_test, labels_pred)


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
    )  # stratified data
    # Perform standardization on training data
    scaler = StandardScaler()
    scaled_features_train = scaler.fit_transform(features_train)
    # Apply the same standardization to testing data
    scaled_features_test = scaler.transform(features_test)
    model = SVC()
    param_grid = {
        'C': np.logspace(-5, 5, num=11),
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'degree': [2, 3, 4],
        'gamma': ['scale', 'auto', 0.1, 1.0],
        'coef0': [-1, 0, 1],
        'shrinking': [True, False],
        'probability': [True, False],
        'class_weight': [None, 'balanced', {0: 0.5, 1: 0.5}],
    }
    grid_search = GridSearchCV(model, param_grid, scoring='f1',
                               cv=5, verbose=1)
    grid_search.fit(scaled_features_train, labels_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    labels_pred = best_model.predict(scaled_features_test)
    f1 = f1_score(labels_test, labels_pred)
    print("Best Hyperparameters:", best_params)
    print("F1:", f1)

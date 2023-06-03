from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler, SMOTE


def logistic_regression(filename: str):
    print("Logistic Regression")
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
    oversampler = SMOTE(random_state=42)
    (
        features_resampled, labels_resampled
    ) = oversampler.fit_resample(features_train,
                                 labels_train)  # Resample training data
    # Perform standardization on training data
    scaler = StandardScaler()
    scaled_features_train = scaler.fit_transform(features_resampled)
    # Apply the same standardization to testing data
    scaled_features_test = scaler.transform(features_test)
    model = LogisticRegression(C=0.01, max_iter=50, penalty='l1', solver='saga')
    model.fit(scaled_features_train, labels_resampled)
    labels_pred = model.predict(scaled_features_test)
    labels_train_pred = model.predict(scaled_features_train)
    return (labels_resampled, labels_train_pred, labels_test, labels_pred)


# Tunes prediction threshold
def threshold_sample(scaled_features_train, scaled_features_test,
                     labels_train, labels_test):
    model = LogisticRegression()
    model.fit(scaled_features_train, labels_train)
    labels_pred = model.predict_proba(scaled_features_test)
    labels_pred_sample = (labels_pred[:, 1] > 0.5).astype(int)
    # Using the initial threshold of 0.5
    initial_f1_score = f1_score(labels_test, labels_pred_sample)
    best_threshold = 0.5
    best_f1_score = initial_f1_score
    for threshold in np.arange(0.1, 0.9, 0.1):
        labels_pred_sample = (labels_pred[:, 1] > threshold).astype(int)
        f1 = f1_score(labels_test, labels_pred_sample)
        if f1 > best_f1_score:
            best_threshold = threshold
            best_f1_score = f1
    end_labels_pred = (labels_pred[:, 1] > best_threshold).astype(int)
    print("Best threshold", best_threshold)
    return end_labels_pred


# Uses gridsearch to tune hyperparameters
def hyperparameter_search(filename):
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
                                 labels_train)#
    scaler = StandardScaler()
    scaled_features_train = scaler.fit_transform(features_resampled)
    # Apply the same standardization to testing data
    scaled_features_test = scaler.transform(features_test)
    grid_params = [
        {'solver': ['saga', 'liblinear'],
         'penalty': ['l1', 'l2'],
         'max_iter': [50, 100, 200, 500, 1000, 2500],
         'C': np.logspace(-3, 3, num=7)},
        {'solver': ['newton-cg', 'lbfgs'],
         'penalty': ['l2'],
         'max_iter': [50, 100, 200, 500, 1000, 2500],
         'C': np.logspace(-3, 3, num=7)},
    ]
    model = LogisticRegression()
    grid_search = GridSearchCV(model, grid_params, scoring='f1',
                               n_jobs=-1, cv=5, verbose=1)
    grid_search.fit(scaled_features_train, labels_resampled)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    labels_pred = best_model.predict(scaled_features_test)
    f1 = f1_score(labels_test, labels_pred)
    print("Best Hyperparameters:", best_params)
    print("F1:", f1)

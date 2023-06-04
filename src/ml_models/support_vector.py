from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline


def support_vector(filename: str):
    print("SVM: ")
    df = pd.read_csv(filename)
    features = df.loc[:, df.columns != "TenYearCHD"]
    labels = df["TenYearCHD"]
    oversampler = RandomOverSampler(random_state=42)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    (
        features_train,
        features_test,
        labels_train,
        labels_test,
    ) = train_test_split(
        scaled_features, labels, test_size=0.2,
        stratify=labels, random_state=42
    )  # stratified data
    oversampler = RandomOverSampler(random_state=42)
    (
        features_resampled, labels_resampled
    ) = oversampler.fit_resample(features_train,
                                 labels_train)  # Resample training data
    model = SVC()
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
    model = SVC()
    grid_params = {
        'classification__C': np.logspace(-3, 3, num=7),
        'classification__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'classification__gamma': ['scale', 'auto'],
        'classification__shrinking': [True, False],
        'classification__class_weight': [None, 'balanced', {0: 0.5, 1: 0.5}],
    }
    model = Pipeline([
        ('sampling', SMOTE()),
        ('scaling', StandardScaler()),
        ('classification', SVC())
    ])
    grid_search = GridSearchCV(model, grid_params, scoring='f1',
                               n_jobs=-1, cv=5, verbose=1)
    grid_search.fit(features_train, labels_train)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print("Best Hyperparameters:", best_params)
    print("F1:", best_score)

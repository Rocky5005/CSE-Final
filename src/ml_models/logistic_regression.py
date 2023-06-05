import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


def logistic_regression(filename: str, pipeline):
    print(pipeline.steps)
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
    pipeline.fit(features_train, labels_train)
    labels_pred = pipeline.predict(features_test)
    labels_train_pred = pipeline.predict(features_train)
    return (labels_train, labels_train_pred, labels_test, labels_pred)


# Uses gridsearch to tune hyperparameters
def hyperparameter_search(filename, model):
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
    grid_params = [
        {'classification__solver': ['saga', 'liblinear'],
         'classification__penalty': ['l1', 'l2'],
         'classification__max_iter': [50, 100, 200, 500, 1000, 2500],
         'classification__C': np.logspace(-3, 3, num=7)},
        {'classification__solver': ['newton-cg', 'lbfgs'],
         'classification__penalty': ['l2'],
         'classification__max_iter': [50, 100, 200, 500, 1000, 2500],
         'classification__C': np.logspace(-3, 3, num=7)},
    ]
    grid_search = GridSearchCV(model, grid_params, scoring='f1',
                               n_jobs=-1, cv=5, verbose=1)
    grid_search.fit(features_train, labels_train)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print(model.steps)
    print("Best Hyperparameters:", best_params)
    print("F1:", best_score)

from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt


def logistic_regression(filename: str):
    print("Logistic Regression")
    df = pd.read_csv(filename)
    features = df.loc[:, df.columns != "TenYearCHD"]
    labels = df["TenYearCHD"]
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
    oversampler = SMOTE(random_state=42)
    (
        features_resampled, labels_resampled
    ) = oversampler.fit_resample(features_train,
                                 labels_train)  # Resample training data
    model = LogisticRegression(C=0.001, max_iter=500,
                               penalty='l2', solver='saga')
    model.fit(features_resampled, labels_resampled)
    labels_pred = model.predict(features_test)
    labels_train_pred = model.predict(features_resampled)
    return (labels_resampled, labels_train_pred, labels_test, labels_pred)


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
    model = Pipeline([
        ('sampling', SMOTE()),
        ('scaling', StandardScaler()),
        ('classification', LogisticRegression())
    ])
    grid_search = GridSearchCV(model, grid_params, scoring='f1',
                               n_jobs=-1, cv=5, verbose=1)
    grid_search.fit(features_train, labels_train)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print("Best Hyperparameters:", best_params)
    print("F1:", best_score)


def plot_learning_curve(features, labels, model):
    train_sizes, train_scores, test_scores = \
        learning_curve(model, features, labels, cv=10, scoring='accuracy',
                       n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 50))
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    plt.plot(train_sizes, train_mean, color='blue', label='Training Accuracy')
    plt.plot(train_sizes, test_mean, color='green',
             linestyle='--', label='Validation Accuracy')
    plt.title('Learning Curve')
    plt.xlabel('Training Data Size')
    plt.ylabel('Model accuracy')
    plt.grid()
    plt.legend(loc='lower right')
    plt.show()

from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from imblearn.pipeline import Pipeline


def k_nearest(filename: str):
    print("K-Nearest Neighbors:")
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
                                 labels_train)
    model = KNeighborsClassifier()
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
        'classification__n_neighbors': range(1, 15),  # K values
        'classification__weights': ['uniform', 'distance'],  # Weight metrics
        'classification__metric': ['euclidean', 'manhattan',
                   'chebyshev', 'minkowski'],  # Distance metrics
        'classification__algorithm': ['brute', 'kd_tree', 'ball_tree', 'auto']  # Algorithms
    }
    model = Pipeline([
        ('sampling', SMOTE()),
        ('scaling', StandardScaler()),
        ('classification', KNeighborsClassifier())
    ])
    grid_search = GridSearchCV(model, param_grid, scoring='f1',
                               cv=5, verbose=1)
    grid_search.fit(features_train, labels_train)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print("Best Hyperparameters:", best_params)
    print("F1:", best_score)

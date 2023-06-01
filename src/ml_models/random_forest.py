from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV


def random_forest(filename: str):
    print("Random Forest")
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
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(scaled_features_train, labels_train)
    labels_pred = model.predict(scaled_features_test)
    labels_train_pred = model.predict(scaled_features_train)
    return (labels_train, labels_train_pred, labels_test, labels_pred)
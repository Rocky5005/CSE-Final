from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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
    # Perform standardization on training data
    scaler = StandardScaler()
    scaled_features_train = scaler.fit_transform(features_train)
    # Apply the same standardization to testing data
    scaled_features_test = scaler.transform(features_test)
    model = LogisticRegression()
    labels_pred = model.predict(scaled_features_test)
    labels_train_pred = model.predict(scaled_features_train)
    labels_train_df = pd.DataFrame(labels_train_pred)
    plt.hist(labels_train_df)
    return (labels_train, labels_train_pred, labels_test, labels_pred)

def threshold_sample(scaled_features_train, scaled_features_test, labels_train, labels_test):
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


def hyperparameter_search(scaled_features_train, scaled_features_test, labels_train, labels_test):
    
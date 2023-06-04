import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE


def compare_performance_with_outliers(filename):
    # Load data from file
    data = pd.read_csv(filename)

    # FIX TARGET AND INPUT FEATURES
    target_variable = "TenYearCHD"
    numerical_features = ["cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]

    # Identify outliers
    z_scores = stats.zscore(data[numerical_features])
    threshold = 3
    outlier_indices = np.where(np.abs(z_scores) > threshold)[0]

    # Create two versions of the dataset
    data_without_outliers = data.drop(outlier_indices)
    X_with_outliers = data[numerical_features]
    y_with_outliers = data[target_variable]
    X_without_outliers = data_without_outliers[numerical_features]
    y_without_outliers = data_without_outliers[target_variable]

    # Train models on the dataset with outliers
    (
        X_train_with_outliers,
        X_test_with_outliers,
        y_train_with_outliers,
        y_test_with_outliers,
    ) = train_test_split(
        X_with_outliers, y_with_outliers, test_size=0.2, random_state=42
    )
    logreg_with_outliers = LogisticRegression()
    logreg_with_outliers.fit(X_train_with_outliers, y_train_with_outliers)
    y_pred_with_outliers = logreg_with_outliers.predict(X_test_with_outliers)

    # Calculate evaluation metrics for dataset with outliers
    accuracy_with_outliers = accuracy_score(
        y_test_with_outliers, y_pred_with_outliers
    )
    precision_with_outliers = precision_score(
        y_test_with_outliers, y_pred_with_outliers
    )
    recall_with_outliers = recall_score(
        y_test_with_outliers, y_pred_with_outliers
    )
    f1_score_with_outliers = f1_score(
        y_test_with_outliers, y_pred_with_outliers
    )

    # Remove outliers and create dataset without outliers
    (
        X_train_without_outliers,
        X_test_without_outliers,
        y_train_without_outliers,
        y_test_without_outliers,
    ) = train_test_split(
        X_without_outliers, y_without_outliers, test_size=0.2, random_state=42
    )

    # Train models on the dataset without outliers
    logreg_without_outliers = LogisticRegression()
    logreg_without_outliers.fit(
        X_train_without_outliers, y_train_without_outliers
    )
    y_pred_without_outliers = logreg_without_outliers.predict(
        X_test_without_outliers
    )

    # Calculate evaluation metrics for dataset without outliers
    accuracy_without_outliers = accuracy_score(
        y_test_without_outliers, y_pred_without_outliers
    )
    precision_without_outliers = precision_score(
        y_test_without_outliers, y_pred_without_outliers
    )
    recall_without_outliers = recall_score(
        y_test_without_outliers, y_pred_without_outliers
    )
    f1_score_without_outliers = f1_score(
        y_test_without_outliers, y_pred_without_outliers
    )

    # Compare performance metrics
    print("Performance with outliers:")
    print("Accuracy:", accuracy_with_outliers)
    print("Precision:", precision_with_outliers)
    print("Recall:", recall_with_outliers)
    print("F1-score:", f1_score_with_outliers)

    print("\nPerformance without outliers:")
    print("Accuracy:", accuracy_without_outliers)
    print("Precision:", precision_without_outliers)
    print("Recall:", recall_without_outliers)
    print("F1-score:", f1_score_without_outliers)


def apply_rfe(filename):

    data = pd.read_csv(filename)

    X = data.drop("TenYearCHD", axis=1)
    y = data["TenYearCHD"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()

    n_features_to_select = 5

    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
    rfe.fit(X_train, y_train)

    selected_features = rfe.support_

    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]

    model.fit(X_train_selected, y_train)

    y_pred = model.predict(X_test_selected)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Selected Features: ", selected_features)

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier


def logistic_regression(filename: str) -> None:  # baseline model
    print("Logistic Regression")
    df = pd.read_csv(filename)
    features = df.loc[:, df.columns != 'TenYearCHD']
    labels = df['TenYearCHD']
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2,
                         stratify=labels, random_state=42)  # stratified data
    model = LogisticRegression(solver='lbfgs', max_iter=4000)
    model.fit(features_train, labels_train)
    labels_pred = model.predict(features_test)
    cm = confusion_matrix(labels_test, labels_pred)
    print(cm)
    accuracy = accuracy_score(labels_test, labels_pred)
    print("Accuracy: ", accuracy)
    report = classification_report(labels_test, labels_pred)
    print("Classification Report:")
    print(report)


def naive_bayes(filename: str) -> None:
    print("Naive Bayes:")
    df = pd.read_csv(filename)
    features = df.loc[:, df.columns != 'TenYearCHD']
    labels = df['TenYearCHD']
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2,
                         stratify=labels, random_state=42)  # stratified data
    model = BernoulliNB()
    model.fit(features_train, labels_train)
    labels_pred = model.predict(features_test)
    cm = confusion_matrix(labels_test, labels_pred)
    print(cm)
    accuracy = accuracy_score(labels_test, labels_pred)
    print("Accuracy: ", accuracy)
    report = classification_report(labels_test, labels_pred)
    print("Classification Report:")
    print(report)


def k_nearest(filename: str) -> None:
    print("K-Nearest Neighbors:")
    df = pd.read_csv(filename)
    features = df.loc[:, df.columns != 'TenYearCHD']
    labels = df['TenYearCHD']
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2,
                         stratify=labels, random_state=42)  # stratified data
    k = 3  # change value using testing
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(features_train, labels_train)
    labels_pred = model.predict(features_test)
    cm = confusion_matrix(labels_test, labels_pred)
    print(cm)
    accuracy = accuracy_score(labels_test, labels_pred)
    print("Accuracy: ", accuracy)
    report = classification_report(labels_test, labels_pred)
    print("Classification Report:")
    print(report)


def support_vector(filename: str) -> None:
    print("SVM: ")
    df = pd.read_csv(filename)
    features = df.loc[:, df.columns != 'TenYearCHD']
    labels = df['TenYearCHD']
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2,
                         stratify=labels, random_state=42)  # stratified data
    model = SVC(kernel='linear')
    model.fit(features_train, labels_train)
    labels_pred = model.predict(features_test)
    cm = confusion_matrix(labels_test, labels_pred)
    print(cm)
    accuracy = accuracy_score(labels_test, labels_pred)
    print("Accuracy: ", accuracy)


def gradient_boost(filename: str) -> None:
    print("XGBoost: ")
    df = pd.read_csv(filename)
    features = df.loc[:, df.columns != 'TenYearCHD']
    labels = df['TenYearCHD']
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2,
                         stratify=labels, random_state=42)
    model = xgb.XGBClassifier(objective='binary:logistic')
    model.fit(features_train, labels_train)
    labels_pred = model.predict(features_test)
    cm = confusion_matrix(labels_test, labels_pred)
    print(cm)
    accuracy = accuracy_score(labels_test, labels_pred)
    print("Accuracy: ", accuracy)
    report = classification_report(labels_test, labels_pred)
    print("Classification Report:")
    print(report)


def random_forest(filename: str) -> None:
    print("Random Forest: ")
    df = pd.read_csv(filename)
    features = df.loc[:, df.columns != 'TenYearCHD']
    labels = df['TenYearCHD']
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2,
                         stratify=labels, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features_train, labels_train)
    labels_pred = model.predict(features_test)
    cm = confusion_matrix(labels_test, labels_pred)
    print(cm)
    accuracy = accuracy_score(labels_test, labels_pred)
    print("Accuracy: ", accuracy)
    report = classification_report(labels_test, labels_pred)
    print("Classification Report:")
    print(report)


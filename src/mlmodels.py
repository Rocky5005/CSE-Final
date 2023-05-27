import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def logistic_regression(filename: str) -> None:
    df = pd.read_csv(filename)
    features = df.loc[:, df.columns != 'TenYearCHD']
    labels = df['TenYearCHD']
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2,
                         stratify=labels, random_state=42)
    model = LogisticRegression()
    model.fit(features_train, labels_train)
    labels_pred = model.predict(features_test)
    cm = confusion_matrix(labels_test, labels_pred)
    print(cm)
    accuracy = accuracy_score(labels_test, labels_pred)
    print("Accuracy: ", accuracy)

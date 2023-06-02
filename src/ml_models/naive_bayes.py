from sklearn.naive_bayes import BernoulliNB
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler


def naive_bayes(filename: str):
    print("Naive Bayes:")
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
                                 labels_train)  # Resample training data
    model = BernoulliNB()
    model.fit(features_resampled, labels_resampled)
    labels_pred = model.predict(features_test)
    labels_train_pred = model.predict(features_resampled)
    return (labels_resampled, labels_train_pred, labels_test, labels_pred)


def find_alpha(filename: str):  # alpha does not seem to have much of an impact
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
                                 labels_train)
    alpha_values = [0.001, 0.1, 0.01, 0.0001, 1, 10]
    best_alpha = None
    best_f1_score = 0.0
    for alpha in alpha_values:
        model = BernoulliNB(alpha=alpha)
        model.fit(features_resampled, labels_resampled)
        labels_pred = model.predict(features_test)
        f1 = f1_score(labels_test, labels_pred)
        if f1 > best_f1_score:
            best_f1_score = f1
            best_alpha = alpha
    print("Best alpha", best_alpha)
    print("Best f1:", best_f1_score)


def find_threshold(filename):
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
                                 labels_train)
    model = BernoulliNB()
    model.fit(features_resampled, labels_resampled)
    labels_pred_prob = model.predict_proba(features_test)[:, 1]
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    best_threshold = None
    best_f1_score = 0.0
    for threshold in thresholds:
        labels_pred = (labels_pred_prob >= threshold).astype(int)
        f1 = f1_score(labels_test, labels_pred)
        if f1 > best_f1_score:
            best_f1_score = f1
            best_threshold = threshold
    best_model = BernoulliNB()
    best_model.fit(features_resampled, labels_resampled)
    labels_test_pred_prob = best_model.predict_proba(features_test)[:, 1]
    labels_pred = (labels_test_pred_prob >= best_threshold).astype(int)
    test_f1_score = f1_score(labels_test, labels_pred)
    print("Best threshold", best_threshold)
    print("Test f1", test_f1_score)

# Some kind of cross validation
# Feature selection

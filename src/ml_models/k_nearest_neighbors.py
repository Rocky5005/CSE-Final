from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

def k_nearest(filename: str):
    print("K-Nearest Neighbors:")
    df = pd.read_csv(filename)
    oversampler = RandomOverSampler(random_state=42)
    features_resampled, labels_resampled = oversampler.fit_resample(features,
                                                                    labels)
    features = df.loc[:, df.columns != "TenYearCHD"]
    labels = df["TenYearCHD"]
    (
        features_train,
        features_test,
        labels_train,
        labels_test,
    ) = train_test_split(
        features_resampled, labels_resampled, test_size=0.2, random_state=42
    )  # stratified data
    k = 3  # change value using testing
    # Perform standardization on training data
    scaler = StandardScaler()
    scaled_features_train = scaler.fit_transform(features_train)

    # Apply the same standardization to testing data
    scaled_features_test = scaler.transform(features_test)

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(scaled_features_train, labels_train)
    labels_pred = model.predict(scaled_features_test)
    model.fit(features_train, labels_train)
    labels_train_pred = model.predict(features_train)
    return (labels_train, labels_train_pred, labels_test, labels_pred)

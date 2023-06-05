from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os


def main():
    file_locate('main.py')
    print("This calculator is meant to be used as a tool, and should never replace the advice of licensed medical professional.")
    features = collect_input()
    model(features, 'cleaned-framingham.csv')


def file_locate(__file__: str) -> None:
    pathstr = os.path.realpath(__file__)
    directory = os.path.dirname(pathstr)
    directory = directory.replace('src', 'data')
    os.chdir(directory)


def collect_input():
    male = input("Are you a male? 1 for yes, 0 for no. ")
    age = input("What is your age? ")
    cigsPerDay = input("How many cigarettes do you smoke per day? ")
    BPMeds = input("Are you taking blood pressure medication? 1 for yes, 0 for no. ")
    prevalentStroke = input("Does your family have a history of strokes? 1 for yes, 0 for no. ")
    prevalentHyp = input("Does your family have a history of hypertension? 1 for yes, 0 for no. ")
    diabetes = input("Do you have diabetes? 1 for yes, 0 for no. ")
    totChol = input("What is your total cholesterol? (mg/dL) ")
    sysBP = input("What is your systolic blood pressure? (mmHg) ")
    diaBP = input("What is your diastolic blood pressure? (mmHg) ")
    BMI = input("What is your BMI? ")
    heartRate = input("What is your heart rate? (beats/min) ")
    glucose = input("What is your blood glucose level? (mg/dL) ")
    features = np.array([[male, age, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp,
                          diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose]])
    return features

def model(input_features, filename):
    df = pd.read_csv(filename)
    features = df.loc[:, df.columns != "TenYearCHD"]
    labels = df["TenYearCHD"]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features.values)
    scaled_input = scaler.transform(input_features)
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
    model = LogisticRegression()
    model.fit(features_resampled, labels_resampled)
    prediction = model.predict(scaled_input)
    if prediction[0] == 0:
        print("There is a low 10 year risk of future heart disease.")
    if prediction[0] == 1:
        print("There is a high 10 year risk of future heart disease.")


if __name__ == '__main__':
    main()
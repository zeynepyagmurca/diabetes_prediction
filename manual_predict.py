import joblib
import pandas as pd
from src.new_data_preprocessing import new_data_prep
import json


def predictnewdata(data):
    # Load the model, encoder, and scaler
    model = joblib.load("models/voting_clf.pkl")

    # Apply the necessary preprocessing steps

    df_processed = new_data_prep(data)  # Make sure to apply the same preprocessing to new data

    # Make predictions

    prediction = model.predict(df_processed)
    return prediction


# Example usage
new_data = {'PREGNANCIES': [2],
    'GLUCOSE': [120],
    'BLOODPRESSURE': [70],
    'SKINTHICKNESS': [25],
    'INSULIN': [80],
    'BMI': [28.5],
    'DIABETESPEDIGREEFUNCTION': [0.35],
    'AGE': [32]}

prediction = predictnewdata(new_data)
print(prediction)


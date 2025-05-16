import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import json
from src.utils import grab_col_names, replace_with_thresholds, outlier_thresholds


def diabetes_data_prep(dataframe):
    dataframe.columns = [col.upper() for col in dataframe.columns]

    # Feature Engineering (Öznitelik işleme)
    dataframe['NEW_GLUCOSE_CAT'] = pd.cut(x=dataframe['GLUCOSE'], bins=[-1, 139, 200], labels=["normal", "prediabetes"])
    dataframe['NEW_AGE_CAT'] = pd.cut(x=dataframe['AGE'], bins=[-1, 35, 55, 100], labels=["young", "middleage", "old"])
    dataframe['NEW_BMI_RANGE'] = pd.cut(x=dataframe['BMI'], bins=[-1, 18.5, 24.9, 29.9, 100],
                                        labels=["underweight", "healthy", "overweight", "obese"])
    dataframe['NEW_BLOODPRESSURE'] = pd.cut(x=dataframe['BLOODPRESSURE'], bins=[-1, 79, 89, 123],
                                            labels=["normal", "hs1", "hs2"])

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)



    # OneHotEncoder'ı fit et ve dönüştür
    encoder = OneHotEncoder(drop=None, handle_unknown='ignore', sparse_output=False)
    dataframe_encoded = encoder.fit_transform(dataframe[cat_cols])


    # Kolon isimlerini al
    encoded_col_names = encoder.get_feature_names_out(cat_cols)

    # DataFrame'e dönüştür
    dataframe_encoded = pd.DataFrame(dataframe_encoded, columns=encoded_col_names)

    # DataFrame'i birleştir
    dataframe = dataframe.drop(columns=cat_cols).reset_index(drop=True)
    dataframe = pd.concat([dataframe, dataframe_encoded], axis=1)

    limits={}
    for col in num_cols:
        low,up = outlier_thresholds(dataframe,col)
        limits[col]={"low":low,"up":up}
    joblib.dump(limits,"models/outlier_limits.pkl")

    for col in num_cols:
        replace_with_thresholds(dataframe,col)

    scaler = StandardScaler()
    dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])

    y = dataframe["OUTCOME"]
    X = dataframe.drop(["OUTCOME"], axis=1)


    # Kaydedilecek dosyalar
    joblib.dump(encoder, "models/encoder.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    with open("models/num_cols.json", "w") as f:
        json.dump(num_cols, f)

    with open('models/cat_cols.json', 'w') as f:
        json.dump(cat_cols, f)

    return X, y


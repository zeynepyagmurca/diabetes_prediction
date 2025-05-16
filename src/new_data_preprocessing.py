import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import json
from src.utils import grab_col_names,replace_with_th
import numpy as np



# thresholds = json.loads("models/thresholds.json")

def new_data_prep(new_data):
    new_array = np.array(new_data).reshape(1, -1)
    new_df = pd.DataFrame(new_array,columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'])


    new_df.columns = [col.upper() for col in new_df.columns]
    encoder = joblib.load("models/encoder.pkl")
    scaler = joblib.load("models/scaler.pkl")
    num_cols = json.load(open('models/num_cols.json'))
    cat_cols = json.load(open('models/cat_cols.json'))

    # Feature Engineering (Öznitelik işleme)
    new_df['NEW_GLUCOSE_CAT'] = pd.cut(x=new_df['GLUCOSE'], bins=[-1, 139, 200], labels=["normal", "prediabetes"])
    new_df['NEW_AGE_CAT'] = pd.cut(x=new_df['AGE'], bins=[-1, 35, 55, 100], labels=["young", "middleage", "old"])
    new_df['NEW_BMI_RANGE'] = pd.cut(x=new_df['BMI'], bins=[-1, 18.5, 24.9, 29.9, 100],
                                     labels=["underweight", "healthy", "overweight", "obese"])
    new_df['NEW_BLOODPRESSURE'] = pd.cut(x=new_df['BLOODPRESSURE'], bins=[-1, 79, 89, 123],
                                         labels=["normal", "hs1", "hs2"])

    new_df_encoded = encoder.transform(new_df[cat_cols])

    # 5. Feature isimlerini al
    encoded_col_names = encoder.get_feature_names_out(cat_cols)

    # 6. Yeni DataFrame'e çevir
    new_df_encoded = pd.DataFrame(new_df_encoded, columns=encoded_col_names, index=new_df.index)

    # 7. Orijinal kategorik sütunları çıkar, encode edilmişleri ekle
    new_df = new_df.drop(columns=cat_cols)
    new_df = pd.concat([new_df, new_df_encoded], axis=1)

    limits = joblib.load('models/outlier_limits.pkl')
    for col in num_cols:
        replace_with_th(new_df,limits,col)

    new_df[num_cols] = scaler.fit_transform(new_df[num_cols])

    return new_df





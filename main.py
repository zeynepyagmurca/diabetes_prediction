from src.data_preprocessing import diabetes_data_prep
from src.model_training import hyperparameter_optimization, voting_classifier

import pandas as pd


def main():
    # Ham veriyi yükle
    df = pd.read_csv("data/diabetes.csv")

    # Veriyi işle ve model için hazırla
    X, y = diabetes_data_prep(df)

    best_models= hyperparameter_optimization(X, y, cv=3, scoring="roc_auc")


    voting_clf=voting_classifier(best_models,X, y)


if __name__ == "__main__":
    print("Proje çalıştırılıyor...")
    main()



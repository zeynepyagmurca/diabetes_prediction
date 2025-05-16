import joblib
import pandas as pd
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from config import knn_params, cart_params, rf_params, lightgbm_params, classifiers
from src.data_preprocessing import diabetes_data_prep  # Verinin hazır hale getirilmesi



# Hyperparameter optimization
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.base import clone

def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}

    # classifiers listesindeki her modelin hiperparametrelerini optimize ediyoruz
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model

    return best_models



# Modeli eğitme ve VotingClassifier ile birleştirme
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_validate
import joblib


def voting_classifier(best_models, X, y):
    print("Voting Classifier...")

    # Voting Classifier modelini oluşturuyoruz
    voting_clf = VotingClassifier(
        estimators=[('KNN', best_models["KNN"]),
                    ('RF', best_models["RF"]),
                    ('LightGBM', best_models["LightGBM"])],
        voting='soft'
    ).fit(X, y)

    # Performansı çapla (cross-validation ile)
    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])

    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")

    # Modeli kaydet
    joblib.dump(voting_clf, "models/voting_clf.pkl")
    print("Model saved as voting_clf.pkl")

    # Modeli döndür
    return voting_clf





# Hyperparameter configurations
import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

knn_params = {
    "n_neighbors": range(2, 50)
}


cart_params = {
    'max_depth': range(1, 20),
    "min_samples_split": range(2, 30)
}

rf_params = {
    "max_depth": [8, 15, None],
    "max_features": [5, 7, "auto"],
    "min_samples_split": [15, 20],
    "n_estimators": [200, 300]
}



lightgbm_params = {
    "learning_rate": [0.01, 0.1],
    "n_estimators": [300, 500],
    "colsample_bytree": [0.7, 1]
}


classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)]
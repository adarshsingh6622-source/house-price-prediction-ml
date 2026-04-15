import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

from lightgbm import LGBMRegressor
import os



def train_lgbm(df):

    os.makedirs("models", exist_ok=True)

    X = df.drop('SalePrice', axis=1)
    y = np.log1p(df['SalePrice'])

    X = pd.get_dummies(X)

    selector = SelectKBest(f_regression, k=60)
    X_selected = selector.fit_transform(X, y)

    selected_cols = X.columns[selector.get_support()]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8
    )

    model.fit(X_train, y_train)

    scores = cross_val_score(model, X_scaled, y, cv=3)
    print("CV Score:", scores.mean())

    

    pickle.dump(model, open("models/lgbm_model.pkl", "wb"))
    pickle.dump(scaler, open("models/scaler.pkl", "wb"))
    pickle.dump(selected_cols, open("models/columns.pkl", "wb"))

    return model, X_test, y_test
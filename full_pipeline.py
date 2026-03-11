# HOUSE PRICE PREDICTION PIPELINE

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

print("Libraries Loaded")

# 2. LOAD DATA

df = pd.read_csv("Data/train.csv")
print("Dataset Loaded")
print("Shape:", df.shape)
print(df.head())

#3. BASIC DATA INSPECTION
print(df.info())
print(df.describe())

# 4. MISSING VALUES

missing = df.isnull().sum()
missing = missing[missing > 0]
print("Missing Values")
print(missing)

# 5. DROP MISSING
df = df.dropna()
print("After cleaning shape:", df.shape)

# 6. EDA VISUALIZATION

df = pd.read_csv("Data/train.csv")
df = df.drop(columns=["PoolQC", "Fence", "MiscFeature"])
df.fillna(df.mean(numeric_only=True), inplace=True)
for col in df.select_dtypes(include="object"):
    df[col].fillna(df[col].mode()[0], inplace=True)
plt.figure()
sns.histplot(df["SalePrice"], kde=True)
plt.title("SalePrice Distribution")
plt.show()
plt.figure()
sns.scatterplot(x=df["GrLivArea"], y=df["SalePrice"])
plt.title("Living Area vs Price")
plt.show()
plt.savefig("saleprice_distribution.png")
plt.close()

# 7. FEATURE / TARGET

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# 8. FEATURE ENGINEERING
X = pd.get_dummies(X)
print("Feature after encoding:", X.shape)

# 9. TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("Train Shape:", X_train.shape)

# 10. FEATURE SCALING
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 11. MODEL COMPARISON

lr = LinearRegression()
rf = RandomForestRegressor()
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)
print("Linear Regression R2:", r2_score(y_test, lr_pred))
print("Random Forest R2:", r2_score(y_test, rf_pred))

# 12. HYPERPARAMETER TUNING

param_grid = {

    "n_estimators": [100, 200],

    "max_depth": [10, 20]

}


grid = GridSearchCV(

    RandomForestRegressor(),

    param_grid,

    cv=3

)


grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("Best Model:", best_model)

# 13. FINAL PREDICTION

pred = best_model.predict(X_test)

# 14. MODEL EVALUATION

print("R2 Score:", r2_score(y_test, pred))

print("MAE:", mean_absolute_error(y_test, pred))

# 15. FEATURE IMPORTANCE

importance = best_model.feature_importances_
plt.figure()
plt.bar(range(len(importance)), importance)
plt.title("Feature Importance")
plt.show()

# 16. SAVE MODEL

pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(X.columns, open("features.pkl", "wb"))
print("Model Saved Successfully")



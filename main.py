from src.preprocessing import load_data, clean_data
from src.feature_engineering import feature_engineering
from src.train_lgbm import train_lgbm
from src.evaluate import evaluate

df = load_data("data/train.csv")
df = clean_data(df)
df = feature_engineering(df)

model, X_test, y_test = train_lgbm(df)
evaluate(model, X_test, y_test)
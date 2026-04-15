from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

def evaluate(model, X_test, y_test):

    pred = model.predict(X_test)

    pred = np.expm1(pred)
    y_test = np.expm1(y_test)

    print("R2:", r2_score(y_test, pred))
    print("MAE:", mean_absolute_error(y_test, pred))
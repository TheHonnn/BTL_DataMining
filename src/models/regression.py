# src/models/regression.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


def prepare_regression_data(df):
    """
    Chuẩn bị dữ liệu cho regression
    """

    features = [
        "age",
        "campaign",
        "previous",
        "emp.var.rate",
        "cons.price.idx",
        "euribor3m"
    ]

    X = df[features].dropna()

    # ví dụ dự đoán chỉ số tài chính
    y = df.loc[X.index, "nr.employed"]

    return X, y


def train_models(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "XGBRegressor": XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1
        )
    }

    results = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)

        rmse = np.sqrt(mean_squared_error(y_test, pred))

        results[name] = {
            "MAE": mae,
            "RMSE": rmse
        }

    return results
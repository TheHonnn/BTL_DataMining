from src.data.loader import load_raw_data
from src.features.builder import build_features
from src.models.regression import prepare_regression_data, train_models


df = load_raw_data()

df = build_features(df)

X, y = prepare_regression_data(df)

results = train_models(X, y)

print("\nRegression Results")

for model, metric in results.items():
    print(model, metric)
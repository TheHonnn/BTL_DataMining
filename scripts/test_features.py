from src.data.loader import load_raw_data
from src.features.builder import build_features, build_product_basket

df = load_raw_data()

df = build_features(df)

print(df.head())

basket = build_product_basket(df)

print("\nProduct basket:")
print(basket.head())
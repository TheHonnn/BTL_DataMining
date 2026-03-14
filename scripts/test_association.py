from src.data.loader import load_raw_data
from src.features.builder import build_features, build_product_basket
from src.mining.association import (
    run_apriori,
    generate_rules,
    get_top_rules,
    recommend_product_bundle
)

df = load_raw_data()

df = build_features(df)

basket = build_product_basket(df)

frequent = run_apriori(basket)

rules = generate_rules(frequent)

top_rules = get_top_rules(rules)

print("\nTop Cross-sell Rules:")
print(top_rules)

bundles = recommend_product_bundle(rules)

print("\nRecommended Product Bundles:")
print(bundles.head())
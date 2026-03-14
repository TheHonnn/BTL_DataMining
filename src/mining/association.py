# src/mining/association.py

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def mine_association_rules(basket_df, min_support=0.05, min_confidence=0.3):
    """
    Khai phá luật kết hợp (Association Rules)
    """

    # Frequent Itemsets
    frequent_itemsets = apriori(
        basket_df,
        min_support=min_support,
        use_colnames=True
    )

    # Generate rules
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )

    # Sort theo lift
    rules = rules.sort_values(by="lift", ascending=False)

    return rules


def get_top_rules(rules, n=10):
    """
    Lấy top luật có lift cao nhất
    """

    return rules[[
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift"
    ]].head(n)
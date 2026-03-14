# src/features/builder.py

import pandas as pd
from sklearn.preprocessing import StandardScaler


def build_features(df):
    """
    Tạo các feature mới phục vụ Data Mining
    """

    df = df.copy()

    # ===== Nhóm tuổi =====
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 50, 65, 100],
        labels=["young", "adult", "mid_age", "senior", "elder"]
    )

    # ===== Khách hàng có vay =====
    df["has_loan"] = (
        (df["housing"] == "yes") |
        (df["loan"] == "yes")
    ).astype(int)

    # ===== Tổng số lần liên hệ =====
    df["total_contacts"] = df["campaign"] + df["previous"]

    # ===== Cường độ marketing =====
    df["contact_rate"] = df["campaign"] / (df["previous"] + 1)

    # ===== Khách hàng tiềm năng =====
    df["potential_client"] = (
        df["poutcome"] == "success"
    ).astype(int)

    return df


def build_product_basket(df):
    """
    Tạo giỏ sản phẩm cho Association Rules
    """

    basket = pd.DataFrame()

    basket["housing_loan"] = (df["housing"] == "yes").astype(int)
    basket["personal_loan"] = (df["loan"] == "yes").astype(int)
    basket["subscribed_deposit"] = (df["y"] == "yes").astype(int)

    basket = basket.astype(bool)

    return basket


def scale_numeric_features(df, columns):
    """
    Chuẩn hóa dữ liệu cho clustering/regression
    """

    scaler = StandardScaler()

    df_scaled = df.copy()

    df_scaled[columns] = scaler.fit_transform(df[columns])

    return df_scaled, scaler
# src/data/loader.py
import pandas as pd
import yaml


def load_config(config_path="configs/params.yaml"):
    """Đọc file cấu hình YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw_data(config_path="configs/params.yaml"):
    """Đọc dữ liệu gốc từ CSV."""
    config = load_config(config_path)
    raw_path = config["data"]["raw_path"]

    try:
        df = pd.read_csv(raw_path, sep=";")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {raw_path}")


def check_schema(df):
    """Kiểm tra schema và giá trị thiếu."""
    print("\n--- SCHEMA DỮ LIỆU ---")
    df.info()

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if not missing.empty:
        print("\n--- GIÁ TRỊ THIẾU ---")
        print(missing)
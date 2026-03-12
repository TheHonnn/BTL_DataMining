import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def replace_unknown(df):
    """Thay thế 'unknown' thành NaN và điền giá trị thiếu bằng Median/Mode."""
    df_cleaned = df.replace('unknown', np.nan)
    
    # Phân tách cột số và cột chữ để xử lý điền giá trị thiếu hàng loạt (vectorized)
    num_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns

    if len(num_cols) > 0:
        df_cleaned[num_cols] = df_cleaned[num_cols].fillna(df_cleaned[num_cols].median())
    if len(cat_cols) > 0:
        df_cleaned[cat_cols] = df_cleaned[cat_cols].fillna(df_cleaned[cat_cols].mode().iloc[0])
        
    return df_cleaned

def encode_features(df):
    """Mã hóa nhãn 'y' và One-Hot cho các biến phân loại."""
    df_encoded = df.copy()
    
    # Mã hóa nhãn mục tiêu 'y'
    if 'y' in df_encoded.columns:
        df_encoded['y'] = LabelEncoder().fit_transform(df_encoded['y'])
    
    # Lấy danh sách cột phân loại và thực hiện One-Hot Encoding
    cat_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        # Thêm dtype=int để kết quả là 0/1 thay vì True/False (hỗ trợ tốt hơn cho các model)
        df_encoded = pd.get_dummies(df_encoded, columns=cat_cols, drop_first=True, dtype=int)
        
    return df_encoded

def scale_numerical(df):
    """Chuẩn hóa các biến số bằng StandardScaler."""
    df_scaled = df.copy()
    
    # Lấy danh sách cột số nguyên thủy (tránh scale các cột One-Hot hoặc nhãn 'y')
    num_cols = df_scaled.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'y' in num_cols:
        num_cols.remove('y')
        
    if num_cols:
        df_scaled[num_cols] = StandardScaler().fit_transform(df_scaled[num_cols])
        
    return df_scaled

def run_cleaning_pipeline(raw_path, processed_path):
    """Chạy toàn bộ luồng tiền xử lý và lưu file."""
    # Đọc dữ liệu
    df_raw = pd.read_csv(raw_path, sep=';')
    
    # Chạy chuỗi xử lý (pipeline)
    df = replace_unknown(df_raw)
    df = encode_features(df)
    df_final = scale_numerical(df)
    
    # Lưu file
    df_final.to_csv(processed_path, index=False)
    print(f"Đã xử lý xong và lưu dữ liệu sạch tại: {processed_path}")
    print(f"Kích thước sau xử lý: {df_final.shape}")
    
    return df_final
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, average_precision_score

def split_and_balance_data(df, target_col='y', test_size=0.2, random_state=42):
    """Chia tập train/test và cân bằng dữ liệu bằng SMOTE trên tập Train."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Chia tập train/test (stratify để giữ nguyên tỷ lệ nhãn gốc trên tập test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Áp dụng SMOTE CHỈ TRÊN TẬP TRAIN
    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    return X_train_smote, X_test, y_train_smote, y_test

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Huấn luyện và trả về kết quả đánh giá các mô hình."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42) 
    }
    
    results = []
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Tính toán Metrics
        f1 = f1_score(y_test, y_pred)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        
        results.append({
            "Mô hình": name,
            "F1-Score": round(f1, 4),
            "PR-AUC": round(pr_auc, 4)
        })
        
    return models, pd.DataFrame(results)
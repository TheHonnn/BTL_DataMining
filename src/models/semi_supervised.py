import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import f1_score, average_precision_score
from sklearn.utils.class_weight import compute_class_weight


# =========================================================
# 1. SIMULATE UNLABELED DATA
# =========================================================

def simulate_unlabeled_data(y_train, labeled_ratio=0.2, random_state=42):
    """
    Giả lập thiếu nhãn bằng cách che một phần nhãn (semi-supervised setting)
    """
    np.random.seed(random_state)

    y_train_semi = np.copy(y_train)

    mask_size = int(len(y_train) * (1 - labeled_ratio))
    mask_indices = np.random.choice(len(y_train), size=mask_size, replace=False)

    y_train_semi[mask_indices] = -1

    print("===== SIMULATE UNLABELED DATA =====")
    print(f"Total train samples: {len(y_train_semi)}")
    print(f"Labeled samples: {np.sum(y_train_semi != -1)}")
    print(f"Unlabeled samples: {np.sum(y_train_semi == -1)}")

    return y_train_semi


# =========================================================
# 2. HYPERPARAMETER TUNING
# =========================================================

def tune_base_model(X_train, y_train_labeled):

    print("\n===== HYPERPARAMETER TUNING =====")

    base_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    )

    param_grid = {
        "C": np.logspace(-3, 3, 20),
        "solver": ["lbfgs"]
    }

    search = RandomizedSearchCV(
        base_model,
        param_grid,
        n_iter=10,
        scoring="f1",
        cv=3,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train_labeled)

    print("Best parameters:", search.best_params_)

    return search.best_estimator_


# =========================================================
# 3. TRAIN MODELS
# =========================================================

def train_semi_supervised(X_train, X_test, y_train_true, y_train_semi, y_test):

    # ---- tách dữ liệu có nhãn ----
    labeled_indices = np.where(y_train_semi != -1)[0]

    X_train_labeled = X_train.iloc[labeled_indices]
    y_train_labeled = y_train_semi[labeled_indices]

    # =====================================================
    # 1. SUPERVISED MODEL
    # =====================================================

    print("\n===== TRAIN SUPERVISED MODEL =====")

    best_model = tune_base_model(X_train_labeled, y_train_labeled)

    model_sup = best_model
    model_sup.fit(X_train_labeled, y_train_labeled)

    y_pred_sup = model_sup.predict(X_test)
    y_prob_sup = model_sup.predict_proba(X_test)[:, 1]

    f1_sup = f1_score(y_test, y_pred_sup)
    pr_auc_sup = average_precision_score(y_test, y_prob_sup)

    # =====================================================
    # 2. SEMI SUPERVISED MODEL
    # =====================================================

    print("\n===== TRAIN SEMI-SUPERVISED MODEL =====")

    model_semi = SelfTrainingClassifier(
        best_model,
        threshold=0.85,
        verbose=True
    )

    model_semi.fit(X_train, y_train_semi)

    y_pred_semi = model_semi.predict(X_test)
    y_prob_semi = model_semi.predict_proba(X_test)[:, 1]

    f1_semi = f1_score(y_test, y_pred_semi)
    pr_auc_semi = average_precision_score(y_test, y_prob_semi)

    # =====================================================
    # RESULTS
    # =====================================================

    results = pd.DataFrame({
        "Method": [
            "Supervised-only (limited labels)",
            "Semi-supervised (Self-training)"
        ],
        "F1 Score": [
            round(f1_sup, 4),
            round(f1_semi, 4)
        ],
        "PR-AUC": [
            round(pr_auc_sup, 4),
            round(pr_auc_semi, 4)
        ]
    })

    return results, model_semi


# =========================================================
# 4. FULL PIPELINE
# =========================================================

def run_experiment(df, labeled_ratio=0.2):

    print("===== START EXPERIMENT =====")

    if "duration" in df.columns:
        df = df.drop(columns=["duration"])

    X = df.drop(columns=["y"])
    y = df["y"].values

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    y_train_semi = simulate_unlabeled_data(
        y_train,
        labeled_ratio=labeled_ratio
    )

    results, model = train_semi_supervised(
        X_train,
        X_test,
        y_train,
        y_train_semi,
        y_test
    )

    return results, model
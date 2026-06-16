"""
=============================================================
  Fraud Detection Pipeline — DecodeLabs Data Science Project 2
  Supervised Learning with SMOTE + Logistic Regression + Random Forest
=============================================================

Usage:
    python fraud_detection_pipeline.py

Requirements:
    pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn

Dataset:
    Download creditcard.csv from:
    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    Place it in the same directory as this script.
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

# CRITICAL: Use imblearn's Pipeline, NOT sklearn's
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH    = "creditcard.csv"
OUTPUT_DIR   = "fraud_outputs"
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def banner(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def save_fig(name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  💾 Saved: {path}")
    plt.close()


# ─────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────
def load_data():
    banner("STEP 1: Loading Dataset")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"\n❌ '{DATA_PATH}' not found.\n"
            "   Download it from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "   Then place creditcard.csv in the same folder as this script."
        )

    df = pd.read_csv(DATA_PATH)
    print(f"  Shape        : {df.shape}")
    print(f"  Missing vals : {df.isnull().sum().sum()}")

    counts = df['Class'].value_counts()
    pcts   = df['Class'].value_counts(normalize=True) * 100
    print(f"\n  Legitimate (0): {counts[0]:,}  ({pcts[0]:.2f}%)")
    print(f"  Fraudulent (1): {counts[1]:,}  ({pcts[1]:.2f}%)")
    print(f"  Imbalance ratio: {counts[0] // counts[1]}:1")

    # Plot class distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(['Legitimate', 'Fraudulent'], counts.values,
                color=['#2196F3', '#F44336'], edgecolor='black', linewidth=0.5)
    axes[0].set_title('Class Distribution (Count)', fontweight='bold')
    axes[0].set_ylabel('Transactions')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 500, f'{v:,}', ha='center', fontsize=10)

    axes[1].pie(counts.values, labels=['Legitimate', 'Fraudulent'],
                colors=['#2196F3', '#F44336'], autopct='%1.2f%%',
                startangle=90, explode=(0, 0.1))
    axes[1].set_title('Class Distribution (%)', fontweight='bold')

    plt.suptitle('The Reality of Financial Datasets', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig('01_class_distribution.png')

    return df


# ─────────────────────────────────────────────
# STEP 2: PREPARE FEATURES
# ─────────────────────────────────────────────
def prepare_features(df):
    banner("STEP 2: Feature Preparation")

    # Drop Time — not predictive; keep Amount (pipeline scaler handles it for LR)
    X = df.drop(columns=['Class', 'Time'])
    y = df['Class']

    print(f"  Features : {X.shape[1]} columns  ({list(X.columns[:5])} ... )")
    print(f"  Target   : {y.shape[0]:,} samples")
    return X, y


# ─────────────────────────────────────────────
# STEP 3: STRATIFIED TRAIN/TEST SPLIT
# ─────────────────────────────────────────────
def split_data(X, y):
    banner("STEP 3: Stratified Train/Test Split")
    print("  ⚠️  Splitting BEFORE SMOTE — the Golden Rule of leak-free pipelines.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"\n  Train : {X_train.shape[0]:,} samples | Fraud: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
    print(f"  Test  : {X_test.shape[0]:,} samples | Fraud: {y_test.sum():,} ({y_test.mean()*100:.2f}%)")
    print("  ✅ Test set is untouched. SMOTE stays inside the training pipeline.")

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# STEP 4: BUILD & TUNE PIPELINES
# ─────────────────────────────────────────────
def build_lr_pipeline(X_train, y_train):
    banner("STEP 4a: Logistic Regression Pipeline + GridSearchCV")
    print("  Pipeline: StandardScaler → SMOTE → LogisticRegression")

    pipeline = Pipeline(steps=[
        ('scaler',     StandardScaler()),
        ('smote',      SMOTE(random_state=RANDOM_STATE)),
        ('classifier', LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        ))
    ])

    param_grid = {
        'smote__k_neighbors'  : [3, 5, 7],
        'classifier__C'       : [0.01, 0.1, 1.0],
        'classifier__solver'  : ['lbfgs', 'liblinear']
    }

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )

    print(f"\n  🔍 Searching {len(param_grid['smote__k_neighbors']) * len(param_grid['classifier__C']) * len(param_grid['classifier__solver'])} combinations × {CV_FOLDS} folds...")
    grid.fit(X_train, y_train)

    print(f"\n  ✅ Best Params  : {grid.best_params_}")
    print(f"  ✅ Best CV AUC  : {grid.best_score_:.4f}")
    return grid


def build_rf_pipeline(X_train, y_train):
    banner("STEP 4b: Random Forest Pipeline + GridSearchCV")
    print("  Pipeline: SMOTE → RandomForestClassifier  (no scaler — trees are scale-invariant)")

    pipeline = Pipeline(steps=[
        ('smote',      SMOTE(random_state=RANDOM_STATE)),
        ('classifier', RandomForestClassifier(
            n_jobs=-1,
            random_state=RANDOM_STATE,
            class_weight='balanced_subsample'
        ))
    ])

    param_grid = {
        'smote__k_neighbors'         : [3, 5, 7],
        'classifier__n_estimators'   : [100, 200],
        'classifier__max_depth'      : [10, 20, None],
        'classifier__min_samples_leaf': [1, 2]
    }

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )

    n_combos = (len(param_grid['smote__k_neighbors']) *
                len(param_grid['classifier__n_estimators']) *
                len(param_grid['classifier__max_depth']) *
                len(param_grid['classifier__min_samples_leaf']))
    print(f"\n  🔍 Searching {n_combos} combinations × {CV_FOLDS} folds...")
    print("  ⏳ This may take a few minutes...")
    grid.fit(X_train, y_train)

    print(f"\n  ✅ Best Params  : {grid.best_params_}")
    print(f"  ✅ Best CV AUC  : {grid.best_score_:.4f}")
    return grid


# ─────────────────────────────────────────────
# STEP 5: EVALUATE ON TEST SET
# ─────────────────────────────────────────────
def evaluate_model(name, grid, X_test, y_test):
    banner(f"STEP 5: Evaluating {name} on Untouched Test Set")
    print("  ⚠️  Accuracy is intentionally excluded — it's misleading on imbalanced data.\n")

    best   = grid.best_estimator_
    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_prob)

    print(f"  Precision : {precision:.4f}   (when we flag fraud, are we right?)")
    print(f"  Recall    : {recall:.4f}   (of all fraud, how much did we catch?)")
    print(f"  F1 Score  : {f1:.4f}   (harmonic mean)")
    print(f"  ROC-AUC   : {roc_auc:.4f}   (target: > 0.85)")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Legitimate','Fraudulent'])}")

    return y_pred, y_prob, {
        'name': name, 'precision': precision,
        'recall': recall, 'f1': f1, 'roc_auc': roc_auc
    }


# ─────────────────────────────────────────────
# STEP 6: VISUALIZE RESULTS
# ─────────────────────────────────────────────
def plot_results(lr_data, rf_data, y_test):
    banner("STEP 6: Generating Evaluation Charts")

    lr_pred, lr_prob, lr_scores = lr_data
    rf_pred, rf_prob, rf_scores = rf_data

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Model Evaluation Dashboard', fontsize=16, fontweight='bold')

    # Confusion Matrix — LR
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, lr_pred),
        display_labels=['Legitimate', 'Fraudulent']
    ).plot(ax=axes[0, 0], colorbar=False, cmap='Blues')
    axes[0, 0].set_title('Logistic Regression\nConfusion Matrix', fontweight='bold')

    # Confusion Matrix — RF
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, rf_pred),
        display_labels=['Legitimate', 'Fraudulent']
    ).plot(ax=axes[0, 1], colorbar=False, cmap='Oranges')
    axes[0, 1].set_title('Random Forest\nConfusion Matrix', fontweight='bold')

    # ROC Curve — LR
    fpr, tpr, _ = roc_curve(y_test, lr_prob)
    axes[1, 0].plot(fpr, tpr, color='#1565C0', lw=2, label=f'LR  AUC={lr_scores["roc_auc"]:.4f}')
    axes[1, 0].plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    axes[1, 0].axhline(y=0.85, color='green', linestyle=':', alpha=0.5, label='Target 0.85')
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_title('Logistic Regression\nROC Curve', fontweight='bold')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    # ROC Curve — RF
    fpr, tpr, _ = roc_curve(y_test, rf_prob)
    axes[1, 1].plot(fpr, tpr, color='#E65100', lw=2, label=f'RF  AUC={rf_scores["roc_auc"]:.4f}')
    axes[1, 1].plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    axes[1, 1].axhline(y=0.85, color='green', linestyle=':', alpha=0.5, label='Target 0.85')
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].set_title('Random Forest\nROC Curve', fontweight='bold')
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    save_fig('02_evaluation_dashboard.png')

    # Model comparison bar chart
    comparison = pd.DataFrame([lr_scores, rf_scores]).set_index('name')
    comparison.columns = ['Precision', 'Recall', 'F1', 'ROC-AUC']

    fig, ax = plt.subplots(figsize=(9, 5))
    comparison.T.plot(kind='bar', ax=ax, color=['#1565C0', '#E65100'],
                      edgecolor='black', linewidth=0.5, width=0.6)
    ax.set_title('LR vs Random Forest — Key Metrics\n(Accuracy excluded by design)', fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_xticklabels(['Precision', 'Recall', 'F1', 'ROC-AUC'], rotation=0)
    ax.axhline(y=0.85, color='green', linestyle='--', alpha=0.6, label='AUC Target')
    ax.set_ylim(0, 1.08)
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_fig('03_model_comparison.png')

    return comparison


def plot_feature_importance(rf_grid, feature_names):
    rf_clf = rf_grid.best_estimator_.named_steps['classifier']
    importances = pd.Series(rf_clf.feature_importances_, index=feature_names)
    top = importances.nlargest(15).sort_values()

    fig, ax = plt.subplots(figsize=(9, 6))
    top.plot(kind='barh', ax=ax, color='#E65100', edgecolor='black', linewidth=0.5)
    ax.set_title('Top 15 Feature Importances (Random Forest)', fontweight='bold')
    ax.set_xlabel('Importance Score')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    save_fig('04_feature_importance.png')


# ─────────────────────────────────────────────
# STEP 7: FINAL SUMMARY
# ─────────────────────────────────────────────
def print_summary(comparison):
    banner("FINAL SUMMARY")
    print(comparison.to_string(float_format=lambda x: f"{x:.4f}"))
    winner = comparison['ROC-AUC'].idxmax()
    print(f"\n  🏆 Best model by ROC-AUC: {winner}")
    print(f"\n  📁 All charts saved to: ./{OUTPUT_DIR}/")
    print("""
  Ze"""
=============================================================
  Fraud Detection Pipeline — DecodeLabs Data Science Project 2
  Supervised Learning with SMOTE + Logistic Regression + Random Forest
=============================================================

Usage:
    python fraud_detection_pipeline.py

Requirements:
    pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn

Dataset:
    Download creditcard.csv from:
    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    Place it in the same directory as this script.
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

# CRITICAL: Use imblearn's Pipeline, NOT sklearn's
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH    = "creditcard.csv"
OUTPUT_DIR   = "fraud_outputs"
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def banner(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def save_fig(name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  💾 Saved: {path}")
    plt.close()


# ─────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────
def load_data():
    banner("STEP 1: Loading Dataset")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"\n❌ '{DATA_PATH}' not found.\n"
            "   Download it from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "   Then place creditcard.csv in the same folder as this script."
        )

    df = pd.read_csv(DATA_PATH)
    print(f"  Shape        : {df.shape}")
    print(f"  Missing vals : {df.isnull().sum().sum()}")

    counts = df['Class'].value_counts()
    pcts   = df['Class'].value_counts(normalize=True) * 100
    print(f"\n  Legitimate (0): {counts[0]:,}  ({pcts[0]:.2f}%)")
    print(f"  Fraudulent (1): {counts[1]:,}  ({pcts[1]:.2f}%)")
    print(f"  Imbalance ratio: {counts[0] // counts[1]}:1")

    # Plot class distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(['Legitimate', 'Fraudulent'], counts.values,
                color=['#2196F3', '#F44336'], edgecolor='black', linewidth=0.5)
    axes[0].set_title('Class Distribution (Count)', fontweight='bold')
    axes[0].set_ylabel('Transactions')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 500, f'{v:,}', ha='center', fontsize=10)

    axes[1].pie(counts.values, labels=['Legitimate', 'Fraudulent'],
                colors=['#2196F3', '#F44336'], autopct='%1.2f%%',
                startangle=90, explode=(0, 0.1))
    axes[1].set_title('Class Distribution (%)', fontweight='bold')

    plt.suptitle('The Reality of Financial Datasets', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig('01_class_distribution.png')

    return df


# ─────────────────────────────────────────────
# STEP 2: PREPARE FEATURES
# ─────────────────────────────────────────────
def prepare_features(df):
    banner("STEP 2: Feature Preparation")

    # Drop Time — not predictive; keep Amount (pipeline scaler handles it for LR)
    X = df.drop(columns=['Class', 'Time'])
    y = df['Class']

    print(f"  Features : {X.shape[1]} columns  ({list(X.columns[:5])} ... )")
    print(f"  Target   : {y.shape[0]:,} samples")
    return X, y


# ─────────────────────────────────────────────
# STEP 3: STRATIFIED TRAIN/TEST SPLIT
# ─────────────────────────────────────────────
def split_data(X, y):
    banner("STEP 3: Stratified Train/Test Split")
    print("  ⚠️  Splitting BEFORE SMOTE — the Golden Rule of leak-free pipelines.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"\n  Train : {X_train.shape[0]:,} samples | Fraud: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
    print(f"  Test  : {X_test.shape[0]:,} samples | Fraud: {y_test.sum():,} ({y_test.mean()*100:.2f}%)")
    print("  ✅ Test set is untouched. SMOTE stays inside the training pipeline.")

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# STEP 4: BUILD & TUNE PIPELINES
# ─────────────────────────────────────────────
def build_lr_pipeline(X_train, y_train):
    banner("STEP 4a: Logistic Regression Pipeline + GridSearchCV")
    print("  Pipeline: StandardScaler → SMOTE → LogisticRegression")

    pipeline = Pipeline(steps=[
        ('scaler',     StandardScaler()),
        ('smote',      SMOTE(random_state=RANDOM_STATE)),
        ('classifier', LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        ))
    ])

    param_grid = {
        'smote__k_neighbors'  : [3, 5, 7],
        'classifier__C'       : [0.01, 0.1, 1.0],
        'classifier__solver'  : ['lbfgs', 'liblinear']
    }

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )

    print(f"\n  🔍 Searching {len(param_grid['smote__k_neighbors']) * len(param_grid['classifier__C']) * len(param_grid['classifier__solver'])} combinations × {CV_FOLDS} folds...")
    grid.fit(X_train, y_train)

    print(f"\n  ✅ Best Params  : {grid.best_params_}")
    print(f"  ✅ Best CV AUC  : {grid.best_score_:.4f}")
    return grid


def build_rf_pipeline(X_train, y_train):
    banner("STEP 4b: Random Forest Pipeline + GridSearchCV")
    print("  Pipeline: SMOTE → RandomForestClassifier  (no scaler — trees are scale-invariant)")

    pipeline = Pipeline(steps=[
        ('smote',      SMOTE(random_state=RANDOM_STATE)),
        ('classifier', RandomForestClassifier(
            n_jobs=-1,
            random_state=RANDOM_STATE,
            class_weight='balanced_subsample'
        ))
    ])

    param_grid = {
        'smote__k_neighbors'         : [3, 5, 7],
        'classifier__n_estimators'   : [100, 200],
        'classifier__max_depth'      : [10, 20, None],
        'classifier__min_samples_leaf': [1, 2]
    }

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )

    n_combos = (len(param_grid['smote__k_neighbors']) *
                len(param_grid['classifier__n_estimators']) *
                len(param_grid['classifier__max_depth']) *
                len(param_grid['classifier__min_samples_leaf']))
    print(f"\n  🔍 Searching {n_combos} combinations × {CV_FOLDS} folds...")
    print("  ⏳ This may take a few minutes...")
    grid.fit(X_train, y_train)

    print(f"\n  ✅ Best Params  : {grid.best_params_}")
    print(f"  ✅ Best CV AUC  : {grid.best_score_:.4f}")
    return grid


# ─────────────────────────────────────────────
# STEP 5: EVALUATE ON TEST SET
# ─────────────────────────────────────────────
def evaluate_model(name, grid, X_test, y_test):
    banner(f"STEP 5: Evaluating {name} on Untouched Test Set")
    print("  ⚠️  Accuracy is intentionally excluded — it's misleading on imbalanced data.\n")

    best   = grid.best_estimator_
    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_prob)

    print(f"  Precision : {precision:.4f}   (when we flag fraud, are we right?)")
    print(f"  Recall    : {recall:.4f}   (of all fraud, how much did we catch?)")
    print(f"  F1 Score  : {f1:.4f}   (harmonic mean)")
    print(f"  ROC-AUC   : {roc_auc:.4f}   (target: > 0.85)")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Legitimate','Fraudulent'])}")

    return y_pred, y_prob, {
        'name': name, 'precision': precision,
        'recall': recall, 'f1': f1, 'roc_auc': roc_auc
    }


# ─────────────────────────────────────────────
# STEP 6: VISUALIZE RESULTS
# ─────────────────────────────────────────────
def plot_results(lr_data, rf_data, y_test):
    banner("STEP 6: Generating Evaluation Charts")

    lr_pred, lr_prob, lr_scores = lr_data
    rf_pred, rf_prob, rf_scores = rf_data

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Model Evaluation Dashboard', fontsize=16, fontweight='bold')

    # Confusion Matrix — LR
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, lr_pred),
        display_labels=['Legitimate', 'Fraudulent']
    ).plot(ax=axes[0, 0], colorbar=False, cmap='Blues')
    axes[0, 0].set_title('Logistic Regression\nConfusion Matrix', fontweight='bold')

    # Confusion Matrix — RF
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, rf_pred),
        display_labels=['Legitimate', 'Fraudulent']
    ).plot(ax=axes[0, 1], colorbar=False, cmap='Oranges')
    axes[0, 1].set_title('Random Forest\nConfusion Matrix', fontweight='bold')

    # ROC Curve — LR
    fpr, tpr, _ = roc_curve(y_test, lr_prob)
    axes[1, 0].plot(fpr, tpr, color='#1565C0', lw=2, label=f'LR  AUC={lr_scores["roc_auc"]:.4f}')
    axes[1, 0].plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    axes[1, 0].axhline(y=0.85, color='green', linestyle=':', alpha=0.5, label='Target 0.85')
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_title('Logistic Regression\nROC Curve', fontweight='bold')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    # ROC Curve — RF
    fpr, tpr, _ = roc_curve(y_test, rf_prob)
    axes[1, 1].plot(fpr, tpr, color='#E65100', lw=2, label=f'RF  AUC={rf_scores["roc_auc"]:.4f}')
    axes[1, 1].plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    axes[1, 1].axhline(y=0.85, color='green', linestyle=':', alpha=0.5, label='Target 0.85')
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].set_title('Random Forest\nROC Curve', fontweight='bold')
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    save_fig('02_evaluation_dashboard.png')

    # Model comparison bar chart
    comparison = pd.DataFrame([lr_scores, rf_scores]).set_index('name')
    comparison.columns = ['Precision', 'Recall', 'F1', 'ROC-AUC']

    fig, ax = plt.subplots(figsize=(9, 5))
    comparison.T.plot(kind='bar', ax=ax, color=['#1565C0', '#E65100'],
                      edgecolor='black', linewidth=0.5, width=0.6)
    ax.set_title('LR vs Random Forest — Key Metrics\n(Accuracy excluded by design)', fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_xticklabels(['Precision', 'Recall', 'F1', 'ROC-AUC'], rotation=0)
    ax.axhline(y=0.85, color='green', linestyle='--', alpha=0.6, label='AUC Target')
    ax.set_ylim(0, 1.08)
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_fig('03_model_comparison.png')

    return comparison


def plot_feature_importance(rf_grid, feature_names):
    rf_clf = rf_grid.best_estimator_.named_steps['classifier']
    importances = pd.Series(rf_clf.feature_importances_, index=feature_names)
    top = importances.nlargest(15).sort_values()

    fig, ax = plt.subplots(figsize=(9, 6))
    top.plot(kind='barh', ax=ax, color='#E65100', edgecolor='black', linewidth=0.5)
    ax.set_title('Top 15 Feature Importances (Random Forest)', fontweight='bold')
    ax.set_xlabel('Importance Score')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    save_fig('04_feature_importance.png')


# ─────────────────────────────────────────────
# STEP 7: FINAL SUMMARY
# ─────────────────────────────────────────────
def print_summary(comparison):
    banner("FINAL SUMMARY")
    print(comparison.to_string(float_format=lambda x: f"{x:.4f}"))
    winner = comparison['ROC-AUC'].idxmax()
    print(f"\n  🏆 Best model by ROC-AUC: {winner}")
    print(f"\n  📁 All charts saved to: ./{OUTPUT_DIR}/")
    print("""
  Zero-Leakage Protocol — Checklist:
  ✅ Accuracy discarded — used Precision, Recall, F1, ROC-AUC
  ✅ SMOTE interpolates, never duplicates
  ✅ Train/Test split happened BEFORE SMOTE & scaling
  ✅ Used imblearn.pipeline.Pipeline (not sklearn's)
  ✅ GridSearchCV tunes SMOTE + model params together
  ✅ Test set evaluated in raw imbalanced form
  ✅ StratifiedKFold preserves class ratio every fold
""")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("\n🚀 DecodeLabs — Fraud Detection Pipeline")
    print("   Project 2: Supervised Learning | SMOTE + LR + RF")

    df                             = load_data()
    X, y                           = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    lr_grid = build_lr_pipeline(X_train, y_train)
    rf_grid = build_rf_pipeline(X_train, y_train)

    lr_pred, lr_prob, lr_scores = evaluate_model("Logistic Regression", lr_grid, X_test, y_test)
    rf_pred, rf_prob, rf_scores = evaluate_model("Random Forest",       rf_grid, X_test, y_test)

    comparison = plot_results(
        (lr_pred, lr_prob, lr_scores),
        (rf_pred, rf_prob, rf_scores),
        y_test
    )
    plot_feature_importance(rf_grid, X.columns.tolist())
    print_summary(comparison)


if __name__ == "__main__":
    main()ro-Leakage Protocol — Checklist:
  ✅ Accuracy discarded — used Precision, Recall, F1, ROC-AUC
  ✅ SMOTE interpolates, never duplicates
  ✅ Train/Test split happened BEFORE SMOTE & scaling
  ✅ Used imblearn.pipeline.Pipeline (not sklearn's)
  ✅ GridSearchCV tunes SMOTE + model params together
  ✅ Test set evaluated in raw imbalanced form
  ✅ StratifiedKFold preserves class ratio every fold
""")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("\n🚀 DecodeLabs — Fraud Detection Pipeline")
    print("   Project 2: Supervised Learning | SMOTE + LR + RF")

    df                             = load_data()
    X, y                           = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    lr_grid = build_lr_pipeline(X_train, y_train)
    rf_grid = build_rf_pipeline(X_train, y_train)

    lr_pred, lr_prob, lr_scores = evaluate_model("Logistic Regression", lr_grid, X_test, y_test)
    rf_pred, rf_prob, rf_scores = evaluate_model("Random Forest",       rf_grid, X_test, y_test)

    comparison = plot_results(
        (lr_pred, lr_prob, lr_scores),
        (rf_pred, rf_prob, rf_scores),
        y_test
    )
    plot_feature_importance(rf_grid, X.columns.tolist())
    print_summary(comparison)


if __name__ == "__main__":
    main()

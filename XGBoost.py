import xgboost
import inspect
print("XGBoost version:", xgboost.__version__)
print("XGBoost file location:", xgboost.__file__)
print("XGBClassifier defined in:", inspect.getfile(xgboost.XGBClassifier))


import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score,
    recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import ttest_rel
import shap
sys.modules.pop('xgboost', None)  # clear any stale imports
from xgboost import XGBClassifier
print("Imported clean XGBClassifier from:", XGBClassifier.__module__)


# 0. Load Raman data # replace with your filename
data = pd.read_csv('myfilename.csv')
X = data.iloc[:, 2:].astype(float).values #extract features
y = data.iloc[:, 1].values  #extract class labels
feature_names = data.columns[2:]

# 1. Data preparation
X = np.asarray(X)
y = np.asarray(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Train/validation/test split with stratification
X_train, X_test_val, y_train, y_test_val = train_test_split(
    X, y, test_size=0.35, stratify=y, random_state=42
)
X_test, X_val, y_test, y_val = train_test_split(
    X_test_val, y_test_val, test_size=0.5, stratify=y_test_val, random_state=42
)

# 3. Encode labels for XGBoost multiclass 
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)
y_test_enc = le.transform(y_test)
num_class = len(np.unique(y_train_enc))

# 4. Base XGBoost model setup
base_model = XGBClassifier(
    objective='multi:softprob',
    num_class=num_class,
    eval_metric='mlogloss',
    nthread=-1,
    random_state=42,
)

# 5. Hyperparameter grid for optimization
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [6],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1],
    'reg_alpha': [0],
    'reg_lambda': [1.0],
    'min_child_weight': [1],
}

# 6. Stratified 5-fold CV for grid search
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 7. Grid search with neg_log_loss scoring
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring='neg_log_loss',
    cv=cv,
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train, y_train_enc)

best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("Best multiclass XGBoost parameters:")
for k, v in best_params.items():
    print(f"  {k}: {v}")
print(f"Best cross-validated log-loss on train: {best_score:.6f}")

# 8. Retrain best model on combined train+validation data
X_train_val_combined = np.vstack((X_train, X_val))
y_train_val_combined = np.concatenate((y_train_enc, y_val_enc))

best_estimator = XGBClassifier(
    objective='multi:softprob',
    num_class=num_class,
    eval_metric='mlogloss',
    nthread=-1,
    random_state=42,
    **best_params
)
best_estimator.fit(X_train_val_combined, y_train_val_combined)

# 9. Predictions on test set
val_pred_proba = best_estimator.predict_proba(X_test)
val_pred_enc = np.argmax(val_pred_proba, axis=1)
val_pred = le.inverse_transform(val_pred_enc)

# 10. Predictions on validation set
val_pred_proba_valset = best_estimator.predict_proba(X_val)
val_pred_enc_valset = np.argmax(val_pred_proba_valset, axis=1)
val_pred_valset = le.inverse_transform(val_pred_enc_valset)

# 11. Calculate and display metrics for test set
val_accuracy = accuracy_score(y_test, val_pred)
val_balanced_accuracy_test = balanced_accuracy_score(y_test, val_pred)
val_weighted_recall_test = recall_score(y_test, val_pred, average='weighted')
val_weighted_f1_test = f1_score(y_test, val_pred, average='weighted')
class_report_test = classification_report(y_test, val_pred)
conf_matrix = confusion_matrix(y_test, val_pred)

# 12. Calculate metrics for validation set
val_accuracy_valset = accuracy_score(y_val, val_pred_valset)
val_balanced_accuracy_valset = balanced_accuracy_score(y_val, val_pred_valset)
val_weighted_recall_valset = recall_score(y_val, val_pred_valset, average='weighted')
val_weighted_f1_valset = f1_score(y_val, val_pred_valset, average='weighted')
class_report_valset = classification_report(y_val, val_pred_valset)

print("Test Accuracy (best model):", val_accuracy)
print("Validation Accuracy (best model):", val_accuracy_valset)
print("Classification Report (test set):\n", class_report_test)
print("Classification Report (validation set):\n", class_report_valset)
print("Confusion Matrix (test set):\n", conf_matrix)

# 13. Save confusion matrix plot
plt.figure(figsize=(14, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='BuPu',
            xticklabels=le.classes_, yticklabels=le.classes_,
            cbar=True, square=True)
plt.title("Confusion Matrix (Test Set)")
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.xticks(rotation=40, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout()
plt.savefig("XGB_Confusion_Matrix_Test.png", dpi=300)
plt.close()

# 14. Stability/Significance Check via Paired t-Test
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_seed_42, scores_seed_43 = [], []

for train_idx, val_idx in skf.split(X_train_val_combined, y_train_val_combined):
    X_train_fold, X_val_fold = X_train_val_combined[train_idx], X_train_val_combined[val_idx]
    y_train_fold, y_val_fold = y_train_val_combined[train_idx], y_train_val_combined[val_idx]

    model1 = XGBClassifier(
        objective='multi:softprob',
        num_class=num_class,
        eval_metric='mlogloss',
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,
        reg_alpha=0,
        reg_lambda=1.0,
        min_child_weight=1,
        nthread=-1,
        random_state=42
    )
    model2 = XGBClassifier(
          objective='multi:softprob',
    num_class=num_class,
    eval_metric='mlogloss',
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    reg_alpha=0,
    reg_lambda=1.0,
    min_child_weight=1,
    nthread=-1,
    random_state=43
    )

    cv_scores1 = cross_val_score(model1, X_train_fold, y_train_fold, cv=3, scoring='balanced_accuracy', n_jobs=-1)
    cv_scores2 = cross_val_score(model2, X_train_fold, y_train_fold, cv=3, scoring='balanced_accuracy', n_jobs=-1)

    scores_seed_42.append(np.mean(cv_scores1))
    scores_seed_43.append(np.mean(cv_scores2))

t_stat, p_val = ttest_rel(scores_seed_42, scores_seed_43)
print(f"Paired t-test (Balanced Accuracy): T = {t_stat:.3f}, P = {p_val:.4f}")
if p_val < 0.05:
    print("→ Significant difference detected (non-random improvement).")
else:
    print("→ No significant difference; model stable.")

# 15. Calculate and save Accuracy, Balanced Accuracy, Weighted Recall, and Weighted F1-score for test and validation sets:
test_accuracy = accuracy_score(y_test, val_pred)
test_balanced_accuracy = balanced_accuracy_score(y_test, val_pred)
test_weighted_recall = recall_score(y_test, val_pred, average='weighted')
test_weighted_f1 = f1_score(y_test, val_pred, average='weighted')

val_accuracy = accuracy_score(y_val, val_pred_valset)
val_balanced_accuracy = balanced_accuracy_score(y_val, val_pred_valset)
val_weighted_recall = recall_score(y_val, val_pred_valset, average='weighted')
val_weighted_f1 = f1_score(y_val, val_pred_valset, average='weighted')

print(f"Test Set Metrics:\n Accuracy={test_accuracy:.4f}, Balanced Accuracy={test_balanced_accuracy:.4f}, Weighted Recall={test_weighted_recall:.4f}, Weighted F1-score={test_weighted_f1:.4f}")
print(f"Validation Set Metrics:\n Accuracy={val_accuracy:.4f}, Balanced Accuracy={val_balanced_accuracy:.4f}, Weighted Recall={val_weighted_recall:.4f}, Weighted F1-score={val_weighted_f1:.4f}")

# 16. Log all results to a text file
log_file = "XGB_Model_Results.txt"
with open(log_file, "w", encoding="utf-8") as f:
    f.write(f"Test Accuracy (best model): {val_accuracy:.4f}\n")
    f.write(f"Validation Accuracy (best model): {val_accuracy_valset:.4f}\n\n")
    f.write(f"Classification Report (Test Set):\n{class_report_test}\n")
    f.write(f"Classification Report (Validation Set):\n{class_report_valset}\n")
    f.write(f"Confusion Matrix (Test Set):\n{conf_matrix}\n\n")
    f.write(f"Paired t-test (Balanced Accuracy): T = {t_stat:.3f}, P = {p_val:.4f}\n")
    if p_val < 0.05:
        f.write("→ Significant difference detected (non-random improvement).\n")
    else:
        f.write("→ No significant difference; model stable.\n")
    f.write("\nBest Parameters:\n")
    for k, v in best_params.items():
        f.write(f"  {k}: {v}\n")
    f.write("\n--- Explicit Metrics for Test and Validation Sets ---\n")
    f.write(f"Test Set:\nAccuracy: {test_accuracy:.4f}\nBalanced Accuracy: {test_balanced_accuracy:.4f}\nWeighted Recall: {test_weighted_recall:.4f}\nWeighted F1-score: {test_weighted_f1:.4f}\n\n")
    f.write(f"Validation Set:\nAccuracy: {val_accuracy:.4f}\nBalanced Accuracy: {val_balanced_accuracy:.4f}\nWeighted Recall: {val_weighted_recall:.4f}\nWeighted F1-score: {val_weighted_f1:.4f}\n\n")

print(f"Results and figures are saved as PNG files and {log_file}")

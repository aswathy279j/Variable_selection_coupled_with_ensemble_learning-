import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    balanced_accuracy_score, recall_score, f1_score, ConfusionMatrixDisplay
)
from scipy.stats import ttest_rel
import shap

def rf_oob_earlystop(csv_file,
                          max_trees=300, step=20, oob_tol=0.0005, patience=3,
                          random_state=42):
    """
    Random Forest classification pipeline:
    - Stratified train/validation/test split (60/20/20)
    - OOB-based early stopping for tree selection
    - class_weight='balanced' for imbalance
    - Clear class labels from data for all plots/reports
    - SHAP per class and global feature importance plots (saved headless)
    - fig aesthetics improved
    """

    # Load data and get class labels as they appear in data
    data = pd.read_excel(csv_file) if csv_file.endswith('.xlsx') else pd.read_csv(csv_file)
    X = data.iloc[:, 2:].values
    y = data.iloc[:, 1].values
    feature_names = data.columns[2:]
    base_filename = os.path.splitext(os.path.basename(csv_file))[0]

    # Extract class label names (as string) in their original order in data
    unique_classes = pd.Series(y).astype(str).unique().tolist()

    # Splitting
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.35, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state
    )

    print(f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    log_file = f"RF_Results_{base_filename}.txt"
    with open(log_file, 'w', encoding='utf-8') as f:

        def log(msg):
            print(msg)
            f.write(msg + "\n")

        # OOB + early stopping optimization
        results = []
        best_oob = 0.0
        best_trees = step
        no_improve_count = 0

        log("Optimizing Random Forest using OOB accuracy:")
        for n_trees in range(step, max_trees + step, step):
            model = RandomForestClassifier(
                n_estimators=n_trees,
                max_features="sqrt",
                oob_score=True,
                class_weight='balanced',
                n_jobs=-1,
                random_state=random_state,
                bootstrap=True
            )
            model.fit(X_train, y_train)
            oob_score = model.oob_score_
            results.append((n_trees, oob_score))
            log(f"{n_trees:4d} Trees | OOB Accuracy = {oob_score:.4f}")

            if oob_score - best_oob > oob_tol:
                best_oob = oob_score
                best_trees = n_trees
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    log(f"→ Early stopping triggered at {n_trees} trees.")
                    break

        results_df = pd.DataFrame(results, columns=["n_trees", "oob_accuracy"])
        plt.figure(figsize=(10, 6))
        plt.plot(results_df["n_trees"], results_df["oob_accuracy"], marker='o')
        plt.axvline(best_trees, color='red', linestyle='--', label=f"Optimal = {best_trees}")
        plt.title("Random Forest OOB Accuracy (Early Stopping)")
        plt.xlabel("Number of Trees")
        plt.ylabel("OOB Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"RF_OOB_EarlyStop_{base_filename}.png", dpi=300)
        plt.close()
        log(f"Optimal trees: {best_trees} | Best OOB Accuracy = {best_oob:.4f}")

        # Train best model
        best_model = RandomForestClassifier(
            n_estimators=best_trees,
            max_features="sqrt",
            oob_score=True,
            class_weight='balanced',
            n_jobs=-1,
            random_state=random_state,
            bootstrap=True
        )
        best_model.fit(X_train, y_train)

        # Evaluate validation and test sets
        for split, (X_eval, y_eval) in zip(['Validation', 'Test'], [(X_val, y_val), (X_test, y_test)]):
            y_pred = best_model.predict(X_eval)
            acc = accuracy_score(y_eval, y_pred)
            bal_acc = balanced_accuracy_score(y_eval, y_pred)
            recall = recall_score(y_eval, y_pred, average='weighted')
            f1 = f1_score(y_eval, y_pred, average='weighted')
            log(f"\n{split} Set Metrics:")
            log(f"Accuracy: {acc:.4f}")
            log(f"Balanced Accuracy: {bal_acc:.4f}")
            log(f"Weighted Recall: {recall:.4f}")
            log(f"Weighted F1-score: {f1:.4f}")
            log("\n" + classification_report(y_eval, y_pred, target_names=unique_classes))

            cm = confusion_matrix(y_eval, y_pred, labels=unique_classes)
            plt.figure(figsize=(14, 10))  # larger for long labels
            sns.heatmap(cm, annot=True, fmt='d', cmap='BuPu',
                        xticklabels=unique_classes, yticklabels=unique_classes,
                        cbar=True, square=True)
            plt.title(f"Confusion Matrix ({split} Set)")
            plt.xlabel('Predicted Label', fontsize=14)
            plt.ylabel('True Label', fontsize=14)
            plt.xticks(rotation=40, ha='right', fontsize=12)
            plt.yticks(rotation=0, fontsize=12)
            plt.tight_layout()
            plt.savefig(f"RF_ConfMatrix_{split}_{base_filename}.png", dpi=300)
            plt.close()

        # Paired t-test for stability
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        scores_a, scores_b = [], []
        for train_idx, val_idx in skf.split(X_train, y_train):
            Xa, Xb = X_train[train_idx], X_train[val_idx]
            ya, yb = y_train[train_idx], y_train[val_idx]
            m_a = RandomForestClassifier(n_estimators=best_trees, oob_score=True,
                                         class_weight='balanced', random_state=random_state)
            m_b = RandomForestClassifier(n_estimators=best_trees, oob_score=True,
                                         class_weight='balanced', random_state=random_state+1)
            scores_a.append(np.mean(cross_val_score(m_a, Xa, ya, cv=3, scoring='balanced_accuracy')))
            scores_b.append(np.mean(cross_val_score(m_b, Xb, yb, cv=3, scoring='balanced_accuracy')))
        t_stat, p_val = ttest_rel(scores_a, scores_b)
        log(f"\nPaired t-test (Balanced Accuracy): T = {t_stat:.3f}, P = {p_val:.4f}")
        if p_val < 0.05:
            log("Significant difference detected (non-random improvement).")
        else:
            log(" No significant difference; model stable.")

    print(f"Results and figures are saved to: {log_file}")

    return best_model, best_trees, results_df

# Usage: replace with your filename
model, best_trees, oob_results = rf_oob_earlystop('myfilename.xlsx')


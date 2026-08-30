import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, average_precision_score, confusion_matrix
)
from lightgbm import LGBMClassifier

def load_data():
    processed_dir = "data/processed"
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.squeeze()")).squeeze() if os.path.exists(os.path.join(processed_dir, "y_test.squeeze()")) else pd.read_csv(os.path.join(processed_dir, "y_test.csv")).squeeze()
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test, name="Model"):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1-Score": float(f1),
        "ROC-AUC": float(roc_auc),
        "PR-AUC": float(pr_auc),
        "Confusion_Matrix": cm.tolist()
    }
    
    print(f"\n--- {name} Evaluation ---")
    print(f"Accuracy:        {acc:.4f}")
    print(f"Precision:       {prec:.4f}")
    print(f"Recall:          {rec:.4f}")
    print(f"F1-Score:        {f1:.4f}")
    print(f"ROC-AUC:         {roc_auc:.4f}")
    print(f"PR-AUC (AP):     {pr_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    return metrics

def run_training():
    print("Loading preprocessed datasets...")
    X_train, X_test, y_train, y_test = load_data()
    
    # 1. Baseline Logistic Regression Model
    print("\nTraining Baseline Logistic Regression model...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    lr_model.fit(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, name="Logistic Regression (Baseline)")
    
    # 2. Optimized LightGBM Classifier Model
    print("\nTraining Optimized LightGBM Classifier model...")
    lgb_model = LGBMClassifier(
        learning_rate=0.05,
        max_depth=8,
        num_leaves=64,
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    lgb_metrics = evaluate_model(lgb_model, X_test, y_test, name="LightGBM (Optimized)")
    
    # 3. Save Model Artifacts
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(lr_model, os.path.join(models_dir, "logistic_regression.joblib"))
    joblib.dump(lgb_model, os.path.join(models_dir, "lightgbm.joblib"))
    print(f"\nSaved models to {models_dir}/")
    
    # 4. Save Evaluation Results
    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    
    results = {
        "Logistic Regression": lr_metrics,
        "LightGBM": lgb_metrics
    }
    
    with open(os.path.join(outputs_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    summary_path = os.path.join(outputs_dir, "evaluation_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Hotel Booking Cancellation Prediction - Model Evaluation Summary\n")
        f.write("=================================================================\n\n")
        for model_name, metrics in results.items():
            f.write(f"{model_name}:\n")
            f.write("-" * len(model_name) + "\n")
            f.write(f"  Accuracy:        {metrics['Accuracy']:.4f}\n")
            f.write(f"  Precision:       {metrics['Precision']:.4f}\n")
            f.write(f"  Recall:          {metrics['Recall']:.4f}\n")
            f.write(f"  F1-Score:        {metrics['F1-Score']:.4f}\n")
            f.write(f"  ROC-AUC:         {metrics['ROC-AUC']:.4f}\n")
            f.write(f"  PR-AUC:          {metrics['PR-AUC']:.4f}\n")
            f.write(f"  Confusion Matrix:\n")
            f.write(f"    [[TN: {metrics['Confusion_Matrix'][0][0]}, FP: {metrics['Confusion_Matrix'][0][1]}],\n")
            f.write(f"     [FN: {metrics['Confusion_Matrix'][1][0]}, TP: {metrics['Confusion_Matrix'][1][1]}]]\n\n")
            
    print(f"Saved evaluation metrics to {outputs_dir}/evaluation_metrics.json")
    print(f"Saved evaluation summary text to {outputs_dir}/evaluation_summary.txt")

if __name__ == "__main__":
    run_training()

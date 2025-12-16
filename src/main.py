"""
Algorithmic Fairness Audit & Mitigation Pipeline
Context: Heart Failure Prediction
Author: Kim Ståhlberg
"""
import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --- Configuration ---
DATA_PATH = "data/heart.csv" # Expected location
FEMALE_THRESHOLD = -0.2  # Optimized threshold from Phase 2
N_SPLITS = 4
N_REPEATS = 25

def load_and_prep_data():
    """Loads data and creates intersectional subgroups."""
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        print("Please download the Heart Failure Prediction dataset (heart.csv) and place it in a 'data' folder.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    # create AgeGroup (0.65 split for sample size stability)
    df['AgeGroup'] = pd.qcut(df['Age'], q=[0, 0.65, 1.0], labels=['Young', 'Old'])
    df['Subgroup'] = df['Sex'].astype(str) + " " + df['AgeGroup'].astype(str)
    # Stratification column
    df['strata'] = df['Subgroup'] + "_" + df['HeartDisease'].astype(str)
    
    # Preprocessing
    X_raw = df.drop(columns=['HeartDisease', 'AgeGroup', 'Subgroup', 'strata'])
    X = pd.get_dummies(X_raw, drop_first=True)
    y = df['HeartDisease']
    strata = df['strata']
    return df, X, y, strata

def calculate_metrics(y_true, y_pred, group_name):
    """Calculates Safety (FNR) and Burden (FPR) metrics."""
    positives = y_true == 1
    negatives = y_true == 0
    
    # Safety: Miss Rate (False Negative Rate)
    fnr = (y_pred[positives] == 0).mean() if positives.sum() > 0 else np.nan
    # Burden: False Alarm Rate (False Positive Rate)
    fpr = (y_pred[negatives] == 1).mean() if negatives.sum() > 0 else np.nan
    acc = (y_true == y_pred).mean()
    
    # Return metrics AND counts for context
    return {
        "Group": group_name, 
        "Accuracy": acc, 
        "FNR": fnr, 
        "FPR": fpr, 
        "Count_Sick": positives.sum()
    }

def run_audit(df, X, y, strata):
    print("Running Hierarchical Audit (Baseline vs. Mitigated)...")
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)
    results = []

    for i, (train_idx, test_idx) in enumerate(rskf.split(X, strata)):
        # Split
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # --- MODEL 1: BASELINE (Single Linear SVM) ---
        clf_base = Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="linear"))])
        clf_base.fit(X_train, y_train)
        pred_base = clf_base.predict(X_test)
        
        # --- MODEL 2: MITIGATED (Decoupled + RBF + Threshold) ---
        train_male = X_train['Sex_M'] == 1
        test_male = X_test['Sex_M'] == 1
        
        # Train Male (Linear)
        clf_m = Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="linear"))])
        clf_m.fit(X_train[train_male], y_train[train_male])
        
        # Train Female (RBF + Balanced)
        clf_f = Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="rbf", class_weight='balanced'))])
        clf_f.fit(X_train[~train_male], y_train[~train_male])
        
        # Predict
        pred_fair = pd.Series(index=X_test.index, dtype=int)
        
        # Men: Standard Prediction
        if test_male.sum() > 0:
            pred_fair[test_male] = clf_m.predict(X_test[test_male])
            
        # Women: Threshold Adjustment (-0.2)
        if (~test_male).sum() > 0:
            scores = clf_f.decision_function(X_test[~test_male])
            pred_fair[~test_male] = (scores > FEMALE_THRESHOLD).astype(int)
        
        # --- Evaluate Subgroups ---
        subgroups = df.iloc[test_idx]['Subgroup']
        
        # Collect metrics for specific groups of interest
        for group in ['F Young', 'M Old']:
            mask = subgroups == group
            if mask.sum() > 0:
                # Baseline Results
                results.append({**calculate_metrics(y_test[mask], pred_base[mask], group), "Model": "Baseline"})
                # Fair Results
                results.append({**calculate_metrics(y_test[mask], pred_fair[mask], group), "Model": "Mitigated"})

    return pd.DataFrame(results)

if __name__ == "__main__":
    df, X, y, strata = load_and_prep_data()
    results = run_audit(df, X, y, strata)
    
    # Summary with Mean and Standard Deviation
    # We aggregate Metrics with mean/std, and Counts with mean
    summary = results.groupby(['Model', 'Group']).agg({
        'FNR': ['mean', 'std'],
        'FPR': ['mean', 'std'],
        'Accuracy': ['mean', 'std'],
        'Count_Sick': ['mean']
    })
    
    print("\n=== FINAL AUDIT RESULTS (Mean ± SD) ===")
    # Rounding for clean output
    print(summary.round(3))
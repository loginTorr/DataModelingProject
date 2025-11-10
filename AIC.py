from CSV_Reader import loadData
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import difflib
import math

def _norm(s: str) -> str:
    return ''.join(ch for ch in s.lower() if ch.isalnum())

def _resolve_columns(df, requested):
    norm_map = {_norm(c): c for c in df.columns}
    resolved = []
    for name in requested:
        key = _norm(name)
        if key in norm_map:
            resolved.append(norm_map[key])
            continue
        match = difflib.get_close_matches(key, list(norm_map.keys()), n=1, cutoff=0.75)
        if match:
            resolved.append(norm_map[match[0]])
            continue
        raise KeyError(f"Column '{name}' not found. Available: {list(df.columns)}")
    return resolved

def _gaussian_log_pdf(x, mean, var):
    # x: (F,), mean: (F,), var: (F,)
    return -0.5 * (np.log(2 * math.pi * var) + ((x - mean) ** 2) / var).sum()

def compute_log_likelihood(model: GaussianNB, X: np.ndarray, y: np.ndarray) -> float:
    classes = model.classes_
    class_index = {c: i for i, c in enumerate(classes)}
    means = model.theta_          # shape (C, F)
    vars_ = model.var_            # shape (C, F)
    priors = model.class_prior_   # shape (C,)
    logL = 0.0
    for xi, yi in zip(X, y):
        ci = class_index[yi]
        log_prior = math.log(priors[ci])
        log_cond = _gaussian_log_pdf(xi, means[ci], vars_[ci])
        logL += log_prior + log_cond
    return logL

def compute_aic(model: GaussianNB, X: np.ndarray, y: np.ndarray):
    C = len(model.classes_)
    F = X.shape[1]
    # Parameters: means (C*F) + variances (C*F) + class priors (C-1)
    k = 2 * C * F + (C - 1)
    logL = compute_log_likelihood(model, X, y)
    aic = 2 * k - 2 * logL
    # Optional small-sample correction (AICc)
    n = X.shape[0]
    return {
        "log_likelihood": logL,
        "k": k,
        "AIC": aic,
    }

def run_aic_evaluation():
    rows = loadData()
    headers = rows[0]
    data = np.array(rows[1:], dtype=float)
    df = pd.DataFrame(data, columns=headers)

    features_requested = ["alcohol", "volatile acidity", "total sulfur dioxide", "density"]
    target_requested = ["quality"]

    resolved_features = _resolve_columns(df, features_requested)
    resolved_target = _resolve_columns(df, target_requested)[0]

    X = df[resolved_features].values
    y = df[resolved_target].values.astype(int)

    model = GaussianNB()
    model.fit(X, y)

    classes = [int(c) for c in model.classes_]  # ensure plain Python ints
    metrics = compute_aic(model, X, y)

    print("=== AIC Evaluation: Gaussian Naive Bayes ===")
    print(f"Features: {resolved_features}")
    print(f"Target: {resolved_target}")
    print(f"Classes: {classes}")
    print(f"n (samples): {X.shape[0]}, k (parameters): {metrics['k']}")
    print(f"Log-Likelihood: {metrics['log_likelihood']:.4f}")
    print(f"AIC: {metrics['AIC']:.4f}")

if __name__ == "__main__":
    run_aic_evaluation()

# kfold_validation.py
# K-Fold validation comparing Logistic Regression and Bayesian Network
# Uses your CSV_Reader.loadData()

import numpy as np
import pandas as pd
from CSV_Reader import loadData
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

def make_bin_edges(values, q=3):
    # quantile edges; fall back to equal-width if duplicates collapse bins
    probs = np.linspace(0, 1, q + 1)
    edges = np.unique(np.quantile(values, probs))
    if len(edges) - 1 < q:
        edges = np.linspace(values.min(), values.max(), q + 1)
    return edges

def apply_bins(frame, edges_map):
    out = frame.copy()
    for c, edges in edges_map.items():
        # labels 0..q-1 (or use +1 if you prefer 1..q)
        out[c] = pd.cut(out[c], bins=edges, labels=False, include_lowest=True).astype(int)
    return out

def train_bn(df_train, df_test, feature_names, target, q_states=3):
    # structure: all features -> target
    edges = [(f, target) for f in feature_names]
    model = BayesianNetwork(edges)

    # --- NEW: declare full state space explicitly ---
    state_names = {f: list(range(q_states)) for f in feature_names}
    # for quality, use the actual classes present in the whole dataset
    # (if you want, you can pass them in; here we infer from training fold)
    state_names[target] = sorted(df_train[target].unique().tolist())

    # keep unseen states; pgmpy will create zero rows if a state is absent
    model.fit(
        df_train[feature_names + [target]],
        estimator=MaximumLikelihoodEstimator,
        state_names=state_names
    )

    infer = VariableElimination(model)

    y_true = df_test[target].to_numpy().astype(int)
    y_pred = []
    for _, row in df_test.iterrows():
        evidence = {f: int(row[f]) for f in feature_names}
        q = infer.query([target], evidence=evidence, show_progress=False)
        probs = q.values
        states = q.state_names[target]
        best_idx = int(np.argmax(probs))
        y_pred.append(int(states[best_idx]))

    return accuracy_score(y_true, y_pred)



def train_logistic(X_train, X_test, y_train, y_test):
    """Train and evaluate logistic regression on one fold."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    return accuracy_score(y_test, y_pred)


def main():
    # 1) Load data
    rows = loadData()
    headers = rows[0]
    data = np.array(rows[1:], dtype=int)
    df = pd.DataFrame(data, columns=headers)

    target = "quality"
    features = [c for c in df.columns if c != target]

    X = df[features].to_numpy()
    y = df[target].to_numpy().astype(int)

    # 2) Set up K-Fold (Stratified preserves class distribution)
    k = 5
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    accs_lr, accs_bn = [], []

    # after df, features, target are defined
    # global bin edges for BN to ensure consistent states across folds
    q = 3  # 3 bins works well; change to 4/5 if you want finer states
    bn_edges = {c: make_bin_edges(df[c].to_numpy(), q=q) for c in features}

    # 3) Loop over folds
    fold = 1
    for train_idx, test_idx in kf.split(X, y):
        print(f"\n=== Fold {fold}/{k} ===")
        fold += 1

        # Split into train/test for this fold
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # inside for train_idx, test_idx in kf.split(X, y):

        df_train = pd.DataFrame(X_train, columns=features)
        df_train[target] = y_train
        df_test = pd.DataFrame(X_test, columns=features)
        df_test[target] = y_test

        # <-- NEW: discretize only for BN
        df_train_bn = apply_bins(df_train[features], bn_edges)
        df_train_bn[target] = df_train[target].values
        df_test_bn = apply_bins(df_test[features], bn_edges)
        df_test_bn[target] = df_test[target].values

        # Logistic Regression on continuous features (unchanged)
        acc_lr = train_logistic(X_train, X_test, y_train, y_test)
        accs_lr.append(acc_lr)          # <-- add this
        print(f"Logistic Regression Accuracy: {acc_lr*100:.2f}%")

        # Bayesian Network on discretized copies
        acc_bn = train_bn(df_train_bn, df_test_bn, features, target, q_states=q)
        accs_bn.append(acc_bn)          # <-- add this
        print(f"Bayesian Network Accuracy: {acc_bn*100:.2f}%")



    # 4) Final averaged results
    print("\n=== K-Fold Summary ===")
    print(f"Logistic Regression Mean Accuracy: {np.mean(accs_lr)*100:.2f}% ± {np.std(accs_lr)*100:.2f}%")
    print(f"Bayesian Network Mean Accuracy: {np.mean(accs_bn)*100:.2f}% ± {np.std(accs_bn)*100:.2f}%")

    if np.mean(accs_bn) > np.mean(accs_lr):
        print("\nWinner: Bayesian Network (better average predictive accuracy).")
    elif np.mean(accs_bn) < np.mean(accs_lr):
        print("\nWinner: Logistic Regression (better average predictive accuracy).")
    else:
        print("\nIt’s a tie!")

if __name__ == "__main__":
    main()

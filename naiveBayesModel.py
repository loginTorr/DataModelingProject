from CSV_Reader import loadData
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import difflib

def runNaiveBayes():
    #load data
    rows = loadData()
    headers = rows[0]
    data = np.array(rows[1:], dtype=float)
    df = pd.DataFrame(data, columns=headers)
    print("Columns found:", list(df.columns))

    # Use canonical names; resolver below will handle spaces/underscores/typos
    features = ["alcohol", "volatile acidity", "total sulfur dioxide", "density"]
    target = "quality"

    # Resolve requested names to actual dataframe columns (handles typos/spaces)
    def _norm(s: str) -> str:
        return ''.join(ch for ch in s.lower() if ch.isalnum())

    norm_map = {_norm(c): c for c in df.columns}

    def resolve(name: str) -> str:
        key = _norm(name)
        if key in norm_map:
            return norm_map[key]
        match = difflib.get_close_matches(key, list(norm_map.keys()), n=1, cutoff=0.75)
        if match:
            return norm_map[match[0]]
        raise KeyError(f"Column '{name}' not found. Available columns: {list(df.columns)}")

    resolved_features = [resolve(f) for f in features]
    target_col = resolve(target)

    X = df[resolved_features].values
    y = df[target_col].values.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\n=== Naïve Bayes Classification ===")
    print(f"Features used: {resolved_features} (target: {target_col})")
    print(f"Accuracy: {acc*100:.2f}%\n")
    print("Confusion Matrix:\n", cm)
    #print("\nClassification Report:\n", report)

    class_means = pd.DataFrame(model.theta_, columns=resolved_features, index=model.classes_)
    print("\n=== Mean Feature Values by Quality Class (Model Learned) ===")
    print(class_means.round(3))

    # === NEW: Show predicted probabilities for first 5 test samples ===
    probs = model.predict_proba(X_test)
    sample_df = pd.DataFrame(probs, columns=[f"P(Q={c})" for c in model.classes_])
    sample_df[resolved_features] = X_test
    sample_df["Predicted Quality"] = y_pred
    sample_df["Actual Quality"] = y_test
    print("\n=== Example Predictions (first 5) ===")
    print(sample_df.head(5).round(3))

if __name__ == "__main__":
    runNaiveBayes()

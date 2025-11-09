from CSV_Reader import loadData
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

def runLogisticRegression():
    #load data
    rows = loadData()
    headers = rows[0]
    data = np.array(rows[1:], dtype=float)
    df = pd.DataFrame(data, columns=headers)

    # features
    features = ["alcohol", "volatileacidity", "totalsulfurdioxide", "density", "fixedacidity"]
    target = "quality"

    X = df[features].values
    y = df[target].values.astype(int)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\n=== Logistic Regression Classification ===")
    print(f"Accuracy: {acc*100:.2f}%\n")
    print("Confusion Matrix:\n", cm)
    #print("\nClassification Report:\n", report)

if __name__ == "__main__":
    runLogisticRegression()

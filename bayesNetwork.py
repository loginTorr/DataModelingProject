import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from CSV_Reader import loadData

def main():
    #load the dataset
    rows = loadData()
    headers = rows[0]
    data = np.array(rows[1:], dtype=int)

    df = pd.DataFrame(data, columns=headers)

    #where we set attributes vs target variable
    features = ["alcohol", "volatileacidity", "totalsulfurdioxide", "density", "fixedacidity", "chlorides", "ph", "freesulfurdioxide", "residualsugar", "citricacid", "sulphates"]
    target = "quality"
    df = df[features + [target]]

    #the structure of the Bayes Network
    edges = [(f, target) for f in features]
    model = DiscreteBayesianNetwork(edges)

    model.fit(df, estimator=MaximumLikelihoodEstimator)
    infer = VariableElimination(model)
    y_true = df[target].to_numpy()
    y_pred = []

    for _, row in df.iterrows():
        evidence = {f: int(row[f]) for f in features}
        q = infer.query([target], evidence=evidence, show_progress=False)
        probs = q.values
        state_names = q.state_names[target]
        best_idx = int(np.argmax(probs))
        y_pred.append(int(state_names[best_idx]))

    y_pred = np.array(y_pred, dtype=int)
    acc = (y_pred == y_true).mean()
    print("Features:", features)
    print("Edges:", edges)
    print(f"\nAccuracy (same data): {acc*100:.2f}%")

    #confusion matrix
    labels = np.sort(df[target].unique())
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    lab_to_idx = {lab: i for i, lab in enumerate(labels)}
    for t, p in zip(y_true, y_pred):
        cm[lab_to_idx[t], lab_to_idx[p]] += 1

    print("\nConfusion Matrix (rows=true, cols.pred):")
    print("    " + "    ".join([f"{1:>3}" for l in labels]))
    for i, l in enumerate(labels):
        print(f"{1:>3} " + "  ".join([f"{v:>3}" for v in cm[i]]))

if __name__ == "__main__":
    main()
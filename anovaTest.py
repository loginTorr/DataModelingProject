from CSV_Reader import loadData
import numpy as np
import pandas as pd
from scipy import stats

def runAnova():
    # loads the dataset
    rows =loadData()
    headers = rows[0]
    data = np.array(rows[1:], dtype=float)
    quality = data[:, -1]
    print("\n=== One-Way ANOVA: Attribute vs. Quality ===\n")
    results = []

    for i, attr in enumerate(headers[:-1]): #excludes the quality column
        # separate by quality level
        groups = []
        for quality_level in np.unique(quality):
            groups.append(data[quality == quality_level, i])

        #performs the anova with all quality levels
        f_stat, p_val = stats.f_oneway(*groups)
        results.append((attr, f_stat, p_val))

    #sort by small p_val
    results.sort(key=lambda x: x[2])

    df = pd.DataFrame(results, columns=["Attribute", "F-Stat", "P-Val"])
    print(df.to_string(index=False))

    return df
if __name__ == "__main__":
    runAnova()
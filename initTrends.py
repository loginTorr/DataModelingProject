from CSV_Reader import loadData
import numpy as np
import matplotlib.pyplot as plt
import collections

# Input values must be typed exactly as they are seen in the CSV. I don't have any user handling rip

def getTuples(rows, feature1, feature2):
    correspondingValues = []
    print("First row: " + str(rows[0]))
    if (feature1 in rows[0]) and (feature2 in rows[0]):
        for j in range(len(rows[0])):

            if feature1 == rows[0][j]:
                feature1ColIndex = j

            if feature2 == rows[0][j]:
                feature2ColIndex = j

        for i in range(1, len(rows)):
            aTuple = (float(rows[i][feature1ColIndex]), float(rows[i][feature2ColIndex]))
            correspondingValues.append(aTuple)

        print("Feature1: " + feature1)
        print("Feature2: " + feature2)
        print("List of tuples to plot: " + str(correspondingValues))
    
    return correspondingValues

# got this func online. np.polyfit runs a linear regression and outputs equation of best fit. 
def equationBestFit(tuples):
    x = np.array([int(feature[0]) for feature in tuples])
    y = np.array([int(feature[1]) for feature in tuples])

    # compute slope (m) and intercept (b)
    m, b = np.polyfit(x, y, 1)
    return m, b

# matplot code - source code + github examples helped alot
def plotTrend(tuples, feature1, feature2, slope, intercept):
    x = np.array([p[0] for p in tuples])
    y = np.array([p[1] for p in tuples])

    r = np.corrcoef(x, y)[0, 1]
    r2 = r ** 2

    freq=collections.Counter(tuples)

    plt.scatter(x, y, label='Data')
    plt.plot(x, slope * x + intercept, label=f'Fit: y={slope:.3f}x+{intercept:.3f}')

    # displaying number of occurences in each data point
    for (x_val, y_val), count in freq.items():
        plt.text(x_val, y_val, str(count), fontsize=8, ha='center', va='bottom', color='black')
        # computing pearson correlation and r^2
        plt.text(0.05, 0.95,
                 f"Pearson r = {r:.3f}\nR^2 = {r2:.3f}",
                 transform=plt.gca().transAxes,
                 fontsize=10, va='top', ha='right',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
    
    plt.title(f"{feature2} vs {feature1}")
    plt.xlabel(feature1); plt.ylabel(feature2)
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


def main():
    running = True
    while (running):
        feature1 = input("Type feature 1: ")
        feature2 = input("Type feature 2: ")

        matrix = loadData()

        tupes = getTuples(matrix, feature1, feature2)
        slope, intercept = equationBestFit(tupes)
        print(f"Eq: y = ( {slope:.4f} ) + ( {intercept:.4f} )")
        plotTrend(tupes, feature1, feature2, slope, intercept)


if __name__ == "__main__":
    main()

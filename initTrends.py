from CSV_Reader import loadData
import numpy as np

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
            aTuple = (rows[i][feature1ColIndex], rows[i][feature2ColIndex])
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


def main():
    running = True
    while (running):
        feature1 = input("Type feature 1: ")
        feature2 = input("Type feature 2: ")

        matrix = loadData()

        tupes = getTuples(matrix, feature1, feature2)
        slope, intercept = equationBestFit(tupes)
        print(f"Eq: y = ( {slope:.4f} ) + ( {intercept:.4f} )")


if __name__ == "__main__":
    main()

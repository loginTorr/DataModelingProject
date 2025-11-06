import csv
import numpy as np

## wineVals = np.zeros((4900, 12))
def loadData():
    with open('wine.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        rows = []
        for row in spamreader:
            rows.append([x for x in row])
    return rows

processedCSV = loadData()

wineVals = np.array(processedCSV)
sortedWineVals = [[r for r in wineVals if r[11] == '1'], [r for r in wineVals if r[11] == '2'], [r for r in wineVals if r[11] == '3'], [r for r in wineVals if r[11] == '4'], [r for r in wineVals if r[11] == '5']]
#print(sortedWineVals[3])

class QualityAsDependent:

    def __init__(self, sortedWineVals):
        self.sortedWineVals = sortedWineVals

    def getCertainTotals(self, qualityNum):  ## enter the quality num you want to use and the column of the attribute to get the totals of
        # print("GRABBING TOTALS FOR QUALITY:",(qualityNum+1))
        totals = np.zeros((12,5), dtype=int)

        for row in self.sortedWineVals[qualityNum]:
            # print(row)
            columnNum=0
            for column in row:
                # print(column)
                if column == '1':
                    totals[columnNum][0] += 1 
                if column == '2':
                    totals[columnNum][1] += 1 
                if column == '3':
                    totals[columnNum][2] += 1 
                if column == '4':
                    totals[columnNum][3] += 1 
                if column == '5':
                    totals[columnNum][4] += 1 
                columnNum+=1
        return totals

#obj = QualityAsDependent(sortedWineVals)
#qualityColumnTotals = [obj.getCertainTotals(i) for i in range(5)]

# print("\nTOTALS FOR QUALITY 1:")
# print (qualityColumnTotals[0])

# print("\nTOTALS FOR QUALITY 2:")
# print (qualityColumnTotals[1])

# print("\nTOTALS FOR QUALITY 3:")
# print (qualityColumnTotals[2])

# print("\nTOTALS FOR QUALITY 4:")
# print (qualityColumnTotals[3])

# print("\nTOTALS FOR QUALITY 5:")
# print (qualityColumnTotals[4])

def formatTotals(total, qualityNum):
    print("\nTotals for Quality:", qualityNum)
    print("\nFIXED ACIDITY:", "\nONES:", total[0][0], "\nTWOS:", total[0][1], "\nTHREES:", total[0][2], "\nFOURS:", total[0][3], "\nFIVES:", total[0][4])
    print("\nVOLATILE ACIDITY:", "\nONES:", total[1][0], "\nTWOS:", total[1][1], "\nTHREES:", total[1][2], "\nFOURS:", total[1][3], "\nFIVES:", total[1][4])
    print("\nCITRIC ACID:", "\nONES:", total[2][0], "\nTWOS:", total[2][1], "\nTHREES:", total[2][2], "\nFOURS:", total[2][3], "\nFIVES:", total[2][4])
    print("\nRESIDUAL SUGAR:", "\nONES:", total[3][0], "\nTWOS:", total[3][1], "\nTHREES:", total[3][2], "\nFOURS:", total[3][3], "\nFIVES:", total[3][4])
    print("\nCHLORIDES:", "\nONES:", total[4][0], "\nTWOS:", total[4][1], "\nTHREES:", total[4][2], "\nFOURS:", total[4][3], "\nFIVES:", total[4][4])
    print("\nFREE SULFUR DIOXIDE:", "\nONES:", total[5][0], "\nTWOS:", total[5][1], "\nTHREES:", total[5][2], "\nFOURS:", total[5][3], "\nFIVES:", total[5][4])
    print("\nTOTAL SULFUR DIOXIDE:", "\nONES:", total[6][0], "\nTWOS:", total[6][1], "\nTHREES:", total[6][2], "\nFOURS:", total[6][3], "\nFIVES:", total[6][4])
    print("\nDENSITY:", "\nONES:", total[7][0], "\nTWOS:", total[7][1], "\nTHREES:", total[7][2], "\nFOURS:", total[7][3], "\nFIVES:", total[7][4])
    print("\nPH:", "\nONES:", total[8][0], "\nTWOS:", total[8][1], "\nTHREES:", total[8][2], "\nFOURS:", total[8][3], "\nFIVES:", total[8][4])
    print("\nSULPHATES:", "\nONES:", total[9][0], "\nTWOS:", total[9][1], "\nTHREES:", total[9][2], "\nFOURS:", total[9][3], "\nFIVES:", total[9][4])
    print("\nALCOHOL:", "\nONES:", total[10][0], "\nTWOS:", total[10][1], "\nTHREES:", total[10][2], "\nFOURS:", total[10][3], "\nFIVES:", total[10][4])



if __name__ == "__main__":
    obj = QualityAsDependent(sortedWineVals)
    qualityColumnTotals = [obj.getCertainTotals(i) for i in range(5)]

    running = True
    while(running):
        print("\nPick an option:")
        print("1: See Quality 1 totals")
        print("2: See Quality 2 totals")
        print("3: See Quality 3 totals")
        print("4: See Quality 4 totals")
        print("5: See Quality 5 totals")
        print("6: quit")
        response = input("\nEnter a num:")

        if (response == '1'):
            formatTotals(qualityColumnTotals[0], 1)
        if (response == '2'):
            formatTotals(qualityColumnTotals[1], 2)
        if (response == '3'):
            formatTotals(qualityColumnTotals[2], 3)
        if (response == '4'):
            formatTotals(qualityColumnTotals[3], 4)
        if (response == '5'):
            formatTotals(qualityColumnTotals[4], 5)
        if (response == '6'):
            break

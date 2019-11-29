import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import StandardScaler

# python dataBuild.py train.csv test.csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train",
                        help="filename for features of the training data")
    parser.add_argument("test",
                        help="filename for labels associated with training data")

    args = parser.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    # features minus cover type and ID
    featNames = list(train.keys())[-2:0:-1]
    featNames.reverse()
    
    xTrain = pd.DataFrame(train.loc[:, featNames], columns=featNames)
    xTest = pd.DataFrame(test.loc[:, featNames], columns=featNames)

    # Scale with standard scaler
    scaler = StandardScaler()
    scaler.fit(xTrain.to_numpy())
    xTrain = pd.DataFrame(scaler.transform(xTrain.to_numpy()), columns=featNames)
    xTest = pd.DataFrame(scaler.transform(xTest.to_numpy()), columns=featNames)  

    xTrain.to_csv('xTrain.csv', index=False)
    xTest.to_csv('xTest.csv', index=False)
    train.loc[:, 'Cover_Type'].to_csv('yTrain.csv', index=False, header='Cover_Type')

    return


if __name__ == "__main__":
    main() 
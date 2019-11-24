import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

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

    featNames = list(train.keys())[-2::-1]

    train.loc[:, featNames].to_csv('xTrain.csv', index=False)
    test.loc[:, featNames].to_csv('xTest.csv', index=False)
    train.loc[:, 'Cover_Type'].to_csv('yTrain.csv', index=False, header='Cover_Type')

    return


if __name__ == "__main__":
    main() 
import argparse
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

# example invocation for command line
# python nb.py xTrain.csv yTrain.csv xTest.csv

class NB(object):
    model = None


    def __init__(self):
        self.model = GaussianNB()

    # trains model on X,Y
    def train(self, X, Y):
        self.model.fit(X,Y.ravel())

    # Predicts on features X
    def predict(self, X):
        return self.model.predict(X)

    # returns accuracy of predictions on X, Y
    def predAcc(self, X, Y):
        predictions = self.model.predict(X)
        ct = 0
        for i, j in zip(predictions, Y):
            ct += i == j
        return ct / len(Y)




    



def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("xTrain",
                        type=str,
                        help="filename for features of the training data")
    parser.add_argument("yTrain",
                        type=str,
                        help="filename for labels associated with training data")
    parser.add_argument("xTest",
                        type=str,
                        help="filename for features of the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv("../"+args.xTrain)
    yTrain = pd.read_csv("../"+args.yTrain)
    xTest = pd.read_csv("../"+args.xTest)

    # create an instance of the model
    nb = NB()
    print("NB")



if __name__ == "__main__":
    main()
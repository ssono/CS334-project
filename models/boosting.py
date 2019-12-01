import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold

# example invocation for command line
# python boosting.py 5 .2 5 xTrain.csv yTrain.csv xTest.csv

class Boost(object):
    maxDepth = 1
    lr = .1
    rounds = 1
    model = None

    def __init__(self, maxDepth, lr, rounds):
        self.maxDepth = maxDepth
        self.lr = lr
        self.rounds = rounds
        self.model = xgb.XGBClassifier(num_rounds = rounds, learning_rate = lr, max_depth = maxDepth)
        
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

    def gridSearch(self, X, Y, maxDepth, lr, rounds):
        nfolds = 3
        kf = KFold(nfolds)
        trIndices = []
        tsIndices = []
        for tr, ts in kf.split(X):
            trIndices.append(tr)
            tsIndices.append(ts)

        best = 0
        for m in maxDepth:
            for l in lr:
                for r in rounds:
                    print(m, l, r)
                    total = 0
                    testModel = Boost(m, l, r)
                    for i in range(nfolds):
                        testModel.train(X[trIndices[i], :], Y[trIndices[i], :])
                        acc = testModel.predAcc(X[tsIndices[i], :], Y[tsIndices[i], :])
                        total += acc / nfolds
                    if total > best:
                        self.model = testModel.model
                        self.maxDepth = m               
                        self.lr = l
                        self.rounds = r
                        best = total

        self.train(X,Y)

        return self

def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("maxDepth",
                        type=int,
                        help="max depth")
    parser.add_argument("lr",
                        type=float,
                        help="learning rate")
    parser.add_argument("rounds",
                        type=int,
                        help="times to run")
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
    boost = Boost(args.maxDepth, args.lr, args.rounds)

    # run gridsearch
    # maxDepth = [1, 3, 5, 11, 25, 51]
    # lr = [.1, .3, .5, .11, .25, .51]
    # rounds = [1, 3, 5, 11, 25, 51]
    maxDepth = [1, 3, 5]
    lr = [.1, .3, .5]
    rounds = [1, 3, 5]
    boost= boost.gridSearch(xTrain.to_numpy(), yTrain.to_numpy(), maxDepth, lr, rounds)
    print("Max Depth: " + str(boost.maxDepth))
    print("Learning rate: " + str(boost.lr))
    print("Num of rounds: " + str(boost.rounds))



if __name__ == "__main__":
    main()
import argparse
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

# example invocation for command line
# python knn.py 5 xTrain.csv yTrain.csv xTest.csv

class DT(object):
    maxDepth = 1
    minLeafSamples = 1
    criterion = 'gini'
    model = None


    def __init__(self, maxDepth, minLeafSamples, criterion):
        self.maxDepth = maxDepth
        self.minLeafSamples = minLeafSamples
        self.criterion = criterion
        self.model = DecisionTreeClassifier(min_samples_leaf = minLeafSamples, max_depth = maxDepth, criterion = criterion)

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

    def gridSearch(self, X, Y, maxDepth, minLeaf, criterion):
        nfolds = 3
        kf = KFold(nfolds)
        trIndices = []
        tsIndices = []
        for tr, ts in kf.split(X):
            trIndices.append(tr)
            tsIndices.append(ts)

        best = 0
        for d in maxDepth:
            for l in minLeaf:
                for c in criterion:
                    print(d,l,c)
                    total = 0
                    testModel = DT(d,l,c)
                    for i in range(nfolds):
                        testModel.train(X[trIndices[i], :], Y[trIndices[i], :])
                        acc = testModel.predAcc(X[tsIndices[i], :], Y[tsIndices[i], :])
                        total += acc / nfolds
                    if total > best:
                        self.model = testModel.model
                        self.maxDepth = d                
                        self.minLeafSamples = l
                        self.criterion = c
                        best = total

        self.train(X,Y)

        return self

def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("maxdepth",
                        type=int,
                        help="max depth of tree")
    parser.add_argument("minleafsamples",
                        type=int,
                        help="minimum samples per leaf")
    parser.add_argument("criterion",
                        type=str,
                        help="splitting criteria")
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
    dt = DT(args.maxdepth, args.minleafsamples, args.criterion)

    # run gridsearch
    mds = [1, 3, 5, 11, 25, 51]
    mls = [1, 3, 5, 11, 25, 51]
    criterion = ['gini', 'entropy']
    dt= dt.gridSearch(xTrain.to_numpy(), yTrain.to_numpy(), mds, mls, criterion)
    print("Max Depth: " + str(dt.maxDepth))
    print("Min Leaf Samples: " + str(dt.minLeafSamples))
    print("Splitting Criterion: " + str(dt.criterion))



if __name__ == "__main__":
    main()
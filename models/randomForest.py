import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

# example invocation for command line
# python randomForest.py 10 5 gini 5 5 gini xTrain.csv yTrain.csv xTest.csv

class RF(object):
    nest = 1
    maxFeatures = 1
    criterion = 'gini'
    maxDepth = 1
    minLeafSamples = 1
    model = None


    def __init__(self, nest, maxFeatures, criterion, maxDepth, minLeafSamples):
        self.maxDepth = maxDepth
        self.minLeafSamples = minLeafSamples
        self.criterion = criterion
        self.model = RandomForestClassifier(n_estimators=nest, max_features=maxFeatures, criterion=criterion, max_depth=maxDepth, min_samples_leaf = minLeafSamples)

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

    
    nest = 1
    maxFeatures = 1
    criterion = 'gini'
    maxDepth = 1
    minLeafSamples = 1

    def gridSearch(self, X, Y, nest, maxFeatures, criterion, maxDepth, minLeafSamples):
        nfolds = 3
        kf = KFold(nfolds)
        trIndices = []
        tsIndices = []
        for tr, ts in kf.split(X):
            trIndices.append(tr)
            tsIndices.append(ts)

        best = 0
        for n in nest:
            for m in maxFeatures:
                for c in criterion:
                    for d in maxDepth:
                        for l in minLeafSamples:
                            print(n, m, c, d, l)
                            total = 0
                            testModel = RF(n, m, c, d, l)
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
    parser.add_argument("nest",
                        type=int,
                        help="max number of trees")
    parser.add_argument("maxFeatures",
                        type=int,
                        help="max number of features")
    parser.add_argument("criterion",
                        type=str,
                        help="splitting criteria")
    parser.add_argument("maxDepth",
                        type=int,
                        help="max depth of tree")
    parser.add_argument("minLeafSamples",
                        type=int,
                        help="minimum number sof samples in a leaf")
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
    rf = RF(args.nest, args.maxFeatures, args.criterion, args.maxDepth, args.minLeafSamples)

    # run gridsearch
    nest = [1, 3, 5, 11, 25, 51]
    maxFeatures = [1, 3, 5, 9]
    criterion = ['gini', 'entropy']
    maxDepth = [1, 3, 5, 11, 25, 51]
    minLeafSamples = [1, 3, 5, 11, 25, 51]
    rf= rf.gridSearch(xTrain.to_numpy(), yTrain.to_numpy(), nest, maxFeatures, criterion, maxDepth, minLeafSamples)
    print("Num of trees: " + str(rf.nest))
    print("Max Features: " + str(rf.maxFeatures))
    print("Splitting Criterion: " + str(rf.criterion))
    print("Max Depth: " + str(rf.maxDepth))
    print("Min Leaf Samples: " + str(rf.minLeafSamples))



if __name__ == "__main__":
    main()
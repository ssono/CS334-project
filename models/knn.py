import argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

# example invocation for command line
# python knn.py 5 xTrain.csv yTrain.csv xTest.csv

class KNN(object):
    k = 1
    model = None


    def __init__(self, k):
        self.k = k
        self.model = KNeighborsClassifier(n_neighbors=k)

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

    # Conduct a gridsearch for k using X, Y and cross validation.
    # set self to best and return trained model
    def gridSearch(self, X, Y, ks):
        nfolds = 5
        kf = KFold(nfolds)
        trIndices = []
        tsIndices = []
        for tr, ts in kf.split(X):
            trIndices.append(tr)
            tsIndices.append(ts)

        best = 0
        for k in ks:
            total = 0
            testModel = KNN(k)
            for i in range(nfolds):
                testModel.train(X[trIndices[i], :], Y[trIndices[i], :])
                acc = testModel.predAcc(X[tsIndices[i], :], Y[tsIndices[i], :])
                total += acc / nfolds
            if total > best:
                self.model = testModel.model
                self.k = k
                best = total

        self.train(X,Y)

        return self




    



def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
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
    knn = KNN(args.k)

    # run gridsearch
    ks = [1, 3, 5, 11, 25, 51]
    knn = knn.gridSearch(xTrain.to_numpy(), yTrain.to_numpy(), ks)
    print("K: " + str(knn.k))


if __name__ == "__main__":
    main()
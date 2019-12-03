import argparse
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import KFold

# example invocation for command line
# python svm.py 1 poly 3 0.01 xTrain.csv yTrain.csv xTest.csv

class SVM(object):
    c = 0.1
    kernel = 'poly'
    degree = 3
    gamma = 0.01
    model = None


    def __init__(self, c, kernel, degree, gamma):
        self.c = c
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.model = SVC(C=c, kernel=kernel, degree=degree, gamma=gamma)

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
    def gridSearch(self, X, Y, cs, kernels, degrees, gammas):
        nfolds = 3
        kf = KFold(nfolds)
        trIndices = []
        tsIndices = []
        for tr, ts in kf.split(X):
            trIndices.append(tr)
            tsIndices.append(ts)

        best = 0
        for c in cs:
            for k in kernels:
                for d in degrees:
                    for g in gammas:
                        print(c,k,d,g)
                        total = 0
                        testModel = SVM(c,k,d,g)
                        for i in range(nfolds):
                            testModel.train(X[trIndices[i], :], Y[trIndices[i], :])
                            acc = testModel.predAcc(X[tsIndices[i], :], Y[tsIndices[i], :])
                            total += acc / nfolds
                        if total > best:
                            self.model = testModel.model
                            self.c = c
                            self.kernel = k
                            self.degree = d
                            self.gamma = g
                            best = total

        self.train(X,Y)

        return self




    



def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("C",
                        type=float,
                        help="penalty")
    parser.add_argument("kernel",
                        type=str,
                        help="kernel type")
    parser.add_argument("degree",
                        type=int,
                        help="polynomial degree")
    parser.add_argument("gamma",
                        type=float,
                        help="kernel parameter")
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

    cs = [ 0.01, 0.1, 1]
    kernels = ['poly', 'rbf']
    degrees = [1,2,3]
    gammas = [0.01, 0.1, 1]

    svm = SVM(args.C, args.kernel, args.degree, args.gamma)
    svm = svm.gridSearch(xTrain.to_numpy(), yTrain.to_numpy(), cs, kernels, degrees, gammas)

    print("SVM")
    print("cs: " + svm.c)
    print("cs: " + svm.kernel)
    print("cs: " + svm.degree)
    print("cs: " + svm.gamma)


if __name__ == "__main__":
    main()
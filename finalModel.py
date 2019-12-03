import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from models.boosting import Boost
from models.dt import DT
from models.knn import KNN
from models.nb import NB
from models.randomForest import RF
from models.svm import SVM

# python finalModel.py xTrain.csv yTrain.csv xTest.csv

def visualize(xTrain, yTrain, colNames):
    # DATA VISUALIZATION
    # start with generic dataframe with everything
    alldf = pd.DataFrame(xTrain, columns=colNames)
    alldf['Cover_Type'] = yTrain.to_numpy()

    # For correlation matrix drop one hot encoded
    cormatdf = alldf
    cormatdf = cormatdf.drop(columns=list(set(list(cormatdf.keys())) - set([
        'Elevation',
        'Aspect',
        'Slope',
        'Horizontal_Distance_To_Hydrology',
        'Vertical_Distance_To_Hydrology',
        'Horizontal_Distance_To_Roadways',
        'Hillshade_9am',
        'Hillshade_Noon',
        'Hillshade_3pm',
        'Horizontal_Distance_To_Fire_Points',
        'Cover_Type'

    ])))

    corrMat = cormatdf.corr()

    # plot as heatmap
    sns.heatmap(corrMat,
        xticklabels=cormatdf.columns,
        yticklabels=cormatdf.columns,
        annot=True,
        center=0,
        cmap="YlGnBu")

    plt.title('Continuous feature correlation')

    plt.show()
    return


def main():
    # Read file names
    parser = argparse.ArgumentParser()
    parser.add_argument("xTrain",
                        help="filename for features of the training data")
    parser.add_argument("yTrain",
                        help="filename for labels associated with training data")
    parser.add_argument("xTest",
                        help="filename for features of the test data")

    args = parser.parse_args()

    # load the train and test data assumes you'll use numpy
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    colNames = list(xTrain.keys())
    # visualize(xTrain, yTrain, colNames)

    models = {
        'boost': Boost(5, .2, 5),
        'dt': DT(25, 1, 'entropy'),
        'knn': KNN(1),
        'nb': NB(),
        'rf': RF(51, 25, 'gini', 25, 1),
        'svm': SVM(.1, 'poly', 3, .01)
    }

    X = xTrain.to_numpy()
    Y = yTrain.to_numpy()

    basePreds = []
    for k in models:
        models[k].train(X, Y)
        basePreds.append(list(models[k].predict(xTrain.to_numpy())))
    basePreds = np.array(basePreds)
    basePreds = np.transpose(basePreds)

    metalearner = Boost(5, .2, 5)

    nfolds = 3
    kf = KFold(nfolds)
    trIndices = []
    tsIndices = []
    for tr, ts in kf.split(X):
        trIndices.append(tr)
        tsIndices.append(ts)

    total = 0
    
    for i in range(nfolds):
        metalearner.train(X[trIndices[i], :], Y[trIndices[i], :])
        acc = metalearner.predAcc(X[tsIndices[i], :], Y[tsIndices[i], :])
        total += acc / nfolds

    print("ACC: ", total)

    metalearner.train(X,Y)
    finalPreds = metalearner.predict(xTest.to_numpy())
    finalPreds = np.array([list(range(len(xTest))), finalPreds]).transpose()
    finalPreds = pd.DataFrame(finalPreds, columns=['Id', 'Cover_Type'])
    finalPreds.to_csv('finalPredictions.csv', index=False)
    print(finalPreds)
    return


if __name__ == "__main__":
    main() 
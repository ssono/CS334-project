import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

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

    # use standard scaler
    scaler = StandardScaler()
    scaler.fit(xTrain)
    xTrain = scaler.transform(xTrain)
    xTest = scaler.transform(xTest.to_numpy())

    visualize(xTrain, yTrain, colNames)

    return


if __name__ == "__main__":
    main() 
import operator

import matplotlib.pyplot as plt
import numpy as np


def createDataSet():
    group = np.array([[1, 1.1], [1, 1], [0, 0], [0, 0.1]], dtype=float)
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    """

    :param inX:
    :param dataSet:
    :param labels:
    :param k:
    :return:
    """
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDisIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDisIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    print(classCount)
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())  # get the number of lines in the file
    returnMat = np.zeros((numberOfLines, 3))  # prepare matrix to return
    classLabelVector = []  # prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


if __name__ == '__main__':
    features, labels = file2matrix(
        r'C:\Users\Mechrev0\Desktop\MLiA_SourceCode\machinelearninginaction\Ch02\datingTestSet2.txt')
    print(features)
    print(labels[:20])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(features[:, 1], features[:, 2], s=15.0*np.array(labels), c=15.0*np.array(labels))
    ax.legend()
    plt.show()

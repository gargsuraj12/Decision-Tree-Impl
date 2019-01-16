import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from pprint import pprint
import statistics as st

def getAccuracy(tn, fp, fn, tp):
    num = tp+tn
    deno = tn+fp+fn+tp
    return (num/deno)*100

def getPrecision(tp, fp):
    return (tp/(tp+fp))*100

def getRecall(tp, fn):
    return (tp/(tp+fn))*100

class InternalNode:
    def __init__(self, attrName):
        self.attributeName = attrName
        self.childDict = {}

class LeafNode:
    def __init__(self, label):
        self.label = label     

# Splits the raw data into training data set and test data set
def splitTrainTest(df, testSize):
    if isinstance(testSize, float):
        testSize = round(testSize*len(df))

    indices = df.index.tolist()
    randIndices = random.sample(population=indices, k=testSize)
    testData = df.loc[randIndices]
    trainData = df.drop(randIndices)    
    return trainData, testData

# Checks whether the current class has pure data
def isDataPure(data):
    leftCol = data[:,-1]
    uniqueLabels = np.unique(leftCol)
    if len(uniqueLabels) == 1:
        return True
    return False 

# Returns the label of the pure data
# If current node is leaf and data is impure then returns the label which occurs most
def returnLabel(data):
    leftCol = data[:,-1]
    uniqueLabels, labelCounts = np.unique(leftCol, return_counts=True)
    # print(uniqueLabels, labelCounts)
    if len(uniqueLabels) == 1:
        return uniqueLabels[0]
    else:
        if labelCounts[0] > labelCounts[1]:
            return uniqueLabels[0]
    return uniqueLabels[1]        

# Calculates the overall entropy of the given data/label
def calcOverallEntropy(data):
    leftCol = data[:, -1]
    _, labelCounts = np.unique(leftCol, return_counts=True)
    sampleSpace = len(leftCol)
    probabilities = labelCounts / sampleSpace
    entropy = sum(probabilities * -(np.log2(probabilities)))
    return entropy

# Calculates the entropy of a perticular column/attribute
def calcColumnAvgEntropy(data, colNum):
    column = data[:,colNum]
    sampleSpace = len(column)
    featureValues, featureCounts = np.unique(column, return_counts=True)
    # print("Unique Feature values for col num: ", colNum, " are: ", featureValues)
    probabilities = featureCounts / sampleSpace
    featureEntropies = []
    # Calculating entropy of each feature of the column
    for val in featureValues:
        # print(val)
        columnData = data[column == val]
        entropy = calcOverallEntropy(columnData)
        featureEntropies.append(entropy)
    # print(featureEntropies)
    avgEntropy = sum(probabilities*featureEntropies)
    return avgEntropy

# Return the index of the attribute having the maximun information gain with the supplied data
def nextParentNodeAttribute(data):
    overallEntropy = calcOverallEntropy(data)
    infoGainList = []
    cols = data.shape[1]-1
    # Excluding 'left' column
    for i in range(cols):
        colAvgEntropy = calcColumnAvgEntropy(data, i)
        infoGainList.append(overallEntropy-colAvgEntropy)
    # print(infoGainList)
    maxEntropy = max(infoGainList)
    return infoGainList.index(maxEntropy)


# Building Decision Tree
def buildDecisionTree(data, headerList, depth, edgeLabel):
    
    # Base cases
    if isDataPure(data) or (len(headerList) == 1 and headerList[0] == 'left'):#
        label = returnLabel(data)
        return LeafNode(label)

    # Create N-ary Tree
    else:
        index = nextParentNodeAttribute(data)
        # print("Selected index is: ", index)
        attrName = headerList[index]
        # print("Selected Node at depth: ",depth," is: ", attrName)
        root = InternalNode(attrName)
        selectedColumn = data[:, index]
        uniqueFeatureValues = np.unique(selectedColumn)
        headerList = np.delete(headerList, index, axis = 0)
        # print("and features are: ", uniqueFeatureValues)
        for i in range(len(uniqueFeatureValues)):
            feature = uniqueFeatureValues[i]
            # discarding all the rows containing particular feature values
            newData = data[np.logical_not(data[:, index] == feature)]
            # dropping the selected attribute
            newData = np.delete(newData, index, axis=1)
            childNode = buildDecisionTree(newData, headerList, depth+1, feature)
            root.childDict.update({uniqueFeatureValues[i]:childNode})
        return root


def validateExample(root, header, example):
    # Base condition - We've reached to a leaf node
    if isinstance(root, LeafNode):
        return root.label
    
    # Recurse
    else:
        # print("Root's attribute name is: ", root.attributeName)
        # print("Length of the child nodes are: ", len(root.childDict))
        index = header.index(root.attributeName)
        header.pop(index)
        feature = example[index]
        example = np.delete(example, index, axis=0)
        nextRoot = root.childDict[feature]
        return validateExample(nextRoot, header, example)



# main segment starts here
if __name__ == '__main__':
    df = pd.read_csv("data.csv")
    trainData, testData = splitTrainTest(df, 0.2)
    # trainData, testData = df,df
    # data = trainData.values[:, 5:10]
    data = trainData.values
    # headerList = list(df)
    headerList = df.columns.values
    headerList = headerList[5:]
    # print(headerList)
    # print(data)
    #[trainData.time_spend_company > 2]
    # print(testData.values)
    # print(isDataPure(trainData.values))
    # print(returnLabel(data))
    # calcOverallEntropy(data)
    # entropy = calcColumnAvgEntropy(data, 2)
    # print(entropy)
    # print(nextParentNodeAttribute(data))
    
    root = buildDecisionTree(data, headerList, 1, None)

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(len(testData)):
        actual = testData.values[i, 5:10][-1]
        header = list(headerList)
        # print(testData.values[i, 5:10][-1])
        example  = testData.values[i, 5:9]
        # example  = testData.values[i, 1:9]
        predicted = validateExample(root, header, example)
        # print("Actual is: ", actual, " and predicted is: ", predicted)
        if actual == 0 and predicted == 0:
            tn += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        elif actual == 1 and predicted == 0:
            fn += 1
        else:
            tp += 1

    accuracy = getAccuracy(tn, fp, fn, tp)         
    # precision = getPrecision(tp, fp)
    # recall = getRecall(tp, fn)

    print("Accuracy is: ", accuracy)
    # print("Precision is: ", precision)
    # print("Recall is: ", recall)
    # print("F1 Measure is: ", st.harmonic_mean(precision, recall))
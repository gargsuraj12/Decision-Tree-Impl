import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from pprint import pprint
import statistics as st
import copy

CAT_ATTR_THRESHOLD = 5
LESS_THAN_EQUAL = 0
GREATER_THAN = 1
CAT_ATTR = 0
NUM_ATTR = 1
category_dict = {}

def getAccuracy(tn, fp, fn, tp):
    num = tp+tn
    deno = tn+fp+fn+tp
    return (num/deno)*100

def getPrecision(tp, fp):
    return (tp/(tp+fp))*100

def getRecall(tp, fn):
    return (tp/(tp+fn))*100


class InternalNode:
    def __init__(self, attrName, splitVal):
        self.attributeName = attrName
        self.childDict = {}
        self.isNodeNumerical = False
        self.splitVal = splitVal
        self.positives = 0
        self.negatives = 0


class LeafNode:
    def __init__(self, label, positives, negatives):
        self.label = label
        self.positives = positives
        self.negatives = negatives

# Replacing the values of the numerical attribute with 0s and 1s
def modifyNumAttr(data, colNum, splitVal):
    rows,_ = data.shape
    for i in range(rows):
        if data[i, colNum] <= splitVal:
            data[i, colNum] = 0
        else:
            data[i, colNum] = 1
    return data


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
def setLabel(data):
    leftCol = data[:,-1]
    uniqueLabels, labelCounts = np.unique(leftCol, return_counts=True)
    label = 0
    positives = 0
    negatives = 0
    if len(uniqueLabels) == 1:
        label =  uniqueLabels[0]
    else:
        if labelCounts[0] > labelCounts[1]:
            label = uniqueLabels[0]
        else:    
            label = uniqueLabels[1]
    for i in range(len(uniqueLabels)):
        if uniqueLabels[i] == 0:
            negatives += labelCounts[i]
        if uniqueLabels[i] == 1:
            positives += labelCounts[i]
    leaf = LeafNode(label, positives, negatives)
    return leaf  

# Splits the numerical attribute on the basis of the value
def splitNumAttr(data, colNum, val):
    dataBelow = list()
    dataAbove = list()
    rows,_ = data.shape
    for i in range(rows):
        if data[i, colNum] <= val:
            dataAbove.append(data[i])
        else:
            dataBelow.append(data[i])
    dataBelow = np.array(dataBelow)
    dataAbove = np.array(dataAbove)        
    return dataBelow, dataAbove            

def splitCatAttr(data, colNum, val):
    dataBelow = list()
    dataAbove = list()
    rows,_ = data.shape
    for i in range(rows):
        if data[i, colNum] == val:
            dataAbove.append(data[i])
        else:
            dataBelow.append(data[i])
    dataBelow = np.array(dataBelow)
    dataAbove = np.array(dataAbove)
    return dataBelow, dataAbove

# Calculates the overall entropy of the given data/label
def calcOverallEntropy(data):
    cols = data.shape
    # print(len(cols))
    if(len(cols) == 1):
        leftCol = data
    else:
        leftCol = data[:, -1]

    # leftCol = data[:, -1]
    _, labelCounts = np.unique(leftCol, return_counts=True)
    sampleSpace = len(leftCol)
    probabilities = labelCounts / sampleSpace
    entropy = sum(probabilities * -(np.log2(probabilities)))
    return entropy

# Calculates the entropy of a Categorical column/attribute
def calcCatAttrAvgEntropy(data, colNum):
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
    return None,avgEntropy


# Best splits(Using Bruteforce Method) the numerical attribute and calculates the average entropy
def calcNumAttrAvgEntropy(data, colNum):
    totalRows,_ = data.shape
    column = data[:, colNum]
    splitPoint = None
    avgEntropy = 1000000
    uniqueFeatureValues = np.unique(column)
    
    # Find the best split position for this column
    for val in uniqueFeatureValues:
        # val = (uniqueFeatureValues[i-1] + uniqueFeatureValues[i])/2
        dataBelow, dataAbove = splitNumAttr(data, colNum, val)
        belowEntropy = calcOverallEntropy(dataBelow)
        aboveEntropy = calcOverallEntropy(dataAbove)
        belowRows = len(dataBelow)
        aboveRows = len(dataAbove)
        # t_entropy = (aboveEntropy + belowEntropy)/2
        t_entropy = belowEntropy*(belowRows/totalRows) + aboveEntropy*(aboveRows/totalRows)
        if t_entropy < avgEntropy:
            avgEntropy = t_entropy
            splitPoint = val

    return splitPoint, avgEntropy


# Return the index of the attribute having the maximun information gain with the supplied data
def nextParentNodeAttribute(data, headerList):
    overallEntropy = calcOverallEntropy(data)
    infoGainList = []
    splitValList = []
    cols = data.shape[1]-1
    splitVal, colAvgEntropy = None, None
    # Excluding 'left' column
    for i in range(cols):
        if category_dict[headerList[i]] == CAT_ATTR:
            splitVal, colAvgEntropy = calcCatAttrAvgEntropy(data, i)
        else:    
            splitVal, colAvgEntropy = calcNumAttrAvgEntropy(data, i)
        
        splitValList.append(splitVal)
        infoGainList.append(overallEntropy-colAvgEntropy)

    # print(infoGainList)
    maxEntropy = max(infoGainList)
    index = infoGainList.index(maxEntropy)
    splitVal = splitValList[index]
    return splitVal, index


# Building Decision Tree
def buildDecisionTree(data, headerList, depth, edgeLabel):
    rows,_ = data.shape
    # Base cases
    if isDataPure(data) or (len(headerList) <= 1) or rows<=5: #  or rows<=5 and headerList[0] == 'left'
        leaf = setLabel(data)
        return leaf

    # Create N-ary Tree
    else:
        splitVal, index = nextParentNodeAttribute(data, headerList)
        attrName = headerList[index]
        root = InternalNode(attrName, splitVal)

        # Cheking if the selected attibute is numerical attribute
        if category_dict[headerList[index]] == NUM_ATTR:
            root.isNodeNumerical = True
            data = modifyNumAttr(data, index, splitVal)

        selectedColumn = data[:, index]
        uniqueFeatureValues = np.unique(selectedColumn)
        headerList = np.delete(headerList, index, axis = 0)
        
        for i in range(len(uniqueFeatureValues)):
            feature = uniqueFeatureValues[i]
            # discarding all the rows containing particular feature values
            newData = data[data[:, index] == feature]
            # dropping the selected attribute
            newData = np.delete(newData, index, axis=1)

            childNode = buildDecisionTree(newData, headerList, depth+1, feature)    # Recursion
            
            root.positives += childNode.positives
            root.negatives += childNode.negatives
            root.childDict.update({feature:childNode})

        return root


# Execution validation set
def validateExample(root, header, example):
    # Base condition - We've reached to a leaf node
    if isinstance(root, LeafNode):
        return root.label
    
    # Recurse
    else:
        index = header.index(root.attributeName)
        header.pop(index)
        feature = example[index]
        if root.isNodeNumerical == True:
            if feature <= root.splitVal:
                feature = 0
            else:
                feature = 1
        example = np.delete(example, index, axis=0)
        try:
            # key found
            nextRoot = root.childDict[feature]
            return validateExample(nextRoot, header, example)
        except KeyError:
            # key not found..
            if root.positives > root.negatives:
                return 1
            else:
                return 0
        

# Note : Handling of blank values in test example remianing
# main segment starts here
def main():
    df = pd.read_csv("data.csv")
    posData = df.loc[df['left'] == 0]
    negData = df.loc[df['left'] == 1]
    
    #spliting positive and negative dataset into randomly 80-20 % split
    posTrainData, posTestData = splitTrainTest(posData, 0.2)
    negTrainData, negTestData = splitTrainTest(negData, 0.2)
    
    #merging positive and negative data split so that training and validation dataset contains equal number of positive and negative value of feature label 
    trainData = pd.concat([posTrainData, negTrainData])
    testData = pd.concat([posTestData, negTestData])

    # trainData, testData = splitTrainTest(df, 0.2)
    # trainData, testData = df,df
    data = trainData.values
    headerList = df.columns.values
    _,cols = data.shape
    for i in range(cols):
        uniqueFeatureValues = np.unique(data[:, i])
        if len(uniqueFeatureValues) <= CAT_ATTR_THRESHOLD:
            category_dict[headerList[i]] = CAT_ATTR
        else:
            category_dict[headerList[i]] = NUM_ATTR
    
    # Call to build decision tree
    root = buildDecisionTree(data, headerList, 1, None)

    tp, tn, fp, fn = 0, 0, 0, 0

    # Running validation set
    for i in range(len(testData)):
        actual = testData.values[i, :][-1]
        header = list(headerList)
        example  = testData.values[i, 0:9]
        predicted = validateExample(root, header, example)
        if actual == 0 and predicted == 0:
            tn += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        elif actual == 1 and predicted == 0:
            fn += 1
        else:
            tp += 1

    print("True Negatives: ", tn)
    print("False Positves: ", fp)
    print("False Negatives: ", fn)
    print("True Positives: ", tp)
    
    accuracy = getAccuracy(tn, fp, fn, tp)         
    precision = getPrecision(tp, fp)
    recall = getRecall(tp, fn)

    print("Accuracy is: ", accuracy)
    print("Precision is: ", precision)
    print("Recall is: ", recall)
    l = [precision, recall]
    print("F1 Measure is: ", st.harmonic_mean(l))


if __name__ == '__main__':
    main()

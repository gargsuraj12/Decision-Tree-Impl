import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from pprint import pprint
import statistics as st
import sys

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
        self.positives = 0
        self.negatives = 0

class LeafNode:
    def __init__(self, label, positives, negatives):
        self.label = label
        self.positives = positives
        self.negatives = negatives

# Splits the raw data into training data set and test data set
def splitTrainTest(df, testSize):
    if isinstance(testSize, float):
        testSize = round(testSize*len(df))
    # random.seed(0)
    indices = df.index.tolist()
    randIndices = random.sample(population=indices, k=testSize)
    valData = df.loc[randIndices]
    trainData = df.drop(randIndices)    
    return trainData, valData

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
    # print(uniqueLabels, labelCounts)
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
    if isDataPure(data) or (len(headerList) == 1 ): # and headerList[0] == 'left'
        # label = returnLabel(data)
        # return LeafNode(label)
        leaf = setLabel(data)
        return leaf

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
            # newData = data[np.logical_not(data[:, index] == feature)]
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
        

def main(root, testData):

    tp, tn, fp, fn = 0, 0, 0, 0
    predictedList = []
    for i in range(len(testData)):
        actual = testData.values[i, :][-1]
        header = list(headerList)
        example  = testData.values[i, 0:9]
        predicted = validateExample(root, header, example)
        predictedList.append(predicted)
        if actual == 0 and predicted == 0:
            tn += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        elif actual == 1 and predicted == 0:
            fn += 1
        else:
            tp += 1
            
    testData.insert(len(df.columns), "Predicted", predictedList)
    # print(valData)
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

# Note : Handling of blank values in test example remianing
# main segment starts here
if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("Test File not supplied as an arguement..")
        quit()
    try:
        testDf = pd.read_csv(sys.argv[1])
        testDf = testDf.drop(['satisfaction_level', 'last_evaluation', 'average_montly_hours'], axis=1)
    except FileNotFoundError:
        print("Error: No CSV file exist at '", sys.argv[1], "'")
        quit()

    df = pd.read_csv("data.csv")
    # satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	promotion_last_5years	department	salary	left
    # , 'time_spend_company'
    df = df.drop(['satisfaction_level', 'last_evaluation', 'average_montly_hours'], axis=1)

    posData = df.loc[df['left'] == 0]
    negData = df.loc[df['left'] == 1]
    
    #spliting positive and negative dataset into randomly 80-20 % split
    posTrainData, posTestData = splitTrainTest(posData, 0.2)
    negTrainData, negTestData = splitTrainTest(negData, 0.2)
    
    #merging positive and negative data split so that training and validation dataset contains equal number of positive and negative value of feature label 
    trainData = pd.concat([posTrainData, negTrainData])
    valData = pd.concat([posTestData, negTestData])
    
    data = trainData.values
    headerList = df.columns.values
    
    # Call to build decision tree
    root = buildDecisionTree(data, headerList, 1, None)
    print("*************** For validation data ***************")
    main(root, valData)
    print("*************** For Test Data ***********************")
    main(root, testDf)

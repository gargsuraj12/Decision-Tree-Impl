import numpy as np
import pandas as pd
import random
import statistics as st
import queue, sys
import seaborn as sb

CAT_ATTR_THRESHOLD = 10
LESS_THAN_EQUAL = 0
GREATER_THAN = 1
CAT_ATTR = 0
NUM_ATTR = 1
ENTROPY = 0
GINI_IMPURITY = 1
MISCLASSIFICATION_RATE = 2


category_dict = {}

def getAccuracy(tn, fp, fn, tp):
    num = tp+tn
    deno = tn+fp+fn+tp
    return (num/deno)*100

def getError(tn, fp, fn, tp):
    num = tp+tn
    deno = tn+fp+fn+tp
    error = 1 - (num/deno)
    return error*100

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


def calcOverallMiscRate(data):
    cols = data.shape
    if(len(cols) == 1):
        leftCol = data
    else:
        leftCol = data[:, -1]

    _, labelCounts = np.unique(leftCol, return_counts=True)
    sampleSpace = len(leftCol)
    probabilities = labelCounts / sampleSpace
    if len(probabilities) == 0:
        return 1
    maxProb = np.max(probabilities)
    return (1 - maxProb)

def calcOverallGiniImpurity(data):
    cols = data.shape
    if(len(cols) == 1):
        leftCol = data
    else:
        leftCol = data[:, -1]

    _, labelCounts = np.unique(leftCol, return_counts=True)
    sampleSpace = len(leftCol)
    probabilities = labelCounts / sampleSpace
    if len(probabilities) == 0:
        return 1
    impurity = 1 - sum(np.square(probabilities))
    return impurity


# Calculates the overall entropy of the given data/label
def calcOverallEntropy(data):
    cols = data.shape
    if(len(cols) == 1):
        leftCol = data
    else:
        leftCol = data[:, -1]

    # leftCol = data[:, -1]
    _, labelCounts = np.unique(leftCol, return_counts=True)
    sampleSpace = len(leftCol)
    probabilities = labelCounts / sampleSpace
    if len(probabilities) == 0:
        return 0
    entropy = sum(probabilities * -(np.log2(probabilities)))
    return entropy


# Calculates the entropy of a Categorical column/attribute
def calcCatAttrAvgImpurity(data, colNum, impurityType):
    column = data[:,colNum]
    sampleSpace = len(column)
    featureValues, featureCounts = np.unique(column, return_counts=True)
    probabilities = featureCounts / sampleSpace
    featureImpurities = []

    # Calculating entropy of each feature of the column
    for val in featureValues:
        columnData = data[column == val]
        if impurityType == ENTROPY:
            impurity = calcOverallEntropy(columnData)
        elif impurityType == GINI_IMPURITY:
            impurity = calcOverallGiniImpurity(columnData)
        else:
            impurity = calcOverallMiscRate(columnData)

        featureImpurities.append(impurity)

    avgImpurity = sum(probabilities*featureImpurities)
    return None,avgImpurity


# Best splits(Using Bruteforce Method) the numerical attribute and calculates the average entropy
def calcNumAttrAvgImpurity(data, colNum, impurityType):
    totalRows,_ = data.shape
    column = data[:, colNum]
    splitPoint = None
    avgImpurity = 1000000
    uniqueFeatureValues = np.unique(column)

    # Find the best split position for this column
    for val in uniqueFeatureValues:
        dataBelow, dataAbove = splitNumAttr(data, colNum, val)
        belowRows = len(dataBelow)
        aboveRows = len(dataAbove)
        if impurityType == ENTROPY:
            belowImpurity = calcOverallEntropy(dataBelow)
            aboveImpurity = calcOverallEntropy(dataAbove)
        elif impurityType == GINI_IMPURITY:
            belowImpurity = calcOverallGiniImpurity(dataBelow)
            aboveImpurity = calcOverallGiniImpurity(dataAbove)
        else:
            aboveImpurity = calcOverallMiscRate(dataAbove)
            belowImpurity = calcOverallMiscRate(dataBelow)
            
        t_entropy = belowImpurity*(belowRows/totalRows) + aboveImpurity*(aboveRows/totalRows)
        if t_entropy < avgImpurity:
            avgImpurity = t_entropy
            splitPoint = val

    return splitPoint, avgImpurity


# Return the index of the attribute having the maximun information gain with the supplied data
def nextParentNodeAttribute(data, headerList, impurityType):
    if impurityType == ENTROPY:
        overallImpurity = calcOverallEntropy(data)
    elif impurityType == GINI_IMPURITY:
        overallImpurity = calcOverallGiniImpurity(data)
    else:
        overallImpurity = calcOverallMiscRate(data)    
    infoGainList = []
    splitValList = []
    cols = data.shape[1]-1
    splitVal, colAvgEntropy = None, None
    # Excluding 'left' column
    for i in range(cols):
        if category_dict[headerList[i]] == CAT_ATTR:
            splitVal, colAvgEntropy = calcCatAttrAvgImpurity(data, i, impurityType)
        else:    
            splitVal, colAvgEntropy = calcNumAttrAvgImpurity(data, i, impurityType)
        
        splitValList.append(splitVal)
        infoGainList.append(overallImpurity-colAvgEntropy)

    # print(infoGainList)
    maxEntropy = max(infoGainList)
    index = infoGainList.index(maxEntropy)
    splitVal = splitValList[index]
    return splitVal, index


# Building Decision Tree
def buildDecisionTree(data, headerList, depth, nodes, impurityType, maxNodes, maxHeight):
    rows,_ = data.shape
    # Base cases
    if isDataPure(data) or (len(headerList) <= 1) or rows<=5 or depth+1 == maxHeight or nodes[0] >= maxNodes:
        leaf = setLabel(data)
        nodes[0] += 1
        return depth+1, leaf

    # Create N-ary Tree
    else:
        splitVal, index = nextParentNodeAttribute(data, headerList, impurityType)
        attrName = headerList[index]
        root = InternalNode(attrName, splitVal)
        nodes[0] += 1
        # Cheking if the selected attibute is numerical attribute
        if category_dict[headerList[index]] == NUM_ATTR:
            root.isNodeNumerical = True
            data = modifyNumAttr(data, index, splitVal)

        selectedColumn = data[:, index]
        uniqueFeatureValues = np.unique(selectedColumn)
        headerList = np.delete(headerList, index, axis = 0)
        height = depth
        for i in range(len(uniqueFeatureValues)):
            feature = uniqueFeatureValues[i]
            # discarding all the rows containing particular feature values
            newData = data[data[:, index] == feature]
            # dropping the selected attribute
            newData = np.delete(newData, index, axis=1)
            # Recursion
            h, childNode = buildDecisionTree(newData, headerList, depth+1, nodes, impurityType, maxNodes, maxHeight)
            if h > height:
                height = h
            root.positives += childNode.positives
            root.negatives += childNode.negatives
            root.childDict.update({feature:childNode})

        return height, root


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
        
def PrintTree(root):
    print("Type of root is: ", type(root))
    L = queue.Queue(maxsize=2000) 
    L.put(root)
    cnt =1
    L.put(None)
    while(L.qsize()!=1):
        x = L.get()
        print(type(x))
        #if(x==None):
            #print()
            #print()
            #continue
        if(x != None):
            # print(x.splitVal,end =" ") 
            print("Attribue val is: ", x.attributeName)
            # ,end =" "
            for node in x.childDict.keys():
                L.put(node)
                cnt  = cnt + 1
        else:
            print()
            L.put(None)
    print()
    print("Number of nodes ",cnt)
    print()

# Note : Handling of blank values in test example remianing
# main segment starts here
def main(root, valData):

    tp, tn, fp, fn = 0, 0, 0, 0
    predictedList = []
    # Running validation set
    for i in range(len(valData)):
        actual = valData.values[i, :][-1]
        header = list(headerList)
        example  = valData.values[i, 0:9]
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
    
    valData.insert(len(df.columns), "Predicted", predictedList)
    # print(valData)
    print("True Negatives: ", tn)
    print("False Positves: ", fp)
    print("False Negatives: ", fn)
    print("True Positives: ", tp)
    
    accuracy = getAccuracy(tn, fp, fn, tp)
    error = getError(tn, fp, fn, tp)      
    precision = getPrecision(tp, fp)
    recall = getRecall(tp, fn)
    l = [precision, recall]
    f1Measure = st.harmonic_mean(l)

    print("Accuracy is: ", accuracy)
    print("Error is: ", error)
    print("Precision is: ", precision)
    print("Recall is: ", recall)
    
    print("F1 Measure is: ", f1Measure)
    return accuracy, precision, recall

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("Test File not supplied as an arguement..")
        quit()
    try:
        testDf = pd.read_csv(sys.argv[1])
    except FileNotFoundError:
        print("Error: No CSV file exist at '", sys.argv[1], "'")
        quit()

    df = pd.read_csv("data.csv")
    posData = df.loc[df['left'] == 0]
    negData = df.loc[df['left'] == 1]
    
    # spliting positive and negative dataset into randomly 80-20 % split
    posTrainData, posTestData = splitTrainTest(posData, 0.2)
    negTrainData, negTestData = splitTrainTest(negData, 0.2)
    
    # merging positive and negative data split so that training and validation dataset contains equal number of positive and negative value of feature label 
    trainData = pd.concat([posTrainData, negTrainData])
    valData = pd.concat([posTestData, negTestData])
    data = trainData.values
    headerList = df.columns.values
    _,cols = data.shape
    for i in range(cols):
        uniqueFeatureValues = np.unique(data[:, i])
        if len(uniqueFeatureValues) <= CAT_ATTR_THRESHOLD:
            category_dict[headerList[i]] = CAT_ATTR
        else:
            category_dict[headerList[i]] = NUM_ATTR

    maxHeight = 10
    maxNodes = 2000

    # Call to build decision tree using ENTROPY as a measure
    nodes = [0]
    print("Impurity Type: ENTROPY")
    height, root = buildDecisionTree(data, headerList, 0, nodes, ENTROPY, maxNodes, maxHeight)
    print("Actual Height is: ", height, " and total nodes are: ", nodes[0])
    print("*************** For validation data ***************")
    main(root, valData)
    print("*************** For Test Data ***********************")
    main(root, testDf)
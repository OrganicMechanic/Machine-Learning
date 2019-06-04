#!/usr/bin/env python3
# Joel Doumit - Assignment 1 for CS475. 
# Template provided by:
# Dr. Robert Heckendorn, Computer Science Department, University of Idaho, 2016
#
# ID3 decision tree algorithm
#
import sys
from math import *

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
#
# IO support for reading from stdin and writing to stdout
#

# read in a classifier problem
def readProblem():
    global Features, FeatureList, FeatureValues, Data

    FeatureList = []     # list of all the features including Ans
    FeatureValues = {}   # potential values of all the features, even ones not in examples
    Data = []            # example classification data

    # read number of features
    numFeatures = int(sys.stdin.readline())

    # read in features and answer which must be called: Ans
    for i in range(0, numFeatures+1):
        line = sys.stdin.readline().strip()
        fields = line.split()
        FeatureList.append(fields[0])
        FeatureValues[fields[0]] = fields[2:] # dictionary value is a list

    # read number of samples
    numSamples = int(sys.stdin.readline())
    
    # read in example classifications
    for line in sys.stdin.readlines():
        fields = line.split()
        sample = {}
        for i in range(0, len(FeatureList)):
            sample[FeatureList[i]] = fields[i]
        Data.append(sample)



# write out indented classifier tree
amountIndent = 3*" "

def printDTree(tree):
    printDTreeAux("", tree)
    
def printDTreeAux(indent, tree):
    name = tree[0]
    d = tree[1]
    if type(d) is dict:
        for v in FeatureValues[name]:
            print(indent + name + "=" + v)
            printDTreeAux(indent + amountIndent, d[v])
    else:
        print(indent + d)


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
#
# Decision Tree:  list either [Feature, Value] or [Feature, dictionary of value:tree]
# Author: Robert B. Heckendorn, University of Idaho
# 
# The OUTPUT:
#
# ["Ans", "Yes"]
# or 
# ["Price", {"$": tree1, "$$": tree2, "$$$": tree1}]
# 
# 
# DATA: a list of dictionary for each training case
# 
# [{"Ans": "Yes", "Alt": "Yes", "Price": "$$" ... },
#  ... ]
# 
# FEATUREVALUES: dictionary of Feature:[list of values]
# 
# {"Ans": ["Yes", "No"], "Alt": ["Yes", "No"], "Price": ["$", "$$", "$$$"] ... }
# 
# 

# 
# select(data, feature, value) -> subset of data as list of dictionaries that have feature=value
# count(data, feature, value) -> number of cases in data in which feature has value 
# entropy(data, feature) -> [0, 1], feature is generally "Ans" for ID3


# list of the items in data that have feature equal to value
def select(data, feature, value):
    return [ item for item in data if item[feature]==value ]


# count how many items in the data have feature equal to value
def count(data, feature, value):
    num = 0
    for d in data:
        if d[feature]==value : num+=1
    return num

def calcEntropy(dictVar):
    entropyList = []
    for i in dictVar:
        p = dictVar[i]/sum(dictVar.values())
        entropyVal = p*log2(p)
        entropyList.append(entropyVal)
    
    totalEntropyVal = sum(entropyList)
    return -totalEntropyVal

def finalCalc(dictVar, ansList):
        listToSum = []
        totalDict = 0

        for i in dictVar:
            # A running total of values from our dictionary.
            totalDict += sum(dictVar[i])

        for i in dictVar:
            entropyList = []
            p = sum(dictVar[i])/totalDict
            for j in range(0, len(ansList)):
                dictVarVal = dictVar[i][j]
                if dictVar[i][j] <= 0:
                    entropyList.append(0)
                
                else:
                    dictSum = sum(dictVar[i])
                    ratio = dictVarVal/dictSum
                    entToAppend = ratio*log2(ratio)
                    entropyList.append(entToAppend)
            
            temp = p*sum(entropyList)
            listToSum.append(temp)
        finalVal = -sum(listToSum)
        return finalVal


# what is the entropy of a question about feature?
# sum the entropy over the possible values of the feature.
def entropy(data, feature):
    if feature == "Ans":
        # Create a variable for storing Ans.
        ansSet = set()
        
        for i in data:
            # Add each Ans from feature.
            ansSet.add(i[feature])
        
        # Make a list of ans values, and prepare a dictionary.
        ansList = list(ansSet)
        dictVar = {}
        
        for i in ansList:
            dictVar[i] = 0
        
        for i in data:
            for j in ansList:
                for k in range(0, len(ansList)):
                    if i["Ans"] == ansList[k] and i[feature] == j:
                        dictVar[j] += 1

        r = calcEntropy(dictVar)
        return r

    # If the value of feature isn't "Ans"
    else:
        featureSet = set()
        ansSet = set()
        for i in data:
            featureSet.add(i[feature])

        for i in data:
            ansSet.add(i["Ans"])
        featureList = list(featureSet)
        ansList = list(ansSet)
        dictVar = {}

        for i in featureList:
            dictVar[i] = [0,0]

        for i in data:
            for j in featureList:
                for k in range(0, len(ansList)):
                    if i["Ans"] == ansList[k] and i[feature] == j:
                        print(dictVar)
                        #dictVar[j][k] += 1
        
        v = finalCalc(dictVar, ansList)
        return v

# current entropy - expected entropy after getting info about feature 
# entropy(data, "Ans") - sum_{v=featurevalues} p_v * entropy(select(data, feature, v), "Ans")
# The summing of featurevalues is done in the finalCalc function, which is why it's
# not included here.
def gain(data, feature):
    gainAns = entropy(data, "Ans") - entropy(data, feature)
    return gainAns


# If there one and only one value for the given feature in given data 
# If not return None
def isOneLabel(data, feature):
    valList = [] # The list of values for the given feature
    for i in data:
        if (i[feature] not in valList):
            valList.append(i[feature]) # Append that feature value to the list
    if len(valList) == 1: 
        return 1 # Returning None causes the output to blow up.
    else: 
        return 0
    

# select the most popular Ans value left in the data for the constraints
# up to now.
def maxAns(data):
    currentMax = "" # Current maximum value, which gets updated.
    occNum = 0  # Count of occurences of current most popular value.
    for i in data:
        rowNum = count(data, "Ans", i["Ans"]) # Number of rows from data with same "Ans"
        if (rowNum > occNum):
            occNum = rowNum
            currentMax = i["Ans"]
    return currentMax

# this is the ID3 algorithm
def ID3BuildTree(data, availableFeatures):
    # if data is empty
    if (len(data) == 0):
        return ["Ans", "None"]

    # only one label for the Ans feature at this point?
    elif (isOneLabel(data, "Ans")):
        return ["Ans", data[0]["Ans"]]

    # ran out of discriminating features
    elif (len(availableFeatures) == 0):
        return ["Ans", maxAns(data)]

    # pick maximum information gain
    else:
        Epsilon = 1E-10
        bestFeature = None
        bestGain = None
        for feature in availableFeatures:
            g = gain(data, feature)
            print("GAIN: ", feature, ":", round(g, 4));
            if bestGain == None or g > (bestGain + Epsilon):
                bestGain = g
                bestFeature = feature
                bestList = [feature]
            elif g==bestGain:
                bestList.append(feature)
        print("BEST:", round(bestGain, 4), bestList);
        print()
            
        # recursively construct tree on return
        treeLeaves = {}   # start with empty dictionary
        availableFeatures = availableFeatures[:]
        availableFeatures.remove(bestFeature)
        for v in FeatureValues[bestFeature]:
            treeLeaves[v] = ID3BuildTree(select(data, bestFeature, v), availableFeatures)
        return [bestFeature, treeLeaves]    # list of best feature and dictionary of trees


# do the work
def main():
    readProblem()
    FeatureList.remove("Ans")
    tree = ID3BuildTree(Data, FeatureList)
    printDTree(tree)

main()


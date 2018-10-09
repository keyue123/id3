#!/usr/bin/python                                                                                                                                                                                                                    
#coding=utf-8

#File Name: test.py
#Author   : john
#Mail     : john.y.ke@mail.foxconn.com 
#Created Time: Sat 01 Sep 2018 05:38:56 PM CST

# !/usr/bin/python
# coding=utf-8

# File Name: ID3.py
# Author   : john
# Created Time: Fri 31 Aug 2018 10:19:40 AM CST

from math import log

def createDataSet():
    # outlook: 0 rain   1 overcast   2 sunny
    # tem:     0 cool   1 mild       2 hot
    # hum:     0 normal 1 high
    # windy    0 not    1 medium     2 very
    dataSet = [[1, 2, 1, 0, 'no'],
               [1, 2, 1, 2, 'no'],
               [1, 2, 1, 1, 'no'],
               [2, 2, 1, 0, 'yes'],
               [2, 2, 1, 1, 'yes'],
               [0, 1, 1, 0, 'no'],
               [0, 1, 1, 1, 'no'],
               [0, 2, 0, 0, 'yes'],
               [0, 0, 0, 1, 'no'],
               [0, 2, 0, 2, 'no'],
               [2, 0, 0, 2, 'yes'],
               [2, 0, 0, 1, 'yes'],
               [1, 1, 1, 0, 'no'],
               [1, 1, 1, 1, 'no'],
               [1, 0, 0, 0, 'yes'],
               [1, 0, 0, 1, 'yes'],
               [0, 1, 0, 0, 'no'],
               [0, 1, 0, 1, 'no'],
               [1, 1, 0, 1, 'yes'],
               [1, 2, 0, 2, 'yes'],
               [2, 1, 1, 2, 'yes'],
               [2, 1, 1, 1, 'yes'],
               [2, 2, 0, 0, 'yes'],
               [0, 1, 1, 2, 'no'], ]

    return dataSet

# 获取数据集的熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLable = featVec[-1]  # 取得最后一列数据
        if currentLable not in labelCounts.keys():  # 获取结果
            labelCounts[currentLable] = 0
        labelCounts[currentLable] += 1

    # 计算熵
    Ent = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        Ent -= prob * log(prob, 2)
    # print ("信息熵: ", Ent)
    return Ent


# 划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:  # 每行中第axis个元素和value相等（去除第axis个数据）
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet  # 返回分类后的新矩阵


# 根据香农熵，选择最优的划分方式    #根据某一属性划分后，类标签香农熵越低，效果越好
def chooseBestFeatureToSplit(dataSet):
    baseEntropy = calcShannonEnt(dataSet)  # 计算数据集的香农熵
    numFeatures = len(dataSet[0]) - 1
    bestInfoGain = 0.0  # 最大信息增益
    bestFeature = 0  # 最优特征

    for i in range(0, numFeatures):
        featList = [example[i] for example in dataSet]  # 所有子列表（每行）的第i个元素，组成一个新的列表
        uniqueVals = set(featList)
        newEntorpy = 0.0
        for value in uniqueVals:  # 数据集根据第i个属性进行划分，计算划分后数据集的香农熵
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntorpy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntorpy  # 划分后的数据集，香农熵越小越好，即信息增益越大越好
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# 如果数据集已经处理了所有属性，但叶子结点中类标签依然不是唯一的，此时需要决定如何定义该叶子结点。这种情况下，采用多数表决方法，对该叶子结点进行分类
def majorityCnt(classList):  # 传入参数：叶子结点中的类标签
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
            classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 创建树
def createTree(dataSet, labels):  # 传入参数：数据集，属性标签（属性标签作用：在输出结果时，决策树的构建更加清晰）
    classList = [example[-1] for example in dataSet]  # 数据集样本的类标签
    if classList.count(classList[0]) == len(classList):  # 如果数据集样本属于同一类，说明该叶子结点划分完毕
        return classList[0]
    if len(dataSet[0]) == 1:  # 如果数据集样本只有一列（该列是类标签），说明所有属性都划分完毕，则根据多数表决方法，对该叶子结点进行分类
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 根据香农熵，选择最优的划分方式
    bestFeatLabel = labels[bestFeat]  # 记录该属性标签
    myTree = {bestFeatLabel: {}}  # 树
    del (labels[bestFeat])  # 在属性标签中删除该属性
    # 根据最优属性构建树
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        subDataSet = splitDataSet(dataSet, bestFeat, value)
        myTree[bestFeatLabel][value] = createTree(subDataSet, subLabels)
    print("myTree:", myTree)
    return myTree


# 测试算法：使用决策树，对待分类样本进行分类
def classify(inputTree, featLabels, testVec):  # 传入参数：决策树，属性标签，待分类样本
    firstStr = list(inputTree.keys())[0]  # 树根代表的属性
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)  # 树根代表的属性，所在属性标签中的位置，即第几个属性
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


if __name__ == '__main__':
    dataSet = createDataSet()
    labels = ['outlook', 'tem', 'hum', 'windy']

    labelsForCreateTree = labels[:]
    Tree = createTree(dataSet, labelsForCreateTree)
    testvec = [2, 2, 1, 0]
    print(classify(Tree, labels, testvec))
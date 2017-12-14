from numpy import *
from math import log
from os import listdir
import operator
import matplotlib
import matplotlib.pyplot as plt
import treePlotter

print("hello this tree.py")

#计算信息熵
def calcShannnonEnt(dataset):
    numEntries = len(dataset)
    labelCounts = {}
    for featVec in dataset:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

#创建测试数据
def createDataSet():
    dataset = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no suffacing','flippers']
    return dataset,labels

#按照给定的特征划分数据
#待划分的数据集，划分数据集的特征，需要返回的特征的值
#axis:表示的是列，value表示的是值
def splitDataSet(dataset,axis,value):
    retDataSet = []
    for featvec in dataset:
        if featvec[axis] == value:
            reducedFeatVec = featvec[:axis]    #将axis之前的列赋值
            reducedFeatVec.extend(featvec[axis+1:]) #将axis之后的列赋值,a.extend(b)是将[a,b],a.append(a,[b])
            retDataSet.append(reducedFeatVec)
    return retDataSet

#选择最好的数据集划分方式
#计算信息增益同时选取最大的信息增益 gain = H(D) - H(D/A)
def chooseBestFeatureToSplit(dataset):
    numFeatures = len(dataset[0]) -1#获取特征的个数，由于最后的一列为标签，所以需要减掉1，len,每行的数据长度，文件夹下数据个数
   # print(len(dataset[0]))
   # print(dataset[0])
    baseEntropy = calcShannnonEnt(dataset)   #计算H(D)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataset] #featlist 取的是每一列的值
        #print(featList)
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataset,i,value)
            prob = len(subDataSet)/float(len(dataset))
            newEntropy += prob * calcShannnonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
        return bestFeature

#对于叶子节点不属于一个类，采用投票表决的方法进行。字典对象存储了每个类标签出现的频率
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]

#创建树的函数代码
def createTree(dataset,labels):
    Labels = labels[:]
    classList = [example[-1] for example in dataset]  #classList : yes yes no no no
    #print(classList[0])
    #print(classList.count(classList[0]))  #classlist[0] = yes ,yes的计数为2
    #print(len(classList))  #这个是长度为5
    if classList.count(classList[0]) == len(classList):   #划分的节点类别完全相同，则返回
        return classList[0]
    if len(dataset[0]) == 1:                            #遍历所有特征时候，类别没有完全停止，只能采用多数表决的方式
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataset)   #选择出来的是类的标号
    bestFeatLabel = Labels[bestFeat]               #labels 代表了特征，标号和特征一一对应
    myTree = {bestFeatLabel:{}}
    del(Labels[bestFeat])
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = Labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataset,bestFeat,value),subLabels)
    return myTree
#使用决策树进行分类
#比较testVec变量中的值与树节点的值，如果到达叶子节点，则返回当前节点的分类标签
def classify(inputTree,featLabels,testVec):
    firstSides = list(inputTree.keys())
    firstStr = firstSides[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)

    for key in list(secondDict.keys()):
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()

def grapTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

def testMyTree():
    myDat,mylabels = createDataSet()
    mytree = createTree(myDat,mylabels)

    bestres = chooseBestFeatureToSplit(myDat)
    print(bestres)

    seleMat = splitDataSet(myDat,0,1)
    print(seleMat)

    shanno = calcShannnonEnt(myDat)
    print(shanno)

    print('我的分类')
    print (mylabels)
    myLabel = classify(mytree,mylabels,[1,0])
    print(myLabel)
    myLabel1 = classify(mytree,mylabels,[1,1])
    print(myLabel1)

def classfiGlass():
    fr = open('decisiontree/lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age','prescript','astigmatic','tearRate']
    lensesTree = createTree(lenses,lensesLabels)
    treePlotter.createPlot(lensesTree)

classfiGlass()

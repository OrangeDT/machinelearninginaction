#回归树
from numpy import *
from math import *
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine))  #对于python2中的map，list
        dataMat.append(fltLine)
    return dataMat
#数据集合，待切分的特征，该特征的值，数组过滤的方式切分两个子集并返回
def binSplitDataSet(dataSet,feature,value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:][:]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:][:]
    return mat0,mat1

def regLeaf(dataSet):
    return mean(dataSet[:,-1],0)
def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]
def chooseBestSplit(dataSet,leafType = regLeaf,errType = regErr,ops = (1,4)):
    tolS = ops[0];tolN = ops[1]
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        print(dataSet)
        print(set(dataSet[:,-1].T.tolist()[0]))
        return None,leafType(dataSet)
    m,n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf;bestIndex = 0;bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex]):
            mat0,mat1 = binSplitDataSet(dataSet,featIndex,splitVal)
            if(shape(mat0)[0] < tolN) or(shape(mat1)[0] < tolN):continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if(S - bestS) < tolS:
        return None,leafType(dataSet)
    mat0,mat1 = binSplitDataSet(dataSet,bestIndex,bestValue)
    if(shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None,leafType(dataSet)
    return bestIndex,bestValue
def createTree(dataSet,leafType = regLeaf,errType = regErr,ops = (1,4)):
    feat,val = chooseBestSplit(dataSet,leafType,errType,ops)
    if feat == None:return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet,rSet = binSplitDataSet(dataSet,feat,val)
    retTree['left'] = createTree(lSet,leafType,errType,ops)
    retTree['right'] = createTree(rSet,leafType,errType,ops)
    return retTree
#决策树的剪枝
def isTree(obj):
    return(type(obj).__name__ == 'dict')
def getMean(tree):
    if isTree(tree['right']):tree['right'] =getMean(tree['right'])
    if isTree(tree['left']):tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right'])/2.0
def prune(tree,testData):
    if shape(testData)[0] == 0:return getMean(tree)
    if(isTree(tree['right'])) or isTree(tree['left']):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
    if isTree(tree['left']):tree['left'] = prune(tree['left'],lSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) + \
            sum(power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:return tree
    else:return tree


myDat = loadDataSet('./regTree/ex00.txt')
myMat = mat(myDat)
regTree = createTree(myMat)









# testMat = mat(eye(4))
# print(testMat)
# print()
# print()
#
# mat0,mat1 = binSplitDataSet(testMat,1,0.5)
# print(mat0)
# print()
# print(mat1)


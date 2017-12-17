#利用PCA进行降维度
from numpy import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#python3的map返回的是一迭代器
def loadDataSet(fileName,delim = '\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float,line)) for line in stringArr]
    return mat(datArr)
def pca(dataMat,topNfeat = 9999999):
    meanVals = mean(dataMat,axis = 0)
    meanRemoved = dataMat - meanVals
    covMat = cov(meanRemoved,rowvar = 0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat + 1):-1]
    redEigVects = eigVects[:,eigValInd]
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat,reconMat
dataMat = loadDataSet('./pca/testSet.txt')
lowDMat,reconMat = pca(dataMat,51)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker = '^',s = 90)
ax.scatter(reconMat[:,0].flatten().A[0],reconMat[:,1].flatten().A[0],marker = 'o',s = 50,c= 'red')
plt.show()





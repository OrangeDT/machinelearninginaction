#逻辑回归的python实现
from math import exp
from numpy import *
def loadDataSet():
    dataMat = [];labelMat = []
    fr = open('log/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(intX):
    return 1.0/(1+exp(-intX))
#批量梯度下降
def gradAscent(dataMatIn,classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)#m行n 列
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1)) #[[1],[1],[1]]
    # print(weights)
    # weithtss = ones(n)  #[1,1,1]
    # print(weithtss)
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights
# dataArr,labelMat = loadDataSet()
# myWeights = gradAscent(dataArr,labelMat)
# print(myWeights)

#画出决策边界
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = [];ycord1 = []
    xcord2 = [];ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s = 30,c = 'red',marker = 's')
    ax.scatter(xcord2,ycord2,s = 30,c = 'green')
    x = arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1] * x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2');
    plt.show()

#plotBestFit(myWeights.getA())  #将得到的数组转换成矩阵再作为参数

#随机梯度下降
def stocGradAscent0(dataMatrix,classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)  #[1,1,1]
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        errors = classLabels[i] - h
        weights = weights + alpha * errors * dataMatrix[i]
    return weights
#改进的随机梯度下降
def stocGradAscent1(dataMatrix,classLabels,numIter = 150):
    m,n = shape(dataMatrix) #100行，3列
    weights = ones(n)   #[1,1,1]
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01        #alpha 每次迭代时候都需要调整
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))  #随机选取样本更新参数
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


#预测疝气病马的存活率
def classifyVector(intX,weights):
    prob = sigmoid(sum(intX*weights))
    if prob > 0.5 : return 1.0
    else:return 0.0
def colicTest():
    frTrain = open('horse/horseColicTraining.txt')
    frTest = open('horse/horseColicTest.txt')
    trainingSet = [];trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet),trainingLabels,500)
    errorCount = 0;numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate
def multiTest():
    numTests = 10;errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print ('after %d iterations ther average error rate is :%f '%(numTests,errorSum/float(numTests)))

multiTest()
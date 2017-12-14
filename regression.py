from numpy import *
from math import *
import matplotlib.pyplot as plt
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) -1
    dataMat = [];labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat
#标准回归函数
def standRegres(xArr,yArr):
    xMat = mat(xArr);yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:
        print( "this matrix is singular, can not do inverse")
        return
    ws = xTx.I * (xMat.T * yMat)    #xx.I是求逆
    return ws

#局部加权线性回归
def lwlr(testPoint,xArr,yArr,k = 1.0):
    xMat = mat(xArr);yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))#获取M行M列的单位矩阵
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat * diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular,can not do inverse")
        return
    ws = xTx.I * (xMat.T *(weights * yMat))
    return testPoint * ws
def lwlrTest(testArr,xArr,yArr,k = 1.0):
    m = shape(testArr)[0]   #获取多少行数的数据
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

def test_standRegress():
    xArr,yArr = loadDataSet('./regression/ex0.txt')
    ws = standRegres(xArr,yArr)
    print(ws)
    xMat = mat(xArr)
    yMat = mat(yArr)
    yHat = xMat * ws   #预测的数值
    corr = corrcoef(yHat.T,yMat)
    print(corr)
    #绘制原始数据图像
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0],s=2,c = 'red')
    #绘制生成一个图像
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:,1],yHat)
    plt.show()
#test_standRegress()

def test_lwlr_plot():
    xArr,yArr = loadDataSet('./regression/ex0.txt')
    print(yArr[0])
    re1 = lwlr(xArr[0],xArr,yArr,1.0)
    print(re1)
    re2 = lwlr(xArr[0],xArr,yArr,0.001)
    print(re2)

    #得到的是所有的点的值
    yHat = lwlrTest(xArr,xArr,yArr,1)
    xMat = mat(xArr)
    srtInd = xMat[:,1].argsort(0)
    xSort = xMat[srtInd][:,0,:]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:,1],yHat[srtInd])
    # print(xMat[:,1])
    # print(xMat[:,1].flatten())
    # print(xMat[:,1].flatten().A[0])

    ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2,c = 'red')
    plt.show()

test_lwlr_plot()

#实验：预测鲍鱼的年龄
def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()

def test_baoyu_age():
    abX,abY = loadDataSet('./regression/abalone.txt')
    #训练集上
    yHat01 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
    yHat1 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],1)
    yHat10 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],10)

    rssError01 = rssError(abY[0:99],yHat01.T)
    rssError1 = rssError(abY[0:99],yHat1.T)
    rssError10 = rssError(abY[0:99],yHat10.T)
    print(rssError01,rssError1,rssError10)
    #测试集上
    yHat01 = lwlrTest(abX[100:199],abX[100:199],abY[100:199],0.1)
    yHat1 = lwlrTest(abX[100:199],abX[100:199],abY[100:199],1)
    yHat10 = lwlrTest(abX[100:199],abX[100:199],abY[100:199],10)

    rssError01 = rssError(abY[100:199],yHat01.T)
    rssError1 = rssError(abY[100:199],yHat1.T)
    rssError10 = rssError(abY[100:199],yHat10.T)
    print(rssError01,rssError1,rssError10)




#岭回归
def ridgeRegres(xMat,yMat,lam = 0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print("this matrix is singular,cannot do inverse")
        return
    ws = denom.I *(xMat.T *yMat)
    return ws
def ridgeTest(xArr,yArr):
    xMat = mat(xArr);yMat = mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    xMean = mean(xMat,0)
    xVar = var(xMat,0)  #数据进行标准化，所有的数据减掉均值然后除以房产
    xMat = (xMat - xMean)/xVar
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10)) #为了进行比较采用了30个lamb,比较lamb取很大和很小的时候的差别
        wMat[i,:] = ws.T
    return wMat
def test_ridgeRegres():
    abX,abY = loadDataSet('./regression/abalone.txt')
    ridgeWeights = ridgeTest(abX,abY) #得到30个不同lamb所对应的回归系数
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()

#test_ridgeRegres()

#前向逐步线性回归
def stageWise(xArr,yArr,eps = 0.1,numIt = 100):
    xMat = mat(xArr);yMat = mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean

    xMean = mean(xMat,0)
    xVar = var(xMat,0)
    xMat = (xMat - xMean)/xVar
    m,n = shape(xMat)
    returnMat = zeros((numIt,n))
    ws = zeros((n,1));wsTest = ws.copy();wsMat = ws.copy()
    for i  in range(numIt):
        print(ws.T)
        lowestError = inf;
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat *wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T
    return returnMat

def test_stageWise():
    xArr,yArr = loadDataSet('./regression/abalone.txt')
    stageWise(xArr,yArr,0.01,200)
test_stageWise()





















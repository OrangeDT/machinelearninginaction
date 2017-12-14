from numpy import *
# 测试数据集
def loadSimpleData():
    datMat = matrix([[1.,2.1],
                     [2.,1.1],
                     [1.3,1.],
                     [1.,1.],
                      [2.,1.]])
    classLabels = [1.0,1.0,-1.0,-1.0,1.0]
    return datMat,classLabels
#单层决策树生成函数，如何获取到单层最佳决策树
#dataMatrix:数据集
#dimen :维度，选择的是哪一列
#threshVal:阈值，最小值+步长*步值，
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen]  <= threshVal] = -1.0  #实现数组的过滤，满足要求的设置为
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

#数据集，类标签，权重
def bulidStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr);labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = mat(zeros((m,1)))
    minError = inf
    for i in range(n):   #对数据集上的每一列特征
        rangeMin = dataMatrix[:,i].min();rangeMax = dataMatrix[:,i].max();
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps) + 1):   #对每一个步长，大值减去小值然后除以步伐，得到每一步的步长
            for inequal in ['lt','gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0 #errArr,得到的分错的数组，分对了了的，标志为0
                weightedError = D.T * errArr    #这句话要获得错误的分类权值 [0.2,0.2,0.2,0.2,0.2]*errArr
              #  print("split:dim %d,thresh %.2f,thresh ineqal:%s,the weighted error is %.3f "\
               #       %(i,threshVal,inequal,weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst

#基于单层决策树的adaboost的 决策过程
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = bulidStump(dataArr,classLabels,D)
       # print("D:",D.T)
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        #print("classEst:",classEst.T)
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)  #对应的标记相乘，如果没有分类正确,计算结果为+alpha
        D = multiply(D,exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEst
       # print("aggClassEst:",aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum() /m
       # print('total error:',errorRate,"\n")
        if errorRate == 0.0 :break
    #return weakClassArr
    return weakClassArr,aggClassEst   #为了测试ROC需要修改一下
def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                    classifierArr[i]['thresh'],\
                                    classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        #print(aggClassEst)
    return sign(aggClassEst)
#训练简单的数据
def classifySimpleData():
    datArr,labelArr = loadSimpleData()
    classifierArr = adaBoostTrainDS(datArr,labelArr,30)
    signn = adaClassify([0,0],classifierArr)
    print(signn)

#训练疝气病马的死亡率
#自适应数据加载函数
def loadDataSet(fileName):
    numTest = len(open(fileName).readline().split('\t'))
    dataMat = [];labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numTest-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def TestHorseDepth(times):
    datArr,labelArr = loadDataSet('horse2/horseColicTraining2.txt')
    classifierArray = adaBoostTrainDS(datArr,labelArr,times)

    testArr,testlabelArr = loadDataSet('horse2/horseColicTest2.txt')
    prediction10 = adaClassify(testArr,classifierArray)
    errArr = mat(ones((67,1)))
    errorate = (errArr[prediction10!=mat(testlabelArr).T].sum()) / 67
    return errorate

def testOverfitting():
    times=[1,10,50,100,500,1000,2000]
    errorrateArr = []
    for i in range(len(times)):
        errorate = TestHorseDepth(times[i])
        errorrateArr.append(errorate)
    print(errorrateArr)


#绘制ROC曲线
def plotROC(predStrengths,classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0)
    ySum = 0.0
    numPosClas = sum(array(classLabels) == 1.0)
    yStep = 1/float(numPosClas)
    xStep = 1/float(len(classLabels) - numPosClas)
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0;delY = yStep;
        else:
            delX = xStep;delY = 0;
            ySum += cur[1]
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY],c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel("False Positive Rate");plt.ylabel("True Positive Rate")
    plt.title("ROC curve for AdsBoost Horse Colic Detection System")
    ax.axis([0,1,0,1])
    plt.show()
    print("the area Under the curve is:",ySum * xStep)

def testROC():
    datArr,labelArr = loadDataSet('horse2/horseColicTraining2.txt')
    classifierArray,aggClassEst = adaBoostTrainDS(datArr,labelArr,10)
    plotROC(aggClassEst.T,labelArr)

testROC()
from numpy import *
from os import listdir
import operator
import matplotlib
import matplotlib.pyplot as plt

print("hello this 2-1KNN")

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group , labels

#KNN的算法实现，
def classify0 (inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]   #获取Set的大小
  #  print(dataSetSize)
    diffMat = tile(inX,(dataSetSize,1)) - dataSet  #tile的作用，将inX重叠四行一列，然后与dataSet相减
  #  print(diffMat)
    sqDiffMat = diffMat ** 2   #将每一个元素的值平方
  #  print(sqDiffMat)
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()  #得到的是排序后的索引
#    print(sortedDistIndicies)
    classCount = {}
    for i in range(k):
        votelabel = labels[sortedDistIndicies[i]]
      #  print(votelabel)
        classCount[votelabel] = classCount.get(votelabel,0) + 1
      #  print(classCount)
   # print(classCount)
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]

group,labels = createDataSet()
res = classify0([0,0],group,labels,3)
print (res)

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()    #读取一行数据
    numberOfLines = len(arrayOLines)   #每行数据的长度
    returnMat = zeros((numberOfLines,3))  #创建以0填充的二维数组，三列数据
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()     #截取掉所有的回车字符
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector
#均一化数值
def autoNorm(dataset):
    minval = dataset.min(0)
    maxval = dataset.max(0)
    ranges = maxval - minval
    normDataSet = zeros(shape(dataset))  #h零初始化相等大小的数组
    m = dataset.shape[0]
    normDataSet = dataset - tile(minval,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minval

def Test_data():
    datingDataMat,datingLabels = file2matrix('KNN/datingTestSet2.txt')
    print(datingDataMat[0:20])
    fig  = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(datingLabels),15.0*array(datingLabels))
    plt.show()

    normal,ranges,minval = autoNorm(datingDataMat)
    print("均一化数据")
    print(normal[0:10])
    print(ranges)
    print(minval)

#约会网站人群分类
def classifyPerson():
    resultList = ['little','small','big']
    percentTats = float(input("time spent playing video game"))
    ffmiles = float(input("flier miles earned per year"))
    icescream = float(input("ice cream consumed per year"))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minval = autoNorm(datingDataMat)
    inArr = array([ffmiles,percentTats,icescream])
    classifierResult = classify0((inArr-minval)/ranges,normMat,datingLabels,3)
    print("you will probably like this person",resultList[classifierResult-1])

def classifyPerson():
    classifyPerson()

#将一个32*32的二进制图像转换成1*1024的向量
def img2vector(filename):
    returnVect = zeros((1,1024))   #1行  1024列
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        #print(lineStr)
        for j in range(32):
           # print(lineStr[j])
            returnVect[0,32*i + j] = int(lineStr[j])   #读到的每一个字符，将字符赋值给一维数组对应的位置
    return returnVect

testVector = img2vector('testDigits/0_13.txt')
print(testVector[0,:])

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)#len表示的是文件夹下文件的多少
    trainingMat = zeros((m,1024))
    for i in range(m):                       #获取训练数据和分类的标志
        fileNameStr = trainingFileList[i]    #文件名如：0_10.txt,  1_12.txt
        fileStr = fileNameStr.split('.')[0]  #文件名:0_10
        classNumStr = int(fileStr.split('_')[0]) #得到标志  0
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print ("came back with:%d,real answer:%d" % (classifierResult,classNumStr))
        if(classifierResult != classNumStr):
            errorCount += 1.0
    print("total number of errors %d" % errorCount)
    print("total number of error rate is: %f" % (errorCount/float(mTest)))


handwritingClassTest()


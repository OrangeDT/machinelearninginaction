from numpy import *
def loadDataSet():
    postingList = [['my','dog','has','flea','problems','help','please'],   #0
                   ['maybe','not','take','him','to','dog','park','stupid'],#1
                    ['my','dalmation','is','so','cute','I','love','him'],  #0
                   ['stop','posting','stupid','worthless','garbage'],      #1
                   ['mr','licks','ate','my','steak','how','to','stop','him'], #0
                   ['quit','buying','worthless','dog','food','stupid']]    #1
    classVec = [0,1,0,1,0,1]  #1代表侮辱性的言论，0代表正常言论
    return postingList,classVec
#创建一个包含所有文档出现的不重复的列表
def createVocalList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)
def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word:%s is not in my Vocabulary!" % word)
    return returnVec



#朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) /float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0   #防止出现分子为0的情况
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)   #取log防止下溢
    p0Vect = log(p0Num /p0Denom)
    return p0Vect,p1Vect,pAbusive
#朴素贝叶斯分类
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify*p1Vec) + log(pClass1)
    p0 = sum(vec2Classify*p0Vec) + log(1-pClass1)
    if p1 > p0 :
        return 1
    else:
        return 0
#测试NaiveBayes函数
def testingNB():
    listOposts,listClasses = loadDataSet()
    myVocabList = createVocalList(listOposts)
    print(myVocabList)
    word2vec1 = setOfWords2Vec(myVocabList,listOposts[0])
    print(word2vec1)
    word2vec2 = setOfWords2Vec(myVocabList,listOposts[1])
    print(word2vec2)

    trainmat = []
    for postinDoc in listOposts:
        trainmat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNB0(trainmat,listClasses)
    print(p0V,p1V,pAb)

    testEntry = ['love','my','dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print (testEntry,'classified as : ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print (testEntry,'classified ad : ',classifyNB(thisDoc,p0V,p1V,pAb))

#词集模型：一个词在文档中出现与否作为一个特征
#词袋模型：一个词在文档中出现不止出现一次
def bagOfWords2Vec(vocabList,inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

#############################################
#垃圾邮件过滤，并进行交叉验证
#############################################
############################################
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = [];classList =[];fullText = []
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocalList(docList)  #包含文档不重复的列表
    trainingSet = list(range(50));testSet=[]  #range返回的是range对象，不返回数组对象
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = [];trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is:',float(errorCount)/len(testSet))

spamTest()
#从大规模数据集中寻找物品间隐含关系
def loadDataSet():
    return[[1,3,4],[2,3,5],[1,2,3,5],[2,5]]
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset,C1))
def scanD(D,Ck,minSupport):
    ssCnt = {}
    #print(list(D))
    #print(numItems)
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not can in ssCnt:
                    ssCnt[can] = 1
                else:ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0,key)
            supportData[key] = support
    return retList,supportData


#apriori算法
#怎么会生成这么多的235？？
def aprioriGen(Lk,k): #由频繁项集生成候选项集元素
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk): #有4个频繁一项集，1235.[2,3] [2,5],[3,5],[1,3],i = 0
        for j in range(i+1,lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:                       #因为Lk[*]的前K-1个排序好的元素相等，所以求并集即得k + 1的元素
                retList.append(Lk[i] | Lk[j])  #
    return retList
def apriori(dataSet,minSupport = 0.5):
    C1 = createC1(dataSet)
    D = list(map(set,dataSet))
    L1,supportData = scanD(D,C1,minSupport)  #扫描数据获得满足支持度的
    L = [L1]
    k = 2
    while(len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2],k)
        Lk,supK = scanD(D,Ck,minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L,supportData
def calcConf(freqSet,H,supportData,brl,minConf = 0.7):
    prunedH = []
    for conSeq in H:
        conf = supportData[freqSet] / supportData[freqSet - conSeq]
        if conf >= minConf:
            print (freqSet - conSeq,'-->',conSeq,'conf:',conf)
            brl.append((freqSet-conSeq,conSeq,conf))
            prunedH.append(conSeq)
    return prunedH
def rulesFromConseq(freqSet,H,supportData,brl,minConf = 0.7):
    m = len(H[0])
    if(len(freqSet) > (m + 1)):
        Hmp1 = aprioriGen(H,m + 1)
        Hmp1 = calcConf(freqSet,Hmp1,supportData,brl,minConf)
        if(len(Hmp1) > 1):
            rulesFromConseq(freqSet,Hmp1,supportData,brl,minConf)
def generateRules(L,supportData,minConf = 0.7):
    bigRuleList = []
    for i in range(1,len(L)):   #要从频繁2项集开始
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):        #不是频繁2项集了，是频繁3项集怎么计算
                rulesFromConseq(freqSet,H1,supportData,bigRuleList,minConf)
            else:
                calcConf(freqSet,H1,supportData,bigRuleList,minConf)
    return bigRuleList

dataSet = loadDataSet()
L,suppData = apriori(dataSet,minSupport = 0.5)
print(L)
print("hahfa")
print(suppData)
print("rules")

rules = generateRules(L,suppData,minConf = 0.5)
print(rules)



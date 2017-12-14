import matplotlib.pyplot as plt

#定义描述树节点格式的常量
decisionNode = dict(boxstyle = "sawtooth" ,fc = "0.8")
leafNode = dict(boxstyle = "round4", fc = "0.8")
arrow_args = dict(arrowstyle = "<-")

#执行实际的绘图功能，该函数需要一个绘图区，该区域由全局变量createPlot.ax1定义
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeTxt,xy = parentPt,xycoords = 'axes fraction',\
                            xytext=centerPt,textcoords = 'axes fraction',\
                          va = "center",ha = "center",bbox=nodeType,arrowprops = arrow_args)
#首先创建了一个新图形并清空绘图区，然后在绘图区上绘制两个代表不同类型的树节点
# def createPlot():
#     fig = plt.figure(1,facecolor='white')
#     fig.clf()
#     createPlot.ax1=plt.subplot(111,frameon=False)
#     plotNode('决策节点',(0.5,0.1),(0.1,0.5),decisionNode)
#     plotNode('叶节点',(0.8,0.1),(0.3,0.8),leafNode)
#     plt.show()

#采用嵌套的字典进行树的存储
#树的结构类似于：{'no  suffacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
#获取叶节点的数目和树的层数
def getNumLeafs(mytree):
    numleafs = 0
    #firstStr = mytree.keys()[0]  #python版本的问题，python2的表现方式
    firstSides = list(mytree.keys())
    firstStr = firstSides[0]
    secondDict = mytree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__== 'dict':  #使用Type类型判断，子节点是否也是一个字典类型
            numleafs += getNumLeafs(secondDict[key])
        else:
            numleafs += 1
    return numleafs

#获取树的层数
def getTreeDepth(mytree):
    maxdepth = 0
    #firstStr = mytree.keys()[0]     #python2的方法
    firstSides = list(mytree.keys())
    firstStr = firstSides[0]
    secondDict = mytree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxdepth:
            maxdepth = thisDepth
    return maxdepth

def retrieveTree(i):
    listOfTrees = [{'no  suffacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacign':{0:'no',1:{'flippers':{0:{'head':{0:'no',1:'yes'}},1:'no'}}}}]
    return listOfTrees[i]

def testLeafsDepth():
    myTree = {'no suffacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
    #myTree = retrieveTree(0)
    numLeafs = getNumLeafs(myTree)
    treeDepth = getTreeDepth(myTree)
    print(numLeafs)
    print(treeDepth)


#在父子节点之间填充文本信息
def plotMidText(cntrPt,parentPt,txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid,yMid,txtString)
#计算宽与高
def plotTree(myTree,parentPt,nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    #firstStr = myTree.keys()[0]
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) /2.0/plotTree.totalW,plotTree.yOff)
    plotMidText(cntrPt,parentPt,nodeTxt)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    axprops = dict(xticks=[],yticks=[])
    createPlot.ax1 = plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()

myTree = {'no suffacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
createPlot(myTree)


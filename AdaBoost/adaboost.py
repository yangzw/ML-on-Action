'''
Created on Aug 16, 2013

@author: t-zhyan
'''
from numpy import *

def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = [];lableMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        lableMat.append(float(curLine[-1]))
    return dataMat,lableMat
    
    
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr);labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0;bestStump = {}
    bestClasEst = mat(zeros((m,1)))
    minError = inf
    #goes over all the features in dataset
    for i in range(n):
        rangeMin = dataMatrix[:,i].min();rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax - rangeMin)/numSteps
        #print "dim:",i,"stepSize:",stepSize
        for j in range(-1,int(numSteps)+1):
            #print "range: ",j
            for inequal in ['lt','gt']: #goes over lt and gt
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
#                 print "inequal: ",inequal,"threshVal:",threshVal#,"predictedVals",predictedVals
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
#                 print "errArr:",errArr
                weightedError = D.T * errArr
                if weightedError < minError:
                    #print "i am smaller!"
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
#     print bestStump['dim'],bestStump['thresh'],bestStump['ineq']
#     print "minError",minError
    return bestStump,minError,bestClasEst

# D = mat(ones((5,1))/5)
# dataArr,classLabels = loadsimpData()
# buildStump(dataArr, classLabels, D)D

def adaBoostTrainDS(dataArr,classLables,numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        #get one decision stump
        bestStump,error,classEst = buildStump(dataArr,classLables, D)
        print "D:",D.T
        #begin to calc D(i+1)
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print "classEst: ",classEst.T
        expon = multiply(-1*alpha*mat(classLables).T,classEst)
        D = multiply(D,exp(expon))
#         print "D.sum",D.sum()
        D = D/D.sum()
        aggClassEst += alpha*classEst
#         print "aggClassEst: ",aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLables).T,ones((m,1)))
        errorRate = aggErrors.sum() / m
        print "total error: ",errorRate
        if errorRate == 0.0:break
    return weakClassArr, aggClassEst

# dataArr,classLables = loadSimpData()
# adaBoostTrainDS(dataArr, classLables,9)

def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        #print aggClassEst
    return sign(aggClassEst)
'''
dataArr,classLables = loadSimpData()
classifierArr = adaBoostTrainDS(dataArr, classLables, 30)
print adaClassify([0,0], classifierArr)
'''

# every plot in the figure is decided by the threshold
def plotROC(predStrengths,classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0)
    ySum = 0.0
    numPosClas = sum(array(classLabels) == 1.0)
    yStep = 1 / float(numPosClas)
    xStep = 1 / float(len(classLabels) - numPosClas)
    #argsort() return index, sort predStrengths as threshold to plot ROC
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
    #m =  len(predStrengths.tolist()[0])
    #print m
    #for index in range(m):
        if classLabels[index] == 1.0:
            delX = 0;delY = yStep#one positive node be predicted as negative
        else:
            delX = xStep;delY = 0#one negative node be predicted as negative
            ySum += cur[1]
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY],c='r')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel("false positive rage");plt.ylabel('True positive rate')
    plt.title('ROC curve for adaboost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print "the area under the curve is: ",ySum*xStep
    
datArr,labelArr = loadDataSet('horseColicTraining2.txt')
classifierArray,aggClassEst = adaBoostTrainDS(datArr, labelArr,10)
plotROC(aggClassEst.T,labelArr)
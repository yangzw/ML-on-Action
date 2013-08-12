'''
Created on Aug 9, 2013

@author: t-zhyan
'''
from numpy import *

def loadDataSet():
    dataMat = [];labelMat = []
    fr  = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

#sigmoid
def sigmoid(inX):
    return 1.0 / (1+ exp(-inX) )

#gradient descent
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()#covert row to column
    m,n = shape(dataMatrix)
    alpha = 0.001 #choose learning rate 
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):#repeat maxcycles times
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


#plot the logistic regression best-fit line and dataset
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
            xcord2.append(dataArr[i,1]),ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()
    
'''
dataArr,labelMat = loadDataSet()
weights = gradAscent(dataArr, labelMat)   
plotBestFit(weights.getA())
'''
    
#stochastic gradient ascent, only use one instance at a time to update the weights
def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + [x * alpha * error for x in dataMatrix[i]]
    return weights

'''
weights2 = stocGradAscent0(dataArr, labelMat)
plotBestFit(weights2)
'''

#modified stochastic gradient ascent, randomly shuffle training examples
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            #alpha decreases with iteration, does not go to 0 because of the constant,
            #to reduce the oscillations
            alpha = 4/(1.0+j+i)+0.0001    
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + [x * alpha * error for x in dataMatrix[randIndex]]
            del(dataIndex[randIndex])
    return weights
'''
dataArr,labelMat = loadDataSet()
weights3 = stocGradAscent1(dataArr, labelMat)
plotBestFit(weights3)
'''

def classifyVector(inX,weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0
    
def colicTest():
    frTrain = open('horseColicTraining.txt')
    ftTest = open('horseColicTest.txt')
    trainingSet = [];trainingLabel = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabel.append(float(currLine[21]))
    trainWeights = stocGradAscent1(trainingSet, trainingLabel, 500) # calculate the weights through training data
    #begin to test
    erroCount = 0;numTestVect = 0.0
    for line in ftTest.readlines():
        numTestVect += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(lineArr, trainWeights)) != int(currLine[21]):
            erroCount += 1
    errorRate = (float(erroCount)/numTestVect)
    print "the eroor rate of this test is:%f" % errorRate
    return errorRate

def multiTest():
    numTests = 10;erroSum =  0.0
    for k in range(numTests):
        erroSum += colicTest()
    print "after %d iterations the average error rate is: %f" % (numTests,erroSum/float(numTests))
    
#multiTest()
        
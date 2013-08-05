# -*- coding: utf-8 -*-
'''
Created on Aug 2, 2013
from 《Machine learning on Action》 
@author: t-zhyan
'''
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir

a_path = "C:/Users/t-zhyan/workspace/ML-on-Action/KNN/"

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

'''
group,labels = createDataSet()
print group
print labels
'''

''' 
inX: the input vector to classify
dataSet:full matrix of training examples
labels: the labels of the training examples, already labeled
k
'''
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] 
    diffMat = tile(inX, (dataSetSize,1)) - dataSet #count the diff between inX and training examples
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    #print sqDistances
    distances = sqDistances**0.5             #use matrix to count the distance between inX and every training examples
    sortedDistIndicies = distances.argsort() # get the sorted index saved in sortedDistIndicies    
    classCount={}          
    for i in range(k):                       # vote for the labels
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    '''
    print "-----"
    print sortedClassCount
    print "-----"
    '''
    return sortedClassCount[0][0]


#get matrix and label vector from file
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return   
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector


# normalize
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxvals = dataSet.max(0)
    ranges = maxvals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges, minVals

def datingclassTest(hoRatio):
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)      #use 10% to test, other as training set
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        #print "The classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0
    print "the total error rate is: %f" % (errorCount / float(numTestVecs))
    return errorCount / float(numTestVecs)
 
#check diff error in diff horatio and plot it
def show_diff_horatio():   
    errorarray = []
    for i in range(9):
        errorarray.append(datingclassTest((i+1.0)/10))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(arange(0.0,0.9,0.1),errorarray)
    plt.show()
   
def classifyPerson():
    resultList = ['not at all','in small doses', 'in large doses']
    percentTats = float(raw_input("percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print "You will probably like this person: ",resultList[classifierResult - 1]

        
'''
datingDataMat,datingLabels = file2matrix(a_path + "datingTestSet2.txt")
normMat,ranges, minVals = autoNorm(datingDataMat) 

fig = plt.figure()
ax  = fig.add_subplot(111)
ax.scatter(normMat[:,0],normMat[:,1],15.0*array(datingLabels),15.0*array(datingLabels))
plt.show()

show_diff_horatio()
#classifyPerson()
'''

# convert an 32 * 32 image to 1* 1024 array, so that we could use classify0
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    #====begin to get the trainning set
    hwLabels = []
    trainingFileList = listdir(a_path + 'trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))     #training set with every line contains an 1* 1024 image
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector(a_path + 'trainingDigits/%s' % fileNameStr)
    
    #begin to test    
    testFileList = listdir(a_path + 'testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(a_path + 'testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))
    
handwritingClassTest()
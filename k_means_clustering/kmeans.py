'''
Created on Sep 10, 2013

@author: t-zhyan
'''
from numpy import *

def loadDataSet(fileName):
	dataMat = []
	fr = open(fileName)
	for line in fr.readlines():
		curLine = line.strip().split('\t')
		fltLine = map(float,curLine)
		dataMat.append(fltLine)
	return dataMat

def distEclud(vecA,vecB):
	return sqrt(sum(power(vecA-vecB,2)))

#create cluster centroids randomly
def randCent(dataSet,k):
	n = shape(dataSet)[1]
	centroids = mat(zeros((k,n)))
	for j in range(n):
		minJ = min(dataSet[:,j])
		rangeJ = float(max(dataSet[:,j])-minJ)
		centroids[:,j] = minJ + rangeJ * random.rand(k,1)
	return centroids

def kMeans(dataSet,k,distMeans=distEclud,createCent=randCent):
	m = shape(dataSet)[0]
    #clusterAssment save the index of cluster and the error/distance
	clusterAssment = mat((zeros((m,2))))
	centroids = createCent(dataSet,k)
	clusterChanged = True
	while clusterChanged:
		clusterChanged = False
		for i in range(m):
			minDist = inf;minIndex = -1
			for j in range(k):
				distJI = distMeans(centroids[j,:],dataSet[i,:])
				if distJI < minDist:
					minDist = distJI;minIndex = j
			if clusterAssment[i,0] != minIndex: clusterChanged = True
			clusterAssment[i,:] = minIndex,minDist**2
		print centroids
        #rearrange the centroids
        for cent in range(k):
			ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]
			centroids[cent,:] = mean(ptsInClust,axis = 0)
	return centroids,clusterAssment 

dataMat = mat(loadDataSet('testSet.txt'))
myCentoids,clusterAssing = kMeans(dataMat,4)
print clusterAssing
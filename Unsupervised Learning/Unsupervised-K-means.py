# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 17:38:27 2020

@author: kalla
"""

#Import Modules
import random
from scipy import io
from matplotlib import pyplot as plt
from numpy import linalg as LA,array,zeros,mean,sum as summation
from numpy import argmin,square,average,row_stack
from copy import deepcopy

#load MATLAB file
numpyFile = io.loadmat('AllSamples.mat')
Data = numpyFile['AllSamples']
dataArray = array(Data)
objectiveFunctionValues = [0]*9

'''Helper Functions'''
def distanceBetween(a,b,axis=1):
    '''Compute Vector norms'''
    distance = LA.norm(a-b,axis=axis)
    return distance

def plotObjectiveFunctionStrategy(objectiveFunctionValues,index):
    ''' Plot results'''
    plt.show()                   
    plt.plot(range(2,11,1),objectiveFunctionValues)
    plt.xlabel("K")
    plt.ylabel("Objective Function")
    plt.title("Strategy"+str(index)+ ": Objective Function vs K")
    print("Objective Function Values",objectiveFunctionValues)

def kMeansStrategy1():
    '''loop for k values ranging from 2 to 10'''
    for k in range(2,11,1):
        '''Initialize k centroid values randomly'''
        randomCentroids = random.sample(list(dataArray),k)
        centroids = array(randomCentroids)
        previousCentroids = zeros(centroids.shape)
        clusters = zeros(len(dataArray))
        error = distanceBetween(centroids,previousCentroids,None)
        '''loop until the centroids doesn't change that is until error is greater than 0'''
        while error > 0:
            for sample in range(len(dataArray)):
                distanceToCentroids = distanceBetween(dataArray[sample],centroids)
                clusterNumber = argmin(distanceToCentroids)
                clusters[sample] = clusterNumber
            previousCentroids = deepcopy(centroids)
            
            for i in range(k):
                points=[]
                for j in range(len(dataArray)):
                    if(clusters[j]==i):
                        points.append(dataArray[j])
                centroids[i] = mean(points,axis=0)
            '''calculate the error between the present centroids and the previous centroids'''
            error = distanceBetween(centroids,previousCentroids,None)
        objectiveFunction = 0
        for m in range(len(centroids)):
            points=[]
            for n in range(len(dataArray)):
                if(clusters[n]==m):
                    points.append(dataArray[n])
            '''finding the value of Objective Function'''
            objectiveFunction = objectiveFunction+summation(square(distanceBetween(centroids[m],points)))
            
        objectiveFunctionValues[k-2] = objectiveFunction
        '''plotting the clusters'''
        fig, ax = plt.subplots()
        for i in range(k):
            points=[]
            for j in range(len(dataArray)):
                    if(clusters[j]==i):
                        points.append(dataArray[j])
            points = array(points)
            ax.scatter(points[:,0],points[:,1],cmap='viridis')
            ax.scatter(centroids[:,0],centroids[:,1],c='#000000',marker='o')
            numberOfClustersTitle="Strategy 1 : Clusters="+str(i+1)
            plt.title(numberOfClustersTitle)
    '''plot the objective function value vs k'''
    plotObjectiveFunctionStrategy(objectiveFunctionValues,1)


def kMeansStrategy2():
    '''loop for k values ranging from 2 to 10'''
    for k in range(2,11,1):
        '''Initialize first centroid value randomly'''
        randomFirstCentroid = random.sample(list(dataArray),1)
        centroids = array(randomFirstCentroid)
        '''calculate the remaining centroid values'''
        for value in range(2,k+1):
            xCoordinate=0;yCoordinate=0;maximumDistance=0
            for x in range(len(dataArray)):
                sample=dataArray[x]
                if sample not in centroids:
                    distanceWithCentroid=distanceBetween(sample,centroids)
                    averageDistance=average(distanceWithCentroid)
                    if averageDistance>maximumDistance:
                        maximumDistance=averageDistance
                        xCoordinate=dataArray[x][0];yCoordinate=dataArray[x][1]
            newCentroid=[xCoordinate,yCoordinate]
            '''add the new centroid to the centroids list'''
            centroids= row_stack([centroids,newCentroid])
                        
        
        previousCentroids = zeros(centroids.shape)
        clusters = zeros(len(dataArray))
        error = distanceBetween(centroids,previousCentroids,None)
        '''loop until the centroids doesn't change that is until error is greater than 0'''
        while error > 0:
            for sample in range(len(dataArray)):
                distanceToCentroids = distanceBetween(dataArray[sample],centroids)
                clusterNumber = argmin(distanceToCentroids)
                clusters[sample] = clusterNumber
            previousCentroids = deepcopy(centroids)
            
            for i in range(k):
                points=[]
                for j in range(len(dataArray)):
                    if(clusters[j]==i):
                        points.append(dataArray[j])
                centroids[i] = mean(points,axis=0)
            '''calculate the error between the present centroids and the previous centroids'''
            error = distanceBetween(centroids,previousCentroids,None)
       
        objectiveFunction = 0
        for m in range(len(centroids)):
            points=[]
            for n in range(len(dataArray)):
                if(clusters[n]==m):
                    points.append(dataArray[n])
            '''finding the value of Objective Function'''
            objectiveFunction += summation(square(distanceBetween(centroids[m],points)))
            
        objectiveFunctionValues[k-2] = objectiveFunction
        '''plotting the clusters'''
        fig, ax = plt.subplots()
        for i in range(k):
            points=[]
            for j in range(len(dataArray)):
                    if(clusters[j]==i):
                        points.append(dataArray[j])
            points=array(points)
            ax.scatter(points[:,0],points[:,1],cmap='viridis')
            ax.scatter(centroids[:,0],centroids[:,1],c='#000000',marker='o')
            numberOfClustersTitle="Strategy 2 : Clusters="+str(i+1)
            plt.title(numberOfClustersTitle)
    '''plot the objective function value vs k'''
    plotObjectiveFunctionStrategy(objectiveFunctionValues,2)
            
'''main function'''            
if __name__ == "__main__":
    kMeansStrategy1()
    kMeansStrategy2()
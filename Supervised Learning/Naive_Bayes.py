# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 07:25:12 2020

@author: kalla
"""

import scipy.io
import numpy
import math



#Gaussian distribution

'''def gaussian_probability(x,mean,std):
    return (1/(math.sqrt(2*math.pi)*std))*(math.exp(-((x-mean)**2)/(2*std**2)))'''

def gaussian_probability(arr,cov_mat):
    return (1/(2*math.pi*math.sqrt(numpy.linalg.det(cov_mat))))*(numpy.exp(-(0.5)*numpy.matmul(numpy.matmul(numpy.transpose(arr),numpy.linalg.inv(cov_mat)),arr)))

def naive_bayes():
    #loading data
    Numpyfile= scipy.io.loadmat("E:\SML\mnist_data.mat") 
    
    trX = Numpyfile['trX']
    trY = Numpyfile['trY']
    tsX = Numpyfile['tsX']
    tsY = Numpyfile['tsY']
    digit7_data=[]
    digit8_data=[]
    
    for i in range(len(trY[0])):
        if trY[0][i]==0:
            digit7_data.append(trX[i])
        else:
            digit8_data.append(trX[i])
            
    
    #Finding mean and standard deviation of digit 7 and digit 8
    digit7_data_x1=[]
    digit8_data_x3=[]
    digit7_data_x2=[]
    digit8_data_x4=[]
            
    for i in range(len(digit7_data)):
        digit7_data_x1.append(numpy.mean(digit7_data[i]))    
        
    for i in range(len(digit7_data)):
        digit7_data_x2.append(numpy.std(digit7_data[i]))
        
    for i in range(len(digit8_data)):
        digit8_data_x3.append(numpy.mean(digit8_data[i]))
        
    for i in range(len(digit8_data)):
        digit8_data_x4.append(numpy.std(digit8_data[i]))        
    
     #finding the averages 
    digit7_x1_mean= numpy.mean(digit7_data_x1)
    digit7_x2_mean= numpy.mean(digit7_data_x2)
    digit8_x3_mean= numpy.mean(digit8_data_x3)
    digit8_x4_mean= numpy.mean(digit8_data_x4)
    
    #constructing the array and its corresponding covariance matrix
    arr_7=numpy.array([digit7_data_x1-digit7_x1_mean,digit7_data_x2-digit7_x2_mean])
    arr_8=numpy.array([digit8_data_x3-digit8_x3_mean,digit8_data_x4-digit8_x4_mean])
    cov_7=numpy.cov(arr_7)
    cov_8=numpy.cov(arr_8)
   
    #assuming the features are independent
    cov_7[0][1]=cov_7[1][0]=0
    cov_8[0][1]=cov_8[1][0]=0
    
    
    #for test data
    digit7_data_test=[]
    digit8_data_test=[]
    
    for i in range(len(trY[0])):
        if trY[0][i]==0:
            digit7_data_test.append(trX[i])
        else:
            digit8_data_test.append(trX[i])
    prob_7_data=len(digit7_data_test)/len(trX)*100
    prob_8_data=len(digit8_data_test)/len(trX)*100
      
    
    #finding probabilities
    pred_values=[]    
    a_7=[]
    a_8=[]
    for i in range(len(tsX)):
        x1=numpy.mean(tsX[i])
        x2=numpy.std(tsX[i])
        a_7=[[x1-digit7_x1_mean],[x2-digit7_x2_mean]]
        a_8=[[x1-digit8_x3_mean],[x2-digit8_x4_mean]]
        probability_x1_7=gaussian_probability(a_7,cov_7)
        probability_x1_8=gaussian_probability(a_8,cov_8)  
        probability_7 = prob_7_data * probability_x1_7
        probability_8 = prob_8_data * probability_x1_8
        
        if probability_7>=probability_8:
            pred_values.append(0)
        else:
            pred_values.append(1)
    
    
    #finding the count of values predicted correctly
    count=0
    for i in range(len(pred_values)):
        if pred_values[i]==tsY[0][i]:
            count+=1
     
    #finding the count of values predicted correctly as digit 7 and digit 8       
    count_0=0
    count_1=0
    for i in range(len(pred_values)):
        if pred_values[i]==tsY[0][i] and pred_values[i]==0:
            count_0+=1        
        elif pred_values[i]==tsY[0][i] and pred_values[i]==1:
            count_1+=1  
     
    #finding the total number of 7 and 8 present in testing data    
    total_7=0
    total_8=0       
    for i in range(len(tsY[0])):
        if tsY[0][i]==0:
            total_7+=1
        else:
            total_8+=1        
    
    #Finding the accuracies
    Accuracy_7=(count_0/total_7)*100
    Accuracy_8=(count_1/total_8)*100           
    Accuracy=(count/len(tsY[0]))*100
    
    
    #printing accuracies
    print("Accuracy_7 using Naive Bayes",Accuracy_7) 
    print("Accuracy_8 using Naive Bayes",Accuracy_8)
    print("Total Accuracy using Naive Bayes",Accuracy)
    








































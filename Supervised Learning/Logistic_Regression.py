# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 09:04:18 2020

@author: kalla
"""

import scipy.io
import numpy

#User defined functions
    
def z_value(w,x):
    return w[0]*x[0]+w[1]*x[1]+w[2]*x[2]
    
def sigmoid(z):
    return 1/(1+numpy.exp(-z))

def likelihood(sigmoid,y):
    return y-sigmoid

def ascent(x,y):
    return x[0]*y,x[1]*y,x[2]*y

def logistic_regression():
    #loading data
    Numpyfile= scipy.io.loadmat("E:\SML\mnist_data.mat") 
    
    trX = Numpyfile['trX']
    trY = Numpyfile['trY']
    tsX = Numpyfile['tsX']
    tsY = Numpyfile['tsY']
    
    #Finding mean and standard deviation of digit 7 and digit 8
    digit_data_mean=0
    digit_data_std=0
    LR=[]        
    for i in range(len(trX)):
        digit_data_mean=numpy.mean(trX[i])
        digit_data_std=numpy.std(trX[i])    
        LR.append([1,digit_data_mean,digit_data_std])
        
        
    digit_data_mean_test=0
    digit_data_std_test=0
    LR_test=[]        
    for i in range(len(tsX)):
        digit_data_mean_test=numpy.mean(tsX[i])
        digit_data_std_test=numpy.std(tsX[i])    
        LR_test.append([1,digit_data_mean_test,digit_data_std_test])
    
    #finding the optimized weights
    w=[0,0,0]
    for j in range(12000):
        updated_w0=0
        updated_w1=0
        updated_w2=0
        for i in range(len(trX)):
            w0,w1,w2=ascent(LR[i],likelihood(sigmoid(z_value(w,LR[i])),trY[0][i]))
           
            updated_w0=updated_w0+w0
            updated_w1=updated_w1+w1
            updated_w2=updated_w2+w2       
        
        learning_rate=0.0005 
        
        w[0]=w[0]+learning_rate*updated_w0
        w[1]=w[1]+learning_rate*updated_w1
        w[2]=w[2]+learning_rate*updated_w2
    
    
    pred_values=[]
    
    for i in range(len(LR_test)):
        if sigmoid(z_value(w,LR_test[i]))>=0.5:
            pred_values.append(1)
        else:
            pred_values.append(0)
    
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
    
    #printing the accuracies
    print("Accuracy_7 using Logistic Regression",Accuracy_7) 
    print("Accuracy_8 using Logistic Regression",Accuracy_8)
    print("Total Accuracy using Logistic Regression",Accuracy)

















#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Fri Oct 05 09:27:32 2018

@author: Varun_Garg
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = np.array([['','Col1','Col2'],['Row1',1,2],['Row2',3,4]])

dataframe_f = (pd.DataFrame(data=data[1:,1:],index=data[1:,0],columns=data[0,1:]))



dataframe_f = (pd.DataFrame(data=data[1:,1:],index=[5,6],columns=data[0,1:]))
#print dataframe_f


dataframe_i = pd.DataFrame(data=[4,5,6,7],index=[0,1,2,9990],columns =['one row'])
#print dataframe_i

read_csv= pd.read_csv('iris_data.csv',delimiter=',')
read_csv_full= pd.read_csv('iris.csv',delimiter=',')

read_csv_labels= pd.read_csv('iris_labels.csv',delimiter=',')
read_csv_label_array = pd.read_csv('labels_arrays.csv',delimiter=',')

print "shape of data"
#print read_csv
np.shape(read_csv)
index_iris = range(1, 150)

dataframe_iris= pd.DataFrame(read_csv,index=index_iris)

setosa_m=[1,0,0]
versicolor_m=[0,1,0]
virginica_m=[0,0,1]


dataframe_full_iris= pd.DataFrame(read_csv_full,index=index_iris)

data_values_transpose = np.array(np.transpose(dataframe_iris.values))
data_values           = np.array(dataframe_iris.values)
#print data_values_transpose

setosa_d    =dataframe_iris[0:50]
versicolor_d=dataframe_iris[50:100]
virginica_d =dataframe_iris[100:150]

############histogram of features petal legnth and petal_width

setosa_d['petal_length'].plot.hist(bins=10,color='red')
setosa_d['petal_width'].plot.hist(bins=10,color='red')
versicolor_d['petal_length'].plot.hist(bins=10,color='blue')
versicolor_d['petal_width'].plot.hist(bins=10,color='blue')
virginica_d['petal_length'].plot.hist(bins=10,color='green')
virginica_d['petal_width'].plot.hist(bins=10,color='green')
plt.title("histogram of features petal legnth and petal_width")


plt.figure()
############histogram of features sepal legnth and sepal_width

setosa_d['sepal_length'].plot.hist(bins=10,color='red')
setosa_d['sepal_width'].plot.hist(bins=10,color='red')
versicolor_d['sepal_length'].plot.hist(bins=10,color='blue')
versicolor_d['sepal_width'].plot.hist(bins=10,color='blue')
virginica_d['sepal_length'].plot.hist(bins=10,color='green')
virginica_d['sepal_width'].plot.hist(bins=10,color='green')
plt.title("histogram of features sepal legnth and sepal_width")

plt.figure()
############histogram of features petal legnth and sepal_width

setosa_d['petal_length'].plot.hist(bins=10,color='red')
setosa_d['sepal_width'].plot.hist(bins=10,color='red')
versicolor_d['petal_length'].plot.hist(bins=10,color='blue')
versicolor_d['sepal_width'].plot.hist(bins=10,color='blue')
virginica_d['petal_length'].plot.hist(bins=10,color='green')
virginica_d['sepal_width'].plot.hist(bins=10,color='green')
plt.title("histogram of features petal legnth and sepal_width")

plt.figure()
############histogram of features petal width and  sepal length

setosa_d['petal_width'].plot.hist(bins=10,color='red')
setosa_d['sepal_length'].plot.hist(bins=10,color='red')
versicolor_d['petal_width'].plot.hist(bins=10,color='blue')
versicolor_d['sepal_length'].plot.hist(bins=10,color='blue')
virginica_d['petal_width'].plot.hist(bins=10,color='green')
virginica_d['sepal_length'].plot.hist(bins=10,color='green')

plt.title("histogram of features petal width and  sepal length")



#################calculation of the training model 

first_term= np.matmul(data_values_transpose,data_values)

label_array =np.array(read_csv_label_array)

second_term =np.matmul(data_values_transpose,label_array)

training_model=np.linalg.solve(first_term,second_term)

test_case =np.matmul(np.transpose(training_model) ,np.transpose(data_values[0]))
print test_case




#################testing the model
test_results = np.zeros((3,150))

test_results=np.matmul(np.transpose(training_model) ,np.transpose(data_values))
test_results_t= test_results.T


# function which provides the idex of the greatest value of the array
    
list1 =[-0.17494675, 0.321072,0.85387475]
max_v = max(list1)
print max_v
print list1.index(max(list1))

    


    

### for loop to compare if the max value of test_results and t matrix 



counter_mis=0

for tt in range (0,149): 
    x=test_results_t[tt]   
    y=label_array[tt]  
    if(np.argmax(x) != np.argmax(y)):
        counter_mis=counter_mis+1
    
print "total number of miss classifications from training all of the features in the iris dataset using least square method is"  
print counter_mis


############################training with only two features


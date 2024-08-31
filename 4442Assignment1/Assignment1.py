#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS 4442 - Assignment 1
@author: Shaan Verma
Student #: 250804514
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

'''
Helper functions to be used in the following questions
'''
def weight(x,y):
    # Calculating parts of closed form linear regression weight
    first = np.dot(x.T, x)
    second = np.dot(x.T,y)
    return np.dot(np.linalg.pinv(first),second)

def predict(x,w):
    return np.dot(x,w)

# Average error calculator function to be used in questions
def averageError(xCol,yTrain,weight):
    err = 0
    
    for i in np.arange(0,len(yTrain),1):
        err = err + (np.dot(weight.T,xCol[i])-yTrain[i])**2
        
    return err/len(yTrain)

def cross_valid_helper_1(XTrain,YTrain, lamda, number):
    I_hat = np.eye(number + 1)
    I_hat[0, 0] = 0
    holder1 = np.linalg.inv(np.dot(XTrain.T, XTrain) + (lamda * I_hat))
    holder2 = np.dot(XTrain.T, YTrain)
    weight = np.dot(holder1, holder2)
    return weight

def cross_valid_helper_2(train, train_data, train_label, test, test_data, test_label,
               lam=0, num=1):
    w = cross_valid_helper_1(train_data, train_label, lam, num)
    trainPred = np.dot(train_data, w.T)
    testPred = np.dot(test_data, w.T)
    
    return trainPred, testPred, w

def compute_err(y_pred, label):
    m = len(y_pred)
    err = (1 / m) * np.sum(np.square(np.subtract(y_pred, label)))
    return err



'''
QUESTION 2A
'''
# Reading data from files and placing into numpy arrays
xtr = np.loadtxt('hw1xtr.dat')
ytr = np.loadtxt('hw1ytr.dat') 
xte = np.loadtxt('hw1xte.dat')
yte = np.loadtxt('hw1yte.dat')


# Plotting xtr and ytr data
plt.scatter(xtr,ytr)
plt.xlabel("xtr")
plt.ylabel("ytr")
plt.title("Scatter Plot of xtr and ytr")

# Plotting xte and yte data
plt.figure()
plt.scatter(xte, yte)
plt.title("Graph of xte and yte")
plt.xlabel("xte")
plt.ylabel("yte")


'''
QUESTION 2B - Linear regression
'''
# Creating column of ones
ones_x = np.ones((1,len(xtr)))
ones_y = np.ones((1,len(ytr)))
ones_xte = np.ones((1,len(xte)))
ones_yte = np.ones((1,len(yte)))

#Making xtr into column vector
xtr = xtr.reshape(-1,1)
xte = xte.reshape(-1,1)
ones_x = ones_x.reshape(-1,1)
ones_xte = ones_xte.reshape(-1,1)
x_col = np.hstack((xtr,ones_x))
x_test = np.hstack((xte,ones_xte))

# Making ytr into column vector
y_col = ytr.reshape(-1,1)

#Calculating weight
w1 = weight(x_col,y_col)

# Predicting linear line
p1 =predict(x_col,w1)

dTrain1 = np.linspace(min(xtr),max(xtr),100)

# Plotting xtr and ytr data with linear regression line
plt.figure()
plt.scatter(xtr,ytr)
plt.xlabel("xtr")
plt.ylabel("ytr")
plt.title("Scatter Plot of xtr and ytr with Linear Regression Line")
plt.plot(dTrain1, w1[0].item()*dTrain1 + w1[1].item(), color='red')

# Average error for Linear regression
print("----------------- Linear Regression -----------------")
print("Average Error (Training): ",averageError(x_col, ytr, w1).item())


'''
QUESTION 2C - Linear regression
'''
# Plotting xte and yte data with linear regression line
dTest1 = np.linspace(min(xte),max(xte),100)

plt.figure()
plt.scatter(xte,yte)
plt.xlabel("xte")
plt.ylabel("yte")
plt.title("Scatter Plot of xte and yte with Linear Regression Line")
plt.plot(dTest1, w1[0].item()*dTest1 + w1[1].item(), color='red')

# Average error for Linear Regression test set
print("Average Error (Test): ",averageError(x_test, yte, w1).item())



'''
QUESTION 2D - 2nd order polynomial regression
'''

# Calculating x^2, x^3, x^4 values for training data
squared = np.delete(np.power(x_col,2),1,1)
cubed = np.delete(np.power(x_col,3),1,1)
four = np.delete(np.power(x_col,4),1,1)

# Calculating x^2, x^3, x^4 values for test data 
squaredTest = np.delete(np.power(x_test,2),1,1)
cubedTest = np.delete(np.power(x_test,3),1,1)
fourTest = np.delete(np.power(x_test,4),1,1)

#Appending x^2, x^3, x^4 to respective matrix x values for training data
x_col_squared = np.c_[squared,x_col]
x_col_cubed = np.c_[cubed,x_col_squared]
x_col_four =np.c_[four,x_col_cubed]

#Appending x^2, x^3, x^4 to respective matrix x values for test data
x_col_squared_test = np.c_[squaredTest,x_test]
x_col_cubed_test = np.c_[cubedTest,x_col_squared_test]
x_col_four_test =np.c_[fourTest,x_col_cubed_test]

w2 = weight(x_col_squared, y_col)
p2 = predict(x_col_squared, w2)

dTrain2 = np.linspace(min(xtr), max(xtr), 100)

# Plotting xtr and ytr data with 2nd order regression line
plt.figure()
plt.scatter(xtr,ytr)
plt.xlabel("xtr")
plt.ylabel("ytr")
plt.title("Scatter Plot of xtr and ytr with 2nd Order Polynomial Regression Line")
plt.plot(dTrain2, w2[0].item()*(dTrain2**2) + w2[1].item()*dTrain2 + w2[2].item() ,color='red')

# Average error for Linear Regression training set
print("\n----------------- 2nd Order Polynmial Regression -----------------")
print("Average Error (Training data): ",averageError(x_col_squared, ytr, w2).item())


# Plotting xte and yte data with 2nd order regression line
dTest2 = np.linspace(min(xte), max(xte), 100)

plt.figure()
plt.scatter(xte,yte)
plt.xlabel("xte")
plt.ylabel("yte")
plt.title("Scatter Plot of xte and yte with 2nd Order Polynomial Regression Line")
plt.plot(dTest2, w2[0].item()*(dTest2**2) + w2[1].item()*dTest2 + w2[2].item() ,color='red')

# Average error for Linear Regression test set
print("Average Error (Test data): ",averageError(x_col_squared_test, yte, w2).item())



'''
QUESTION 2E - 3rd order polynomial regression
'''
w3 = weight(x_col_cubed, y_col)
p3 = predict(x_col_cubed, w3)

dTrain3 = np.linspace(min(xtr), max(xtr), 100)

# Plotting xtr and ytr data with 3rd order regression line
plt.figure()
plt.scatter(xtr,ytr)
plt.xlabel("xtr")
plt.ylabel("ytr")
plt.title("Scatter Plot of xtr and ytr with 3rd Order Polynomial Regression Line")
plt.plot(dTrain3, w3[0].item()*(dTrain3**3) + w3[1].item()*(dTrain3**2) + w3[2].item()*(dTrain3) + w3[3].item() ,color='red')

# Average error for Linear Regression training set
print("\n----------------- 3rd Order Polynomial Regression -----------------")
print("Average Error (Training data): ",averageError(x_col_cubed, ytr, w3).item())

dTest3 = np.linspace(min(xte), max(xte), 100)

# Plotting xte and yte data with 3rd order regression line
plt.figure()
plt.scatter(xte,yte)
plt.xlabel("xte")
plt.ylabel("yte")
plt.title("Scatter Plot of xte and yte with 3rd Order Polynomial Regression Line")
plt.plot(dTest3, w3[0].item()*(dTest3**3) + w3[1].item()*(dTest3**2) + w3[2].item()*(dTest3) + w3[3].item() ,color='red')

# Average error for Linear Regression training set
print("Average Error (Test data): ",averageError(x_col_cubed_test, yte, w3).item())



'''
QUESTION 2F - 4th order polynomial regression
'''
w4 = weight(x_col_four, y_col)
p4 = predict(x_col_four, w4)

dTrain4 = np.linspace(min(xtr), max(xtr), 100)

# Plotting xtr and ytr data with 3rd order regression line
plt.figure()
plt.scatter(xtr,ytr)
plt.xlabel("xtr")
plt.ylabel("ytr")
plt.title("Scatter Plot of xtr and ytr with 4th Order Polynomial Regression Line")
plt.plot(dTrain4, w4[0].item()*(dTrain4**4) + w4[1].item()*(dTrain4**3) + w4[2].item()*(dTrain4**2) + w4[3].item()*(dTrain4) + w4[4].item() ,color='red')

# Average error for Linear Regression training set
print("\n----------------- 4th Order Polynomial Regression -----------------")

print("Average Error (Training data): ",averageError(x_col_four, ytr, w4).item())


dTest4 = np.linspace(min(xte), max(xte), 100)

# Plotting xte and yte data with 4th order regression line
plt.figure()
plt.scatter(xte,yte)
plt.xlabel("xte")
plt.ylabel("yte")
plt.title("Scatter Plot of xte and yte with 4th Order Polynomial Regression Line")
plt.plot(dTest4, w4[0].item()*(dTest4**4) + w4[1].item()*(dTest4**3) + w4[2].item()*(dTest4**2) + w4[3].item()*(dTest4) + w4[4].item() ,color='red')

# Average error for Linear Regression training set
print("Average Error (Test data): ",averageError(x_col_four_test, yte, w4).item())



'''
QUESTION 3 - Regularization and Cross-Validation
'''

# Setting I hat matrix
identity = np.identity(5)
identity[0,0] = 0

# Declaring lambda values
Lam = np.array([0.01,0.1,1,10,100,1000,10000])
LamLog = np.log10(Lam)
LamWeights = []

# Solution for L2-Norm
for i in Lam:
    w_L2 = np.linalg.pinv(np.dot(x_col_four.T, x_col_four) + i*identity)
    w_L2 = np.dot(w_L2,x_col_four.T)
    w_L2 = np.dot(w_L2,ytr)
    LamWeights.append(w_L2)

# Holders for training and test error
LamTrainError = []
LamTestError = []

# Calculating and populating error arrays
for j in LamWeights:
    LamTrainError.append(averageError(x_col_four, ytr, j))

for k in LamWeights:
    LamTestError.append(averageError(x_col_four_test, yte, k))

# Graphing training and test error as a function of lambda
plt.figure()
plt.plot(LamLog,LamTrainError, marker='o', label="Training")
plt.plot(LamLog, LamTestError, marker='o', label="Test")
plt.title("Training and Test Error as a Function of Lambda")
plt.ylabel("Error")
plt.xlabel("Lambda (log10)")
plt.legend()

# Holder for weight values
lw0 = []
lw1 = []
lw2 = []
lw3 = []
lw4 = []

# Populating weight arrays
for l in LamWeights:
    lw4.append(l[0])
    lw3.append(l[1])
    lw2.append(l[2])
    lw1.append(l[3])
    lw0.append(l[4])

# Graphing Wights as a function of lambda
plt.figure()
plt.plot(LamLog,lw0, label='w0')
plt.plot(LamLog,lw1, label='w1')
plt.plot(LamLog,lw2, label='w2')
plt.plot(LamLog,lw3, label='w3')
plt.plot(LamLog,lw4, label='w4')
plt.title("Weight Parameters as a Function of Lambda")
plt.ylabel("Weight")
plt.xlabel("Lambda (log10)")
plt.legend()
plt.show()

# Cross-validation procedure
def crossValidation(xTrainData, xColumnTrainData, yTrainData, xTestData):
    print("Cross Validation")
    lambd = np.array([0.01, 0.1, 1, 10, 100, 1000, 10000])

    for u in lambd:
        trainError = []
        valError = []
        kf = KFold(n_splits=5)
        for train_index, test_index in kf.split(xColumnTrainData):
            X_train, X_val = xColumnTrainData[train_index], xColumnTrainData[test_index]
            y_train, y_val = np.asarray(yTrainData)[train_index], np.asarray(yTrainData)[test_index]
            train_pred, val_pred, w_cross = cross_valid_helper_2(xTrainData, X_train, y_train,xTestData, X_val, y_val,u, num=4)
            
          
            trainError.append(compute_err(train_pred, y_train))
            valError.append(compute_err(val_pred, y_val))
        
        print("Lambda {} train error: {} validation error: {}".format(u, np.mean(
            trainError), np.mean(valError)))

crossValidation(xtr, x_col_four, ytr, xte)


# Plotting xte and yte data with lambda regression line
plt.figure()
plt.scatter(xte,yte)
plt.xlabel("xte")
plt.ylabel("yte")
plt.title("Scatter Plot of xte and yte with Best Lambda Value")
plt.plot(dTest4, LamWeights[0][0].item()*(dTest4**4) + LamWeights[0][1].item()*(dTest4**3) + LamWeights[0][2].item()*(dTest4**2) + LamWeights[0][3].item()*(dTest4) + LamWeights[0][4].item() ,color='red')



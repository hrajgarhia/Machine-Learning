import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path='/Users/harshitrajgarhia/PycharmProjects/Machine-Learning/machine_learning_examples-master/linear_regression_class/'

# load the data
X=[]
Y=[]

for line in open(path+'data_1d.csv'):
    x,y = line.split(',')
    X.append(float(x))
    Y.append(float(y))

# converting lists to numpy array since they are more useful

X=np.array(X)
Y=np.array(Y)

#PLOT the data

plt.scatter(X,Y)
plt.show()

#calculate a and b from the formula derived

denominator= X.dot(X)-X.mean()*X.sum()
a= (X.dot(Y) -X.mean()*Y.sum())/denominator
b=(Y.mean()*(X.dot(X))-X.mean()*X.dot(Y))/denominator

yhat = a*X + b

#plot the line of best fit

plt.scatter(X,Y)
plt.plot(X,yhat)
plt.show()

#calculate r-squared error

d1=Y-yhat
d2=Y-Y.mean()

rsq= 1-(d1.dot(d1))/(d2.dot(d2))

print("R-squared is :", rsq)
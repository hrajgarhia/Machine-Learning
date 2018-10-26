import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



path='/Users/harshitrajgarhia/PycharmProjects/Machine-Learning/machine_learning_examples-master/linear_regression_class/'

# load the data
X=[]
Y=[]

for line in open(path+'data_2d.csv'):
    x1,x2,y = line.split(',')
    X.append([float(x1),float(x2),1])
    Y.append(float(y))


# converting lists to numpy array since they are more useful

X=np.array(X)
Y=np.array(Y)

# PLOT to see the data in 3d
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(X[:,0],X[:,1],Y)


#calculate weights

w=np.linalg.solve(np.dot(X.T,X),np.dot(X.T,Y))
Yhat=np.dot(X,w)



#calculate r-squared error

d1=Y-Yhat
d2=Y-Y.mean()

rsq= 1-(d1.dot(d1))/(d2.dot(d2))

print("R-squared is :", rsq)

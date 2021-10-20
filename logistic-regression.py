# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 23:23:29 2021

@author: Stephen
"""

import numpy as np
import matplotlib.pyplot as plt

#define the logistic regression class
class LogisticRegression(object):
    def __init__(self, eta = 0.01, n_iter = 100):
        self.eta = eta # gradient descent scalar
        self.n_iter = n_iter # gradient descent iterations
    
    def fit(self, X, y):
        #gradient descent algorithm
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)
        m = X.shape[0]
        n_features = X.shape[1]
        self.w_ = np.zeros((n_features,1))
        self.cost_hist_ = np.zeros(self.n_iter)
        for i in range(self.n_iter):
            error = self.activation(X) - y
            self.w_ -= self.eta * X.T.dot(error)/m
            self.cost_hist_[i] = (self.calc_cost(X, y))
        return self
    
    def net_input(self, X):
        #Compute Xw
        return X.dot(self.w_)
    
    def activation(self, X):
        #Apply sigmoidal functional to net_input
        return 1/(1+np.exp(-self.net_input(X)))
    
    def predict(self, X):
        #Predict class label.
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)
        return np.where(self.activation(X) > 0.5, 1, 0) 
    
    def calc_cost(self, X, y):
        #Calculate cost function value
        m = X.shape[0]
        probs = self.activation(X)
        return -y.T.dot(np.log(probs)) - (1-y).T.dot(np.log(1-probs))/m

#generate data     
num_points = 1000 # points per blob
blob1 = np.random.normal(2, 1, (num_points, 2))
blob2 = np.random.normal(5, 1, (num_points, 2))
X = np.concatenate((blob1, blob2), axis = 0)
y = np.concatenate((np.zeros(num_points), np.ones(num_points))).reshape(-1,1)

shuffle = np.random.permutation(range(num_points*2))
end_point = int(0.7*num_points*2)
trainX, trainY = X[shuffle[0:end_point,]], y[shuffle[0:end_point]]
testX, testY = X[shuffle[end_point:,]], y[shuffle[end_point:]]

#fit a model for different etas
etas = [0.01, 0.03, 0.1, 0.3]
for i in range(len(etas)):  
    eta = etas[i]
    lr = LogisticRegression(eta =  eta, n_iter = 1000)
    lr.fit(trainX, trainY)
    plt.subplot(2,2,i+1)
    plt.plot(lr.cost_hist_)
    plt.title(f"eta = {eta} Cost Function History")
plt.tight_layout()
plt.show()

#go with eta = 0.1
lr = LogisticRegression(eta =  0.1, n_iter = 5000)
lr.fit(trainX, trainY)
plt.plot(lr.cost_hist_)
plt.title(f"eta = {eta} Cost Function History")
w = lr.w_

#plot data along with decision boundary
x_line = np.linspace(trainX[:,0].min() -1, trainX[:,1].max() +1, 1000)
y_line = (-x_line * w[1] -w[0])/w[2]
plt.figure()
plt.scatter(trainX[:,0], trainX[:,1], c = np.ravel(trainY))
plt.plot(x_line, y_line, '-r')
plt.show()

#output test accuracy
train_acc = (lr.predict(trainX) == trainY).mean()
test_acc = (lr.predict(testX) == testY).mean()
print(f"train accuracy = {round(train_acc*100,2)}.")
print(f"test accuracy = {round(test_acc*100,2)}.")



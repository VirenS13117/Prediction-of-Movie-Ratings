import pandas as pd
import numpy as np
import random
import math
from sklearn.manifold import TSNE
from sklearn.cross_validation import KFold
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import BaggingRegressor
from scipy.misc import comb
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore')


def train_test(data):
	cols = len(data[0,:])
	rows = len(data[:,0])
	X = np.ones(shape=(rows,cols))
	Y = np.zeros(shape=(rows,1))
	Y[:,0] = data[:,cols-1]
	for i in range(0,cols-1):
		X[:,i+1] = data[:,i]
	return X,Y


def multilinear_Regression(data,X,Y,max_itr,alpha,delta):
	delta_sqr = pow(delta,2)
	theta = np.zeros(shape=(len(X[0]),1))
	for j in range(max_itr):
		for i in range(0,len(X[0])):
			hypothesis = np.dot(X,theta)
			loss = Y - hypothesis
			loss = loss[:,0]
			temp = np.zeros(shape=(len(X[0]),1))
			temp[i][0] = np.dot(loss,X[:,i]) + 2*delta_sqr*theta[i][0]
			
			theta[i][0] = theta[i][0] + alpha*(1.0/len(X))*temp[i][0]
	return theta


def random_forest_regression(trainProduct,testProduct,features_to_use,target):
	X = trainProduct[features_to_use]
	y = trainProduct[target]
	testIn = testProduct[features_to_use]
	testOut = testProduct[target]

	print "target values"
	print y.values.ravel()

	rf_model = RandomForestRegressor(n_estimators=150, max_features=6,    # Num features considered
	                                  oob_score=True, n_jobs = -1, random_state = 1) 

	rf_model.fit(X, y.values.ravel())

	test_preds = rf_model.predict(testIn)

	for i in range(100):
	    print testOut.values[i]," ",test_preds[i]

	return test_preds

def ridge_regression(trainProduct, testProduct):
	trainX, trainY = train_test(np.array(trainProduct))
	testX, testY = train_test(np.array(testProduct))
	
	theta = multilinear_Regression(trainProduct, trainX, trainY, 1500, 0.001, 0.1)
	hypothesis = np.dot(testX,theta)
	return hypothesis



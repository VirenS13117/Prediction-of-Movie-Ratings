import pandas as pd
import numpy as np
import random
import math
from sklearn.manifold import TSNE
from sklearn.cross_validation import KFold
from sklearn import metrics
from sklearn import ensemble
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import BaggingRegressor
from scipy.misc import comb
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets, linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#import seaborn as sns
#get_ipython().magic(u'matplotlib inline')
#sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings('ignore')


def loss(predictVal,ExpectVal, length):
	squared_loss=0;
	loss = (ExpectVal - predictVal)**2

	squared_loss = np.sum(loss)
	print squared_loss/(2*length)


def stacking_using_linear(ridgeOut, forrestOut, testOut):
	prediction = pd.DataFrame({'forestPred': forrestOut})#,'ridgePred': ridgeOut })
	prediction['ridgePred']=ridgeOut
	prediction['expected']=testOut
	stacker= linear_model.LinearRegression()
	stacker.fit(prediction[['forestPred', 'ridgePred']], prediction['expected'])
	print stacker.coef_
	st=stacker.predict(prediction[['forestPred', 'ridgePred']])
	st = st.tolist()
	testPred=pd.DataFrame(st, index=None, columns=list('0'));
	#print testPred
	print "MSE value is:", mean_squared_error(prediction['expected'],testPred) #mean_squared_error(testOut,testPred)
	print "MAE value is:",mean_absolute_error(prediction['expected'],testPred);
	return testPred


def stacking_using_averaging(ridgeOut, forrestOut, testOut):
	prediction = pd.DataFrame({'forestPred': forrestOut})#,'ridgePred': ridgeOut })
	prediction['ridgePred']=ridgeOut
	average = prediction[['forestPred', 'ridgePred']].mean(axis=1)
	prediction['average'] = average
	#print average;
	print "MSE value is:", mean_squared_error(testOut, average)
	print "MAE value is:", mean_absolute_error( testOut, average);
	return average

def gradientBoosting(trainX, trainY,testX, testY):
	est = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1,max_depth=1, random_state=0, loss='ls').fit(trainX, trainY)
	boostPred = est.predict(testX)
	print "MSE value is:", mean_squared_error(testY, est.predict(testX)) 
	print "MAE value is:", mean_absolute_error(testY, est.predict(testX))
	return boostPred

def adaBoosting(trainX, trainY, testX, testY):
	tree=DecisionTreeRegressor(max_depth=5)
	model1=tree.fit(trainX, trainY)
	adaBoost = AdaBoostRegressor(tree,n_estimators=100, random_state=1)
	adaBoost.fit(trainX, trainY)
	adaPred = adaBoost.predict(testX)
	print "MSE value is: ", mean_squared_error(testY, adaPred)
	print "MAE value is: ", mean_absolute_error(testY, adaPred);
	return adaPred


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

#get_ipython().magic(u'matplotlib inline')

import warnings
warnings.filterwarnings('ignore')

import pre_process
import regression_for_ratings
import ensemble



data_frame1 = pd.DataFrame({0 : [np.nan]})
train_set = data_frame1.copy()
test_set = data_frame1.copy()

features_to_use = ['UserID','Gender','Age','Occupation','Zip-code','MovieID','Genres']
target = ['Rating']

def show_users():
	user1 = pd.read_csv("users.dat",names = ["UserID", "Gender", "Age", "Occupation", "Zip-code"], sep="::|")
	print "................................................................................................."
	print "Users Data"
	print user1

def show_movies():
	movies1 = movies1 = pd.read_csv("movies.dat",names = ["MovieID","Title","Genres"],sep="::|")
	print "................................................................................................."
	print "Movies Data"
	print movies1

def show_ratings():
	ratings1 = pd.read_csv("ratings.dat",names=["UserID", "MovieID","Rating","Timestamp"],sep="::|")
	print "................................................................................................."
	print "Ratings Data"
	print ratings1

def show_pre_process():
	print "Preprocessing the File"
	print "................................................................................................."
	print "Preprocessed Data"
	print data_frame1
	print data_frame1[features_to_use]

def show_correlation():
	co_relationFile = data_frame1.corr(method = 'pearson', min_periods=1)
	co_relationFile.to_csv('co_relationFile.csv')
	new_correlation = pd.read_csv("co_relationFile.csv",header=None)
	print
	print "Correlation between all the columns of data"
	print new_correlation
	print "looking at Correlation values we can say important features_to_use are"
	print features_to_use
	print "target column ", target


def create_sets():
	print "Dataset"
	print "0. movieLens 0.1M : for quick results"
	print "1. movieLens 1M : may take long time"
	print 
	print "Choose Dataset"
	print "70% of the top rows will be added as train and remaining 30% will be added as test"
	choice = input()
	while(choice !=0 and choice != 1):
		print "please enter correct option"
		choice = input()

	train_set, test_set = pre_process.data_set_train_test(data_frame1,choice)
	print "Training Set."
	print train_set
	print ".................................................................."
	print "Test Set"
	print test_set
	return train_set, test_set


def errors(Y_true,Y_predicted, log_true, log_preds):
	print "Mean Squared Error : ", mean_squared_error(Y_true, Y_predicted)
	print "Root Mean Square Error : ", mean_squared_error(Y_true, Y_predicted)**0.5
	print "Mean Absolute Error : ", mean_absolute_error(Y_true, Y_predicted)
	print "Root Mean Square Log Error : ", mean_squared_error(log_true, log_preds)**0.5

 


def random_forest():
	print "Random Forest Regression"
	train_set, test_set = create_sets()
	predictions = regression_for_ratings.random_forest_regression(train_set,test_set,features_to_use,target)
	log_preds = []
	log_true = []
	testOut = test_set[target]
	for i in range(len(testOut)):
	    log_true.append(math.log(testOut.values[i]+1))
	    log_preds.append(math.log(predictions[i]+1))

	errors(test_set[target],predictions,log_true,log_preds)
	plt.scatter(data[0], data[1]);
	plt.scatter(data[0], predicted_values, color='red');
	plt.show()

def ridge():
	print "Ridge Regression"
	train_set, test_set = create_sets()
	predictions = regression_for_ratings.ridge_regression(train_set,test_set)
	log_preds = []
	log_true = []
	testOut = test_set[target]
	for i in range(len(testOut)):
	    log_true.append(math.log(testOut.values[i]+1))
	    log_preds.append(math.log(predictions[i]+1))

	errors(test_set[target],predictions,log_true,log_preds)

def stacking_linear():
    print "Stacking"
    train_set, test_set = create_sets()
    forrestOut = regression_for_ratings.random_forest_regression(train_set,test_set,features_to_use,target)
    ridgeOut = regression_for_ratings.ridge_regression(train_set,test_set)
    pred = ensemble.stacking_using_linear(ridgeOut, forrestOut, test_set[target])

def stacking_averaging():
    print "Stacking using averaging"
    train_set, test_set = create_sets()
    forrestOut = regression_for_ratings.random_forest_regression(train_set,test_set,features_to_use,target)
    ridgeOut = regression_for_ratings.ridge_regression(train_set,test_set)
    pred = ensemble.stacking_using_averaging(ridgeOut, forrestOut, test_set[target])

def adaBoost():
    print "AdaBoosting"
    train_set, test_set = create_sets()
    pred=ensemble.adaBoosting(train_set[features_to_use], train_set[target], test_set[features_to_use], test_set[target])

def gradBoost():
    print "Gradient Boosting"
    train_set, test_set = create_sets()
    pred = ensemble.gradientBoosting(train_set[features_to_use], train_set[target], test_set[features_to_use], test_set[target])



options = {0 : show_users,
		  1 : show_movies,
		  2 : show_ratings,
		  3 : show_pre_process,
		  4 : show_correlation,
		  5 : create_sets,
		  6 : random_forest,
		  7 : ridge,
		  8 : stacking_linear,
          9 : stacking_averaging,
          10 : adaBoost,
          11 : gradBoost,

}

print "Movie Lens Dataset"
print "Operations"
print "0. View Users Data"
print "1. View Movies Data"
print "2. View Ratings Data"
print "3. View Preprocessed Data"
print "4. View Correlation between different columns"
print "5. Create Train and Test Dataset"
print "6. Random Forest Regression for Movies"
print "7. Ridge Regression for Movie Ratings"
print "--------------------------------------"
print " Ensembling Techniques"
print "8. Linear Stacking For Movie Ratings"
print "9. Average Stacking for Movie Ratings"
print "10. AdaBoosting for Movie Ratings"
print "11. Gradient Boosting For Movie Ratings"




print "enter your choice of Operations"
choice = input()
if(choice >= 3):
	data_frame1 = pre_process.process()
	

	
options[choice]()


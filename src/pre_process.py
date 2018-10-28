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

#Label Encoding Function

def convert2(data):
	number = preprocessing.LabelEncoder()
	data = number.fit_transform(data)
	
	return data

# Normalization Function

def normalize(data):
    filter_df = data.copy()
    for i in filter_df.columns:
        if filter_df[i].dtype == 'int64':
            filter_df[i] = np.float64(filter_df[i])
    std_scale = preprocessing.MinMaxScaler().fit(data)
    df_std = std_scale.transform(data)
    k = 0
    print df_std
    for i in filter_df.columns:
        for j in range(len(data)):
            filter_df[i].values[j] = df_std[j][k]
        k += 1
    return filter_df	

def getFinalCount(ageLow,ageHigh,genre,test1):
    count=0;
    for i in range(len(test1)):
        if(test1['Age'][i]>=ageLow and test1['Age'][i]<ageHigh and test1['Genres'][i]==genre):
            count=count+1;
            #print test1['Age'][i];
    print count
    return count


def chi_square(test1):
	
	max_age=max(test1['Age'])
	print max_age
	min_age=min(test1['Age'])
	print min_age

	min_genre=min(test1['Genres'])
	max_genre=max(test1['Genres'])
	print min_genre
	print max_genre
	freq=[];
	for i in range(1,301):
		p = getFinalCount(0,15,i,p1)
		freq.append(p)
	print freq

def process():
	print "................................................................................................."
	user1 = pd.read_csv("users.dat",names = ["UserID", "Gender", "Age", "Occupation", "Zip-code"], sep="::|")
	movies1 = pd.read_csv("movies.dat",names = ["MovieID","Title","Genres"],sep="::|")
	ratings1 = pd.read_csv("ratings.dat",names=["UserID", "MovieID","Rating","Timestamp"],sep="::|")
	joined1 = pd.merge(ratings1, movies1, on='MovieID', how='left')
	joined = pd.merge(user1, joined1, on='UserID', how='left')
	joined["Gender"] = convert2(joined["Gender"])
	joined["Genres"] = convert2(joined["Genres"])
	joined["Zip-code"] = np.int64(joined["Zip-code"])

	#To print file 

	#for i in joined.columns:
	#	print i, np.dtype(joined[i])

	#Permutation of rows to inroduce randomness
	joined = joined.iloc[np.random.permutation(len(joined))]
	joined_1 = joined.copy()
	joined_1 = joined_1.drop('Title',1)
	joined_1 = joined_1.drop('Timestamp',1)
	joined_2 = joined_1.copy()
	joined_2.to_csv("joined1.csv")

	joined_2 = normalize(joined_2)
	joined_2.to_csv("joined.csv")

	# Putting ttarget column at the end

	joined_2['new_column'] = joined_2['Rating']
	joined_2 = joined_2.drop('Rating',1)
	joined_2 = joined_2.rename(columns={'new_column':'Rating'})

	## Calling Chi Square
	#chi_square(joined_2)
	return joined_2

def data_set_train_test(data_frame,phi):
	if(phi==0):
		joined_shape = data_frame.shape
		indice_10_percent = int((joined_shape[0]/100.0)* 10)
		dataFile1 = data_frame[indice_10_percent:]
		dataFile2 = data_frame[:indice_10_percent]
		dataFile1.to_csv("dataFile1",index=False)
		dataFile2.to_csv("dataFile2.csv",index=False)

		joined_shape = dataFile2.shape
		indice_70_percent = int((joined_shape[0]/100.0)* 70)
		dataFile2[indice_70_percent:].to_csv('testFile.csv', index = False)
		dataFile2[:indice_70_percent].to_csv('trainFile.csv', index = False)

		trainProduct = pd.read_csv('trainFile.csv')
		testProduct =  pd.read_csv('testFile.csv')
		return trainProduct, testProduct

	else:
		joined_shape = data_frame.shape
		indice_70_percent = int((joined_shape[0]/100.0)* 70)
		data_frame[indice_70_percent:].to_csv('testFile.csv', index = False)
		data_frame[:indice_70_percent].to_csv('trainFile.csv', index = False)

		trainProduct = pd.read_csv('trainFile.csv')
		testProduct =  pd.read_csv('testFile.csv')
		return trainProduct, testProduct


	
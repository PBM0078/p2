#12151389 Park Byung Moon

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from math import sqrt

def sort_dataset(dataset_df):
	sorted_df = dataset_df.sort_values(by='year')
    	return sorted_df

def split_dataset(dataset_df):	
	X = dataset_df.drop(['salary'], axis=1)  # Assuming 'salary' is the target variable
    	X_numerical = extract_numerical_cols(X)
    	Y = dataset_df['salary']

	X_train, X_test, Y_train, Y_test = train_test_split(X_numerical, Y, test_size=0.2, random_state=42)
    
    	return X_train, X_test, Y_train, Y_test

def extract_numerical_cols(dataset_df):
	numerical_cols = ['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war']
    	return dataset_df[numerical_cols]

def train_predict_decision_tree(X_train, Y_train, X_test):
	dt_model = DecisionTreeRegressor()
    	dt_model.fit(X_train, Y_train)

	dt_predictions = dt_model.predict(X_test)

    	return dt_predictions

def train_predict_random_forest(X_train, Y_train, X_test):
	rf_model = RandomForestRegressor()
    	rf_model.fit(X_train, Y_train)
    
    	rf_predictions = rf_model.predict(X_test)
    
    	return rf_predictions

def train_predict_svm(X_train, Y_train, X_test):
	svm_model = SVR()
    	svm_model.fit(X_train, Y_train)

    	svm_predictions = svm_model.predict(X_test)
    
    	return svm_predictions

def calculate_RMSE(labels, predictions):
	rmse = sqrt(mean_squared_error(labels, predictions))
    	return rmse

if __name__=='__main__':
	#DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
	data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')
	
	sorted_df = sort_dataset(data_df)	
	X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)
	
	X_train = extract_numerical_cols(X_train)
	X_test = extract_numerical_cols(X_test)

	dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
	rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
	svm_predictions = train_predict_svm(X_train, Y_train, X_test)
	
	print ("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))	
	print ("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))	
	print ("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))
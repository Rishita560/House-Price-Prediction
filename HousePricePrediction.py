import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from xgboost import XGBRegressor #one more method to train regression model

#MEDV is the col which we have to predict

dataset = pd.read_csv("C:/Users/sahur/Downloads/boston.csv")

print(dataset)#print whole dataset
print(dataset.head())#print above 5 rows
print(dataset.tail())#print below 5 rows
print(dataset.shape)#print total no. of rows and col
print(dataset.isnull().sum())#count null values
correlation = dataset.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='Blues')
plt.show()

#splitting dataset into 2 parts to train the model: (i)X = all col except last col (ii)Y = only last col
X = dataset.drop('MEDV',axis=1)
Y = dataset['MEDV']
print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=31)
print(X.shape, X_train.shape, X_test.shape)

#Training the model
model = LinearRegression()#Calling model
model.fit(X_train, Y_train)#Fit training part into model
model_prediction = model.predict(X_train)#Checking model
print(model_prediction)

print("Checking")
#Checking accuracy of model
score1 = metrics.r2_score(model_prediction, Y_train)#model_prediction=predicted val, Y_train=actual val; r2 will compare predicted with actual val; greater the r2 score to 1 is better the model will be
print("R2 score =", score1)
score2 = metrics.mean_absolute_error(model_prediction, Y_train)#lesser/closer the val to 0 the better model we have
print("Mean absolute error score =", score2)

#Training model using another method XGBRegressor
model2 = XGBRegressor()
model2.fit(X_train, Y_train)
model2_prediction = model2.predict(X_train)#Checking model
print(model2_prediction)
score1 = metrics.r2_score(model2_prediction, Y_train)#model_prediction=predicted val, Y_train=actual val; r2 will compare predicted with actual val; greater the r2 score to 1 is better the model will be
print("R2 score =", score1)
score2 = metrics.mean_absolute_error(model2_prediction, Y_train)#lesser/closer the val to 0 the better model we have
print("Mean absolute error score =", score2)

'''
#how actually the model is predicting
arr = np.array([[0.04527, 0.0, 11.93, 0, 0.573, 6.12, 76.7, 2.2875, 1, 273, 21.0, 396.90, 9.08]])
print("Actual ans = 20.6")
print(model.predict(arr))
print(model2.predict(arr))
'''

import pickle
pickle.dump(model,open("House Price Prediction.pkl","wb"))
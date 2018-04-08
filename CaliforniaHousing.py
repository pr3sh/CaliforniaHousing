'''California Housing Data

This data set contains information about all the block groups in California from the 1990 Census. 

The task in this project is to aproximate the median house value of each block from the values of the rest of the variables.

Data has been obtained from the LIACC repository. The original page where the data set can be found is: http://www.liaad.up.pt/~ltorgo/Regression/DataSets.html. '''

import tensorflow as tf 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

cal_housing = pd.read_csv('cal_housing_clean.csv')

X = cal_housing.drop('medianHouseValue',axis =1)
y = cal_housing['medianHouseValue']

X_train,X_test, y_train, y_test = train_test_split(X,y,test_size =0.3)

cal_housing.describe()


scaled = MinMaxScaler()

scaled.fit(X_train)

scaled.transform(X_test)


X_test = pd.DataFrame(data =X_test,index = X_test.index,columns = X_test.columns)

X_train = pd.DataFrame(data =X_train,index = X_train.index,columns =X_train.columns)

age = tf.feature_column.numeric_column('housingMedianAge')
rooms = tf.feature_column.numeric_column('totalRooms')
bedrooms = tf.feature_column.numeric_column('totalBedrooms')
pop = tf.feature_column.numeric_column('population')
households = tf.feature_column.numeric_column('households')
income = tf.feature_column.numeric_column('medianIncome')

feat_cols =[age,rooms,bedrooms,pop,households,income]

input_func= tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train, batch_size =10,shuffle =True,num_epochs =1000)

model  = tf.estimator.DNNRegressor(hidden_units=[15,15,15,6],feature_columns=feat_cols) #Deep Neural Network with 4 layers of 15,15,15 & 6 neurons,respectively.

model.train(input_fn=input_func,steps=5000)

input_pred_func = tf.estimator.inputs.pandas_input_fn(x =X_test, batch_size=10,num_epochs =1, shuffle =False)

pred_gen= model.predict(input_fn=input_pred_func)

predictions = list (pred_gen)

final_preds =[]

for pred in predictions:
    final_preds.append(pred['predictions'])


mean_squared_error(y_test,final_preds)**0.5


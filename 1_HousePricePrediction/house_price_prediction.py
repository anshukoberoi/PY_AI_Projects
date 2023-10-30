# Predicting House Prices Using Multiple Linear Regression
# 
# In this project we are going to see how machine learning algorithms help us predict house prices.
# Linear Regression is a model of predicting new future data by using the existing correlation between the old data.
# Here, machine learning helps us identify this relationship between feature(existing/independent) data and output(new/dependent), so we can predict future values.
#
# Multiple Linear Regression is a statistical method used to understand the relationship between multiple independent variables and a single dependent variable.
import pandas as pd
# we use sklearn library in many machine learning calculations..
from sklearn import linear_model
# we import our dataset: housepricesdataset.csv
df = pd.read_csv("housepricesdataset.csv", sep=";")
# lets see and check our data set:
print("df1:",df)
# The following is our feature set:
print("df2:",df[['area', 'roomcount', 'buildingage']])
# The following is the output(result) data:
print("df3:",df['price'])
# df['price']
print("df4:",df.columns)
print("df5:",df['price'].mean())
# we define a linear regression model here:
reg = linear_model.LinearRegression()
reg.fit(df[['area', 'roomcount', 'buildingage']], df['price'])
# lets predict a house with 230 square meters, 4 rooms and 10 years old building.
prediction_data1 = pd.DataFrame({'area': [230], 'roomcount': [4], 'buildingage': [10]})
x = reg.predict(prediction_data1)
# x = reg.predict([[230, 4, 10]])
print("predictedprice1:",x)
# Now lets predict a house with 230 square meters, 6 rooms and 0 years old building - its new building..
prediction_data2 = pd.DataFrame({'area': [230], 'roomcount': [6], 'buildingage': [0]})
x = reg.predict(prediction_data2)
print("predictedprice2:",x)
# Now lets predict a house with 355 square meters, 3 rooms and 20 years old building
# reg.predict([[355,3,20]])
prediction_data3 = pd.DataFrame({'area': [355], 'roomcount': [3], 'buildingage': [20]})
x = reg.predict(prediction_data3)
print("predictedprice3:",x)
# You can make as many prediction as you want..
# reg.predict([[230,4,10], [230,6,0], [355,3,20], [275, 5, 17]])
# prediction_data4 = pd.DataFrame({'area': [275], 'roomcount': [5], 'buildingage': [17]})
prediction_data_n = pd.DataFrame({'area': [230, 230, 355, 275],
                                'roomcount': [4, 6, 3, 5],
                                'buildingage': [10, 0, 20, 17]})
print("predictedprice_N:",reg.predict(prediction_data_n))
# Now we'll see the coefficients of our multi-linear regression formula.
# These can be calculated manually by OLS: "Ordinary Least Squares". (How compiler is calculating these?)
print('coef:',reg.coef_)
print('intercept:',reg.intercept_)
# Lets see the coeffients of Multiple Linear regression formula..
# y = a + b1X1 + b2X2 + b3X3 + b4X4 + b5X5 ...etc
a = reg.intercept_
b1 = reg.coef_[0]
b2 = reg.coef_[1]
b3 = reg.coef_[2]
x1 = 230
x2 = 4
x3 = 10
y = a + b1*x1 + b2*x2 + b3*x3
print("y",y)

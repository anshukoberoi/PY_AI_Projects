# Calculating Employee Salaries using Polynomial Linear Regression
#
# In this project we are going to build a machine learning model for exact calculation of employee salaries.
# Polynomial Linear Regression General Formula:
#
# y = a + b1*x + b2*x^2 + b3*x^3 + b4*x^4 + ....... + bN*x^N
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# We import our dataset using pandas into df dataframe..
df = pd.read_csv("salaries_dataset.csv",sep = ";")
print('df:',df)
# Let's take a look at our dataset - Data Visualization
plt.scatter(df['experience_level'],df['salary'])
plt.xlabel('Experience level')
plt.ylabel('Salary')
#  You can save the figure if you want!
plt.savefig('1.png', dpi=300)
plt.show()
# As you can see data is not distributed linearly..
# If we apply linear regression to the dataset we see an incorrect model graph, let's see:
linear_regression = LinearRegression()
linear_regression.fit(df[['experience_level']],df['salary'])
plt.xlabel('Experience Level)')
plt.ylabel('Salary')
plt.scatter(df['experience_level'],df['salary'])
x_axis_experience = df['experience_level']
y_axis_linear_predictedSalary = linear_regression.predict(df[['experience_level']])
plt.plot(x_axis_experience, y_axis_linear_predictedSalary,color= "green", label = "linear regression")
plt.legend()
plt.show()
# Very bad model prediction, so: It is not correct to apply linear regression for this dataset. Remember, you will choose a model according to your data set!
# First of all, you should have a very good understanding of your dataset !!!
# We decided that polynomial regression, one of the regression types, should be applied for this data set. Now let's see how we implement it:
# We adapt our x value to fit the polynomial function above
# So => 1, x, x^2 (N=2)
# We call the PolynomialFeatures function to create a polynomial regression object.
# We specify the degree (N) of the polynomial when calling this function:
polynomial_features = PolynomialFeatures(degree = 4)
featureset_polynomial = polynomial_features.fit_transform(df[['experience_level']])
# We fit the x axis with featureset_polynomial and y axes by creating our linear_regression object, which is our regression model object,
# and calling its fit method.
# So we train our regression model with data:
linear_regression = LinearRegression()
linear_regression.fit(featureset_polynomial,df['salary'])
# Now that our model is ready, let's see how our model generates a result graph based on the available data:
y_predictedSalaries = linear_regression.predict(featureset_polynomial)
plt.plot(df['experience_level'],y_predictedSalaries,color= "red",label = "polynomial regression")
plt.legend()
# # Let's scatter our data set as points and see if it fits polynomial regression:
plt.scatter(df['experience_level'],df['salary'])
plt.show()
# As you can see, we can say that it definitely fits, polynomial regression is the right choice.
# Now let's make N=3 or 4 and see if we increase the polynomial degree, will it fit better?
# Calculate of a new employee who has experience level 4.5
experience_level_new_employee  = polynomial_features.fit_transform([[4.5]])
print(linear_regression.predict(experience_level_new_employee ))
# The salary he will receive fits the company policy very well ! :)

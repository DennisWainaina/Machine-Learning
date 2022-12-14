# In this program we're going to build a simple linear regression model
# This is to see how years of experience affects salary
# In this case salary will be the dependent variable while years of experience the independent variable
# Salary depends on years of experience
# First step is importing the database
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
df = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Linear_Regression_Introduction/master/'
                 'Salary_Data.csv')
print(df)
# The code above was to see how the dataframe looked like

# Then we check for missing values
print(df.isnull().sum())
# There are no missing values therefore we don't need to fill the missing values

# We then plot the data using a scatter plot to see how the data looks like visually
# This is just to get a rough estimate of how salary varies with years of experience
plt.figure(figsize=(32, 10))
plt.scatter(df['YearsExperience'], df['Salary'], marker='x')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title("Salary vs Years of Experience")
plt.show()

# We now want to build the machine learning model first we split the data into target and input variables
x = df[['YearsExperience']]  # This is the input predictor variables
y = df['Salary']  # This is the target variable

# Now splitting the data into test and train data to build our model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=47)

# We want to now train our model by fitting the training data set of the input and predictor variables
# We use linear regression to plot the values of data to determine the linear relationship between the two values
# This determines the line of best fit from the data given so the task is to find the slope and y-intercept of best fit
linear_regressor = LinearRegression()
print(linear_regressor.fit(x_train, y_train))

# Now printing the y-intercept and the gradient(slope) values
print('The y-intercept is', linear_regressor.intercept_)
print('The gradient is', linear_regressor.coef_)  # Coefficient is the slope of the curve
# With this we have now created our model

# After training our model we want to see how accurate it predicts salries from the years of experience
y_pred = linear_regressor.predict(x_test)  # We use our test data to see how accurately salaries are predicted
plt.plot(x_test, y_test, 'rx')  # These are the actual values
plt.plot(x_test, y_pred, color='black')  # These are the predicted values
plt.show()

# We now see how accurately our model predicted the data using MSE, RMSE and MAE
# MSE takes the average of the squares of the difference between the predicted and actual values
# RMSE is the square root of MSE
# MAE is the absolute value of the difference between the predicted and actual values
print(metrics.mean_absolute_error(y_test, y_pred))  # For the mean absolute error(MAE)
print(metrics.mean_squared_error(y_test, y_pred))  # For the mean squared error(MSE)
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  # For the root mean squared value (RMSE)

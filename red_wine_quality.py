# In this project we will predict wine quality from 1 to 10 using various input variables
# Here we will use a logistic regression model for prediction
# First we load appropiate libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix

# Then we load the dataframe
red_wine_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
                            , sep=";")
print(red_wine_data)

# We check if there are missing values
print(red_wine_data.isnull().sum())  # There are no missing values

# Seperating the categorical and numerical variables
# First the numerical variables
numerical_variables = red_wine_data.select_dtypes(include=np.number).columns.tolist()
print('These are the numerical variables')
print(numerical_variables)

# Then the categorical variables
categorical_variables = list(set(red_wine_data.columns) - set(numerical_variables))
print('These are the categorical variables')
print(categorical_variables)  # There are no categorical variables hence no need for one-hot encoding

# Seperating the data into target and input(predictor) variables
x = red_wine_data.drop('quality', axis=1)  # The imput variables
y = red_wine_data['quality']
print(x)
print(y)

# Splitting data into test and train data for the input and target variables
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
print(x_train)
print(x_test)

# Building the model
model = LogisticRegression()
model.fit(x_train, y_train)
print(model)

# Comparing predicted values to actual values
predictions = model.predict(x_test)  # These are the predicted values for the target variable based on test input
print('The predicted values are')
print(predictions)
print('The actual values are')
print(y_test.values)

# Seeing how accurate our model is
accuracy = accuracy_score(y_test, predictions)
print('The accuracy score is', accuracy)
r2 = r2_score(y_test, predictions)
print('The r2 score is', r2)  # 0.3514
MSE = mean_squared_error(y_test, predictions)
print('The mean squared error is', MSE)
MAE = mean_absolute_error(y_test, predictions)
print('The mean absolute error is', MAE)

# Making confusion matrix to see which predicted data actually happened
cm = confusion_matrix(y_test, predictions)
print(cm)

# PLotting the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

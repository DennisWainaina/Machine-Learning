# In this project we're going to be learning on multiple linear regression
# This is regression where there are more than one input variables used to predict the target variables
# First we import the dataframe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split  # Used to split data into test and training data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics


# First we see how the data looks like
df = pd.read_csv('https://raw.githubusercontent.com/tanlitung/Datasets/master/kc_house_data.csv')
print(df)
print(df.head())

# We then check if there are missing values
print(df.isnull().sum())
# There are no missing values

# We check for the numeric and categorical variables
numerical_variables = df.select_dtypes(include=np.number).columns.tolist()
print('These are the numerical variables')
print(numerical_variables)

# We check for categorical variables
categorical_variables = list(set(df.columns) - set(numerical_variables))
print('These are the categorical variables')
print(categorical_variables)

# Converting categorical variables to numeric using one hot encoding
df_dummy = pd.get_dummies(df, prefix_sep='_', drop_first=True)
print(df_dummy)

# Selecting the important variables for the target and prediction(input) variables and omitting the not important
df.drop('id', axis=1, inplace=True)
df.drop('date', axis=1, inplace=True)
print(df)
# The columns of date and id were dropped as they were not important for the prediction
print(df.info())

# Seeing how the values correlate with each other linearly using the heat map
plt.figure(figsize=(20, 10))
sns.heatmap(df.corr(), annot=True)
plt.show()

# Choosing the target and imput variables
x = df.drop('price', axis=1)
y = df['price']
print(x)
print(y)

# One can also normalise the data this is to make all values be between 0 and 1
# This is used when curve doesn't follow Gaussian distribution
# This is to say that there is a big difference between the smallest and large value
# Hence it becomes harder to compare values to the high difference between them when plotting
# When the curve follow Gaussian distribution one uses standardisation
# This is making the mean 0 and the variance to be 1
# During plotting one can choose the appropiate method depending on the data given
# For example:

min_max_scaler = MinMaxScaler()
col_name = df.drop('price', axis=1).columns[:]
x = df.loc[:, col_name]
y = df['price']
print(x.describe())

# Normalizing x
x = pd.DataFrame(data=min_max_scaler.fit_transform(x), columns=col_name)

# Examining normalised data
print('Before normalising')
print(df.head())
print('After normalising')
print(x.head())

# Let's try something to see about the formula used in normalising
print('The maximum value in the bedrooms column is', df['bedrooms'].max())
print('The minimum value in the bedrooms column is', df['bedrooms'].min())
print('The first value in the bedrooms column is', df['bedrooms'][0])
first_normalised_value = (df['bedrooms'][0] - df['bedrooms'].min())/(df['bedrooms'].max() - df['bedrooms'].min())
print('The first normalised value in the normalised bedroom column is', first_normalised_value)
# Wanted to see if the formula for normalisation actually worked

# Splitting the target and input variables into test and train data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)
print(x_train)
print(x_test)

# Building the model
model = LinearRegression()  # Instead of normalising the data we can use normalise is equals to True
model.fit(x_train, y_train)
print(model)

# Determining the co-efficients of the equation of a straight line (the slope and y-intercept)
print('The y-intercept of the model is', model.intercept_)
print('The gradient of the model is', model.coef_)
# It is important to note that the gradients are many due to the prescence of multiple input variables
# Hence multiple gradients from the different lines all with the same y-intercept

# We want to see how our model looks like therefore we shall visualise it using matplotlib
y_pred = model.predict(x_test)  # We use our test data to see how accurately salaries are predicted
plt.plot(x_test, y_test, 'rx')  # These are the actual values
plt.plot(x_test, y_pred, color='black')  # These are the predicted values
plt.show()

# We want to evaluate our model to see how it works using various criteria as the MAE, MSE and RMSE and R2
# First we evaluate using the RMSE model
train_target_pred = model.predict(x_test)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print('The value of the root mean squared is ', rmse)

# Then we evaluate using the R2 score
r2 = metrics.r2_score(y_test, y_pred)
print('The r2 score is', r2)

# We now want to see the difference between the predicted and actual values for the test dataset
print('For the testing dataset')
output = pd.DataFrame(y_test.iloc[0: 10])
output['predicted'] = y_pred[0: 10]
output['difference'] = output['predicted'] - output['price']  # This is the difference btw actual and predicted price
print(output)
r2 = metrics.r2_score(y_test, y_pred)
print('R2 score for testing dataset is: ', r2)
print('Root mean squared for testing dataset is: ', rmse)

# We now want to do the same for the train dataset
print('For the training dataset')
output = pd.DataFrame(y_train.iloc[0: 10])
y_pred = model.predict(x_train)
output['predicted'] = y_pred[0: 10]
output['difference'] = output['predicted'] - output['price']  # This is the difference btw actual and predicted price
print(output)
r2 = metrics.r2_score(y_train, y_pred)
print('R2 score for training dataset is: ', r2)
print('Root mean squared for training dataset is: ', rmse)

# There is low R2 score and high RMSE showing the data has some problems which may include:
# 1. Data is too dispersed
# 2. Some input variables do not have a linear relationship to the target variable
# These problems can be solved using data pre-processing, trying out other models or even changing the test_split ratio
# It is important to note that data science is not just about building models it is using data to make decisions
# Python is just a tool used for prediction

# In this project we will build a model that predicts the price of a house based on certain parameters such as the
# number of bedrooms, no  of bathrooms and the square feet of the house
# This was a test to see how accurate our model predicted house prices with comparing to real data which is not shown
# First we load important modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge  # To try and improve accuracy
from sklearn.metrics import r2_score

# Then we load the dataframe to see the columns
house_data = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/kc_house_data/'
                         'kc_house_data.csv')
print(house_data)

# Check if there are missing values
print(house_data.isnull().sum())  # No missing values hence no need for cleaning

# We then perform some EDA to understand the data a bit better
# For example what is the effect of bedrooms on the house price?
print(house_data['bedrooms'])

# First we see the prices of 3 bedrooms
three_bedroom_houses = house_data['bedrooms'][house_data['bedrooms'] == 3]
print(three_bedroom_houses)
price_of_three_bedroom_houses = house_data['price'][three_bedroom_houses]
print(price_of_three_bedroom_houses)  # These are the prices of three-bedroom houses

# Then we see the prices of two-bedroom houses
two_bedroom_houses = house_data['bedrooms'][house_data['bedrooms'] == 2]
print(two_bedroom_houses)
price_of_two_bedroom_houses = house_data['price'][two_bedroom_houses]
print(price_of_two_bedroom_houses)  # These are the prices of two-bedroom houses

# Then we see the price of one-bedroom houses
print(house_data['bedrooms'].min())  # Wanted to see minimum no of bedrooms which is apparently 0
one_bedroom_houses = house_data['bedrooms'][house_data['bedrooms'] == 1]
print(one_bedroom_houses)
price_of_one_bedroom_houses = house_data['price'][one_bedroom_houses]
print(price_of_one_bedroom_houses)  # These are the prices of one-bedroom houses
# It appears one-bedroom houses are more expensive than two-bedroom houses
# But three-bedroom houses are the most expensive or let us see the average price of all the houses
print(price_of_three_bedroom_houses.mean())  # 604,000
print(price_of_two_bedroom_houses.mean())  # 180,000
print(price_of_one_bedroom_houses.mean())  # 538,000
#  Three-bedroom houses are therefore the most expensive followed by one bedroom then two-bedroom houses


# We then want to see the effect of waterfronts on house price(target variable)
# First the houses with waterfront views
waterfront_properties = house_data['waterfront'][house_data['waterfront'] == 1]
print(waterfront_properties)
price_of_waterfront_properties = house_data['price'][waterfront_properties]
print(price_of_waterfront_properties)  # These are the prices of houses with waterfront views

# Then the houses without waterfront views
properties_without_waterfornt = house_data['waterfront'][house_data['waterfront'] == 0]
print(properties_without_waterfornt)
price_of_non_waterfront_properties = house_data['price'][properties_without_waterfornt]
print(price_of_non_waterfront_properties)  # These are the prices of houses with waterfront views
# Properties with waterfront are more expensive than the ones without waterfront


# Finally, we want to see the effect on the no of floors on the house price
print(house_data['floors'])  # First we want to see the column
print(house_data['floors'][house_data['floors'] == 3])  # There are houses with 3 floors
print(house_data['floors'].max())  # Max number of floors is 3.5
print(house_data['floors'][house_data['floors'] == 3.5])  # There are houses with 3.5 floors

# First houses with 3 floors price
three_floors = house_data['floors'][house_data['floors'] == 3]
print(three_floors)
price_of_three_floors = house_data['price'][three_floors]
print(price_of_three_floors)

# Then houses with 2 floors price
two_floors = house_data['floors'][house_data['floors'] == 2]
print(two_floors)
price_of_two_floors = house_data['price'][two_floors]
print(price_of_two_floors)

# Then houses with 1 floor price
one_floors = house_data['floors'][house_data['floors'] == 1]
print(one_floors)
price_of_one_floors = house_data['price'][one_floors]
print(price_of_one_floors)
# Same case with bedrooms highest with three high floors followed by one floor then by 2 floors


# After doing some EDA we now want to build our Linear Regression model
# First we seperate the categorical and numerical variables
# We begin with the numerical variables
numerical_variables = house_data.select_dtypes(include=np.number).columns.tolist()
print('The numerical variables are')
print(numerical_variables)

# Then the categorical variables
categorical_variables = list(set(house_data.columns) - set(numerical_variables))  # All columns except numerical ones
print('These are the categorical variables')
print(categorical_variables)  # There are no categorical variables hence no need for one-hot encoding


# Seperating the target and input variables
x = house_data.drop('price', axis=1)  # These are the input variables
y = house_data['price']  # This is the target variable
print(x)
print(y)

# Splitting the target and input variables into training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)
print(x_train)
print(x_test)  # Split was succesful

# We then use this data to train our model
model = LinearRegression()
model.fit(x_train, y_train)

# We then use this model to predict data from our test data
y_predict = model.predict(x_test)
y_predicted = model.predict(x.head(4999))
print(y_predict)
print(y_test)
print(y_predicted)

# Loading another data set which is now the dataframe with the real prices but deleted
eval_data = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/kc_house_data/'
                        'kc_house_new_test_data.csv')
print(eval_data)
# Setting price column equal to test predicted
eval_data['prediction'] = y_predicted
print(eval_data)

# Out of curiosity calculating the R2 score
r2 = r2_score(y_test, y_predict)
print('The r2 score is', r2)

# Saving to csv file
eval_data['prediction'].to_csv('predicted_evaluation_data.csv')
print(eval_data['prediction'].info())

# We now want to inporve accuracy of the model
# We shall use something known as the ridge score
ridge = Ridge(alpha=0)
ridge.fit(x_train, y_train)
print('The training score is', ridge.score(x_train, y_train))
new_predict = ridge.predict(x_test)

# We now compare the accuracy score before and after ridge model
accuracy = r2_score(y_test, y_predict)
print('The r2 score before ridge model is', accuracy)
new_accuracy = r2_score(y_test, new_predict)
print('The new r2 score after ridge model is', new_accuracy)  # Not much of a difference trying another model

# Another method we could try is removing outliers
# To see the outliers we could use a scatter plot
plt.figure(figsize=(32, 10))
sns.boxplot(x='bedrooms', y='price', data=house_data)
plt.xlabel('Bedrooms')
plt.ylabel('House price')
plt.xlim(0, 4)
plt.ylim(100000, 600000)
plt.show()

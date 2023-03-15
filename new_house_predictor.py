# Loading important libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from boruta import BorutaPy

# Loading dataset
df = pd.read_csv(r"C:\Users\user\Desktop\project\train.csv")
print(df)

# Checking info about the dataset
print(df.info())

# Checking if there are null values
print(df.isnull().sum())

# Splitting the numerical and categorical variables
# First the numerical variables
numerical_variables = df.select_dtypes(include=np.number).columns.tolist()
print('These are the numerical variables')
print(numerical_variables)

# Then the categorical variables
categorical_variables = list(set(df) - set(numerical_variables))
print('These are the categorical variables')
print(categorical_variables)

# Checking for null values in the numerical variables
print('Before filling')
print(df[numerical_variables].isnull().sum())  # There are missing values
# Filling the numerical variables with mean of numerical values
df[numerical_variables] = df[numerical_variables].fillna(df[numerical_variables].mean())
# Checking to see if it worked
print('After filling')
print(df[numerical_variables].isnull().sum())

# Checking for null values in the categorical variables
print('Before filling')
print(df[categorical_variables].isnull().sum())  # There are missing values
# Filling the categorical variables with mode of categorical values
df[categorical_variables] = df[categorical_variables].fillna(df[categorical_variables].mode().iloc[0])  # Filling values
# Checking to see if it worked
print('After filling')
print(df[categorical_variables].isnull().sum())

# Now seeing the null values for the whole dataframe
print(df.isnull().sum())  # It worked there are no more null values

# Performing one-hot encoding to the categorical variables
df_categorical = pd.get_dummies(df[categorical_variables], prefix_sep='_', drop_first=True)
# Combine the numerical variables with the encoded categorical variables
df_processed = pd.concat([df[numerical_variables], df_categorical], axis=1)

# Scaling the input variables to improve model performance
scaler = StandardScaler()
df_processed[numerical_variables] = scaler.fit_transform(df_processed[numerical_variables])

# Separating the target and input variables
y = df_processed['SalePrice']  # Target variable
x = df_processed.drop('SalePrice', axis=1)  # Input variables

# Seperating the data into train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

print(x_train)
print(x_test)

# Building the base model
base_model = LinearRegression()
base_model.fit(x_train, y_train)
print(base_model)

# After building the model we now try to use it to make predictions based on the model built
y_predict = base_model.predict(x_test)
print('These are the predictions')
print(y_predict)

# Seeing accuracy of the predictions by comparing them to the actual values
r2 = r2_score(y_test, y_predict)
print('The r2 score of the base model is', r2)  # -2.152*10^21

# Now seeing the effect with using random forest classifier which basically chooses the best from a lot of DTCs
base_model1 = AdaBoostRegressor(n_estimators=100, random_state=1)
base_model1.fit(x_train, y_train)
y_predict = base_model1.predict(x_test)
print(y_predict)

# Seeing the new accuracy, f1 score and r2 score using ensemble methods
r2 = r2_score(y_test, y_predict)
MSE = mean_squared_error(y_test, y_predict)
print('The r2 score of the gradient boosting regressor model is', r2)
print('Mean squared error of the gradient boosting regressor model is', MSE)  # 0.1927

# Seeing the effect of feature selection on the r2 score of the model
# Selecting important features
selected_features = SelectFromModel(estimator=base_model1)
# Training selector
selected_features.fit(x_train, y_train)
# Now with our new data we build a model based on this data
# First transforming our data to include only important features
x_important_train = selected_features.transform(x_train)
x_important_test = selected_features.transform(x_test)
# Training our model based on these features
base_model1.fit(x_important_train, y_train)
y_predict1 = base_model1.predict(x_important_test)

# Seeing the new r2 score after using feature selection
r2 = r2_score(y_test, y_predict1)  # 0.8357
MSE = mean_squared_error(y_test, y_predict1)
print('R2 score after feature selection of the gradient boosting regressor model is', r2)  # 0.8382
print('Mean squared error after feature selection of the gradient boosting regressor model is', MSE)  # 0.1998

# Trying to see the effect of using Boruta methods on the r2 score
boruta_selector = BorutaPy(base_model1, n_estimators=100, verbose=2, random_state=1)
# Finding important features
boruta_selector.fit(np.array(x_train), np.array(y_train))
# Transforming data using boruta to include important features
x_important_train = boruta_selector.transform(np.array(x_train))
x_important_test = boruta_selector.transform(np.array(x_test))
# Training our model based on these features
rf_important = AdaBoostRegressor(n_estimators=100, random_state=1)
rf_important.fit(x_important_train, y_train)
# Predicting data based on our new model
new_predictions = rf_important.predict(x_important_test)
# Seeing r2 score after Boruta methods
r2 = r2_score(y_test, new_predictions)
print('R2 score after Boruta is', r2)  # 0.8106

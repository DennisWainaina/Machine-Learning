# This machine learning model will use the titanic dataset used before to see if the passenger survived
# This will use factors such as the age, class and other features as the predictor variables
# The target variable in this case will be the survived variable
# This is a classification problem as it classifies the outcome into two or more types
# Loading the dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
import graphviz


df = pd.read_csv('https://raw.githubusercontent.com/dphi-official/First_ML_Model/master/titanic.csv')
print(df)
# Before doing anything we check if there are missing values in the variables(columns)
print('Before cleaning the number of missing values in the columns is')
print(df.isnull().sum())

# The first step is pre data processing which involves filling missing values and making categorical variables numeric
# First cleaning the numerical variables
numerical_variables = df.select_dtypes(include=np.number).columns.tolist()  # First we isolate numeric variables
print('The numerical variables are', numerical_variables)
# Filling the numerical variables with mean of numerical values
df[numerical_variables] = df[numerical_variables].fillna(df[numerical_variables].mean())  # Then we fill nulls with mean
print(df[numerical_variables])
# Checking to see if it worked
print('No of null values is', df[numerical_variables].isnull().sum())  # We then see if it worked

# We then clean the categorical variables
categorical_variables = list(set(df.columns) - set(numerical_variables))  # First we isolate categorical variables
print('The categorical variables are', categorical_variables)
df[categorical_variables] = df[categorical_variables].fillna(df[categorical_variables].mode().iloc[0])  # Filling values
print('No of null values is', df[categorical_variables].isnull().sum())
print(numerical_variables + categorical_variables)
# Checking to see if it worked
print('The new number of missing values after cleaning for every column is')
print(df.isnull().sum())
print('The people who embarked from various points are')
print(df['Embarked'].value_counts())

# We then convert the categorical variables to numeric using one-hot encoding
# Converting sex variable to numeric using the lambda function;
df['Ismale'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)
df['Sex'] = df['Ismale']

# Alternatively we could do it for all the categorical variables at once;
df_dummy = pd.get_dummies(df, prefix_sep='_', drop_first=True)
# This is referred to as one-hot encoding where the categorical variables are split into 1 and 0 for yes and no
df1 = df_dummy
print('The columns after one-hot encoding are', df1.columns)
print('The dummy columns are', df_dummy)
print('The no of males after one-hot encoding are', len(df1['Ismale'][df1['Ismale'] == 1]))
print('The no of females after one-hot encoding are', len(df1['Ismale']) - len(df1['Ismale'][df1['Ismale'] == 1]))
print(df_dummy.info())

# Checking if there are missing values in our program after filling
print(df.isnull().sum())

print(len(df.columns))

# We are now building the ML algorithm
# First we take only the variables for which we will be using to build ML model
# These are variables which are relevant to the target variable in question
# Such that they have a chance of affecting the target variable in question
df2 = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Parch', 'SibSp']]
print(df2)

# Next we seperate the target and input variables
# Target variable is the variable to learn more about
x = df2.drop('Survived', axis=1)
print('The input(predictor) variables are')
print(x)

y = df['Survived']
print('The target variable is')
print(y)
# y then becomes the target variable and the others become the inpout variables

# We are now building the machine learning model
# First step is to split the data into training and test data according to a specific ratio for example:
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
# The piece of code takes random values of a certain ratio for test and train date
# It may take different for each time that's why the random state is defined so that it always takes the same
# This is because this is a piece of code being shared between people which may vary with everyone if always random
print(x_train)  # This is the input variable data to be trained
print(x_test)  # This is the input variable data to be tested
print(y_train)  # This is the target variable data to be trained
print(y_test)  # This is the target variable to be tested
# It is important to note that x_train and y_train have the same no of rows showing that splitting occured well
# Now the next task is to build a model using the training data for the input and target variables
# Here we use a module known as DecisionTreeClassifier for example:
model = DecisionTreeClassifier()
data = model.fit(x_train, y_train)

# After building the model we test the accuracy of the model using the target and input variables
# This is done for the training and test data
print('The accuracy of the training data is', accuracy_score(y_train, model.predict(x_train)))
print('The accuracy of the test data is', accuracy_score(y_test, model.predict(x_test)))
# As can be seen the accuracy for the test data set is lower because of fewer data to predict if passenger survived
# The goal is to improve accuracy of test data set
# The higher accuracy for the training data set as opposed for the test data set indicates overfitting
# Over-fitting occurs when the data set is too much trained on the training data that it cannot predict other data
# Outside the training data
# For example the training data could have too much data of one type or even have fewer data sets in the training model
# To work with
# This is a problem when testing with real world data as it cannot generalise to give accurate predictions as it has
# been trained with only a limited set of data in terms of size and type
# An example of this is predicting whether an animal is a dog by only showing hundreds of images of only 2 or 3 types.
# One way to deal with this is to prune the decision tree by reducing the no of decisions it makes for example:
model_improved = DecisionTreeClassifier(max_depth=3)
# This reduces the no of decisions made by the decision tree to 3
model_improved.fit(x_train, y_train)
# Now fitting the data to the decision tree in the code above
print(model_improved)
# Checking to see if it worked by checking accuracy of training and test data in the model
print('The new accuracy of the training data is', accuracy_score(y_train, model_improved.predict(x_train)))
print('The new accuracy of the test data is', accuracy_score(y_test, model_improved.predict(x_test)))
# Data scientists improvise and build better models that provide meaningful results

# We want to see how the model looks like here we will use a module known as graphviz
dot_data = export_graphviz(model_improved, out_file=None, feature_names=x_test.columns, class_names=['0', '1'],
                           filled=True, rounded=True, special_characters=True)

graph1 = graphviz.Source(dot_data, format='png')
graph1.render(view=True)
# The code works and the project is complete

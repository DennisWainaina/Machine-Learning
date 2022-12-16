# This is a mdoel that predicts heart failure in a patient
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

heart_df = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/framingham.csv')
print(heart_df)

# Before we proceed with the building of the regression model and the steps that preceed it let us clean the data
print(heart_df.isnull().sum())

# Next step is to drop the unimportant variables
heart_df.drop('education', axis=1, inplace=True)
print(heart_df)

# It appears we have null values therefore we have to clean the data
# First step is to seperate the data into numeric and categorical variables
numeric_variables = heart_df.select_dtypes(include=np.number).columns
print('These are the numeric variables')
print(numeric_variables)
# Filling the blank spaces in the numeric variables
heart_df[numeric_variables] = heart_df[numeric_variables].fillna(heart_df.mean())
# Checking to see if it worked
print('After filling in the null values')
print(heart_df[numeric_variables].isnull().sum())
# There are no null values in the numerical columns(numeric variables)

# The categorical variables are;
categorical_variables = list(set(heart_df.columns) - set(numeric_variables))
print(categorical_variables)
# There are no categorical variables

# Seeing how the dataframe looks like
print(heart_df.info())
print(heart_df.describe())
print(heart_df['male'])

# We want to predict the chances of a person having CHD in 10 years
# First we want to see the number of people who have chances of getting CHD in 10 years
print(len(heart_df['TenYearCHD'][heart_df['TenYearCHD'] == 1]))
CHD = heart_df['TenYearCHD'][heart_df['TenYearCHD'] == 1]
print('These are the people with CHD in 10 years')
print(CHD)

# First we do some exploratory data analysis to understand the data
# Out of curiosity seeing how many smokers have chances of coronary heart disease
CHD_smokers = CHD[heart_df['currentSmoker'] == 1]
print('The no of people who are smokers with chances of heart disease are', len(CHD_smokers))

# Seeing how many people with diabetes have chances of heart disease in 10 years
CHD_diabetes = CHD[heart_df['diabetes'] == 1]
Diabetes = heart_df['diabetes'][heart_df['diabetes'] == 1]
print(Diabetes)
print('The no of people who have diabetes with chances of heart disease are', len(CHD_diabetes))

# Seeing how many people with prevalent hypertension have chances of heart disease in 10 years
CHD_hypertension = CHD[heart_df['prevalentHyp'] == 1]
Hypertension = heart_df['prevalentHyp'][heart_df['prevalentHyp'] == 1]
print(Hypertension)
print('The no of people who have hypertension with chances of heart disease are', len(CHD_hypertension))

# Seeing how many people with prevalent stroke have chances of heart disease in 10 years
CHD_hypertension = CHD[heart_df['prevalentStroke'] == 1]
stroke = heart_df['prevalentStroke'][heart_df['prevalentStroke'] == 1]
print(stroke)
print('The no of people who have had strokes is', len(stroke))
print('The no of people who have prevalent strokes with chances of heart disease are', len(stroke))

# Seeing the maximum number of cigarettes smoked per day by one person
print('The max amount of cigarettes smoked per day by one person is', heart_df['cigsPerDay'].max())

# Out of curiosity seeing if the person has chance of CHD in 10 years
heavy_smoker = heart_df['cigsPerDay'].max()
alive = heart_df['TenYearCHD'][heart_df['cigsPerDay'] == heavy_smoker]
print('The answer is')
print(alive)

# Plotting the data
# For the male column
plt.hist(heart_df['male'])
plt.ylabel('No with heart disease')
plt.xlabel('ismale')
plt.title('No of males with heart disease')
plt.show()

# Let's try and see if we can plot for all the variables at the same time
plt.hist(heart_df['TenYearCHD'], label='TenYearCHD')
plt.legend()
plt.show()

# Drawing countplot of CHD data
print(heart_df.columns[0])
sns.countplot(x='TenYearCHD', data=heart_df)
plt.show()

# Logistic regression is the use of regression analysis to predict categorical(dependent) variable from input variables.
# It uses a linear equation of a line where y is the predicted value and B0 and B1 are the parameters.
# It then converts the predicted value to a probability between 0 and 1 using the sigmoid function
# Positive values are predictive of class 1 and negative values of class 0
# Types of logistic regression binomial, multinomial and ordinal
# Binomial is a yes or no true or false choice where it makes a choice btw 2 possible categories
# In multinomial target variable has 3 or more categories e.g. fruit- apple, banana or orange
# In ordinal target variable has 3 or more ordinal categories such as good, bad or worse
# Since the probability value gotten from the sigmoid function we need a threshold value
# This threshold value(decision boundary) tells if the class is 1 meaning yes or 0
# It is p >= 0.5 class 1 and p < 0.5 class 0 

# Now splitting the data into the target and input variables
df = heart_df[['age', 'male', 'cigsPerDay', 'totChol', 'sysBP', 'glucose', 'TenYearCHD']]
x = df.drop('TenYearCHD', axis=1)
y = df['TenYearCHD']

# Creating the training and test variables
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)
print(x_train)
print(y_train)
print(x_test)

# Building the model
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(y_pred)

# Checking accuracy of model
accuracy = accuracy_score(y_test, y_pred)
print('The accuracy of the model is', accuracy)

# We then describe the perfomance of a model using something known as the confusion matrix
# This matrix compares the no of predicted values to the no of actual values
# For example it can have the no of values predicted for yes compared to the actual values of yes
# Or it can have the no of values predicted for no compared to the actual values of no
# yes is 1 and no is 0
# There are four terms in a confusion matrix TP, TN, FP and FN
# TP stands for True positive meaning what predicted yes actually happened
# TN stands for True negative meaning what predicted no didn't happen
# FP stands for False positive meaning what predicted yes didn't happen
# FN stands for False negative meaning what predicted no actually happened
# There are 4 metrics used to evaluate the confusion matrix
# These are the accuracy, sensitivity, specificity and precision
# Accuracy is how accurate the model is (TP + TN)/(total)
# Sensitivity is how often does it predict yes (TP)/(TP + FN)
# Sensitivity is how often does it predict no (TN)/ (TN + FP)
# Precision is if it predicts yes how correct is the model (TP)/(predicted yes)

# Plotting the confusion matrix
cm = confusion_matrix(y_test, y_pred)
conf_matrix = pd.DataFrame(data=cm, columns=['Predicted:0', 'Predicted:1'], index=['Actual:0', 'Actual:1'])
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.show()

# Evaluating the confusion matrix
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

Accuracy = TP/(TP + TN + FP + FN)
Sensitivity = TP/(TP + FN)
Specificity = TN/(TN + FP)
Precision = TP/(TP + FN)

print('Accuracy of model is', accuracy)
print('Sensitivity of model is', Sensitivity)
print('Specificity of model is', Specificity)
print('Precision of model is', Precision)

# Drawing ROC curve
y_pred_prob = model.predict_proba(x_test)[:, :]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:, 1])
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC Curve for Heart Disease Classifier')
plt.xlabel('False Positivity Rate(1 - Specificity)')
plt.ylabel('True Positive Rate(Sensitivity)')
plt.grid(True)
plt.show()

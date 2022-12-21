# Here we will learn about evaluating our models using various metrics and parameter tuning
# We evaluate our data to see how good it is how it compares to other models and how it will perform on new data
# One of thr models used to evaluate accuracy is the accuracy score which divides the correctly predicted over
# total predicted

# For this we shall use a dataset which determines if a person has diabetes(target variable) based on certain factors
# First step is importing important modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import export_graphviz
import graphviz

# Since columns are numerical we give columns our own name for understanding purposes
col = ["num_preg", "plasma_glucose_conc", "D_blood_pressure", "skin_fold_thickness", "serum_insulin",
       "body_mass_index", "pedigree_func", "age", "diabetes"]
diabetes_df = pd.read_csv("https://raw.githubusercontent.com/dphi-official/ML_Models/master/Performance_Evaluation/"
                          "diabetes.txt", names=col)
print(diabetes_df)

# First step is to clean the data
print(diabetes_df.isnull().sum())  # No missing values hence no need for cleaning

# Splitting dataset into categorical and numerical variables
# First the numerical variables
print('The numerical variables are')
numerical_variables = diabetes_df.select_dtypes(include=np.number).columns.tolist()

# Then the categorical variables
print('The categorical variables are')
categorical_variables = list(set(diabetes_df.columns) - set(numerical_variables))
print(categorical_variables)  # No categorical variables hence no need for one-hot encoding

# Choosing the input and target variables
x = diabetes_df.drop('diabetes', axis=1)
y = diabetes_df['diabetes']
print(x)
print(y)

# Splitting input and target variables into training and test data set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
print(x_train)  # Split was successful
print(x_test)

# Before trying with new models lets see how it would look with decison tree
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
print(model)

# Viewing the model
dot_data = export_graphviz(model, out_file=None, feature_names=x_test.columns, class_names=['0', '1'],
                           filled=True, rounded=True, special_characters=True)

graph1 = graphviz.Source(dot_data, format='png')
graph1.render(view=True)

# We want to see how accurate our model is
y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Testing accuracy between two different models
mlp = MLPClassifier(max_iter=1000)
mlp.fit(x_train, y_train)
y_predicted = mlp.predict(x_test)
cm1 = confusion_matrix(y_test, y_predicted)
print(cm1)


# There is a way to get the tn, fp, fn, tp in one line from the confusion matrix as shown
tn, fp, fn, tp = confusion_matrix(y_test, y_predicted).ravel()
print('The value of true positive is', tp)
print('The value of true negative is', tn)
print('The value of false positive is', fp)
print('The value of false negative is', fn)

# Checking accuracy of model
accuracy = accuracy_score(y_test, y_predicted)
print('The accuracy score of the model is', accuracy)

# Checking sensitivity of model (how good model is at detecting positives)
sensitivity = recall_score(y_test, y_predicted)
print('The sensitivity of the model is', sensitivity)

# The specificity of the model is
specificity = recall_score(y_test, y_predicted, pos_label=0)
print('The specificity of the model is', specificity)

# Value of accuracy is always between sensitivity and specificity

# To make our model more accurate one of the ways is to correct for imbalance
# This is when a dataset has more of one type of data in the dataset than the other
# This results in less accurate predictions as the model learns to predict the higher amount better than lower amount
# For example in our dataset:
print(diabetes_df['diabetes'].value_counts())  # 500 people without and 268 with hence imbalance

# We can even plot the ROC curve for determining threshold
y_pred_prob = mlp.predict_proba(x_test)[:, :]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:, 1])
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC Curve for Diabetes Classifier')
plt.xlabel('False Positivity Rate(1 - Specificity)')
plt.ylabel('True Positive Rate(Sensitivity)')
plt.grid(True)
plt.show()

# Calculating the MAE, MSE and RMSE and R2 score
MAE = mean_absolute_error(y_test, y_predicted)
print('The mean absolute error is', MAE)
MSE = mean_squared_error(y_test, y_predicted)
print('The mean squared error is', MSE)
RMSE = np.sqrt(MSE)
print('The root mean squared error is', RMSE)
R2 = r2_score(y_test, y_predicted)
print('The r2 score is', R2)

# Cross-validation is seeing how our graph performs on new data as opposed to our trained dataset
# Types of cross-validation are k-fold cross validation and Leave one out cross validation
# In k-fold cross validation data is split into parts (k folds) and some taken as test and train dataset
# In leave one out cross validation all but one part is taken as training dataset
# In order to perform cross-validation we must test our training data against our actual dataset for example:
cv_results = cross_validate(mlp, diabetes_df.iloc[:, :-1], diabetes_df.iloc[:, -1], cv=10,
                            scoring=['accuracy', 'precision', 'recall'])
print(cv_results)
print('Accuracy is', cv_results['test_accuracy'].mean())
print('Precision is', cv_results['test_precision'].mean())
print('Recall is', cv_results['test_recall'].mean())
# The code above is for k-cross validation which takes one part for testing and the others for training in the divided
# k-folds

# We can also use Leave one cross validation which is a special type of k-cross validation
cv_results1 = cross_validate(mlp, diabetes_df.iloc[:, :-1], diabetes_df.iloc[:, -1], cv=LeaveOneOut(),
                             scoring=['accuracy'])
print(cv_results1)
print('Accuracy is', cv_results1['test_accuracy'].mean())

# We now move to the part of hyperparameter tuning which is the adjusting of the data to optimise a certain output like
# The precision and the accuracy using something known as GridSearchCV
# By tuning our model using GridSearch we can improve accuracy

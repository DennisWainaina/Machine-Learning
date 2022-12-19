# Multi-class classification is a type of classification where the choices are more than 2
# As opposed to binary classification where the choice is binary yes or no survived or not here outcomes are more than 2
# An example of this is identifying a handwritten digit which is the task today
# Here we will use logistic regression to predict a handwritten digit and compare it to the actual value

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns

# First we will import a database consisting of images of handwritting consisting of 1700+ images of 8 by 8 arrays
# The digits.data is our dataset to be trained
digits = load_digits()
print(dir(digits))
print(digits.data[0])

# We now want to see the image
plt.gray()
plt.matshow(digits.images[0])


# What if we wanted to see mulitple images
for i in range(5):
    plt.gray()
    plt.matshow(digits.images[i])

# What if we wanted to see our target variables
print(digits.target[0:5])

# Training our model becomes
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2,
                                                    random_state=47)
# Seeing if test size actually worked
print(len(x_train))
print(len(x_test))

# Building our model
model = LogisticRegression(max_iter=3000)
model.fit(x_train, y_train)
print(model)

# Seeing accuracy of the model
print(model.score(x_test, y_test))  # Calculates y pred from x test values

# We now want to see how good our model is at predicting by seeing if it predicts an actual value
print(digits.target[42])  # Actual value is 6
# Seeing if our model predicts
print(model.predict([digits.data[42]]))  # Predicted value is the same as actual value so model is accurate

# We now use a confusion matrix to see how many predicted the model actually got right and wrong
y_predicted = model.predict(x_test)
cm = confusion_matrix(y_test, y_predicted)
print(cm)

# Visualising the data
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression

# load the dataset
data = datasets.load_breast_cancer()
X, y = data.data, data.target

#print(f"Shape of X: {X.shape}")
#print(f"Shape of y: {y.shape}")

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# function to calculate accuracy
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

# load the logistic regression model
model = LogisticRegression()

# fit the model
model.fit(X_train, y_train)

# predict with sample data
result = model.predict(X_test)

# check the accuracy of the model
print(f"Accuracy of the Logistic regression model: {accuracy(y_test, result)}. ✅")
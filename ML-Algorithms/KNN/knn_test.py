import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from knn import KNN

# set the color map
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# load the dataset - Sample(Iris Dataset)
dataset = datasets.load_iris()

# get the rows and columns from the dataset
X, y = dataset.data, dataset.target

# split the dataset into train and test samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

#print(f"X_train Shape: {X_train.shape}\n")  # Output: (120, 4) -> 120 rows, 4 columns
#print(f"Single sample from X_train: {X_train[0]}\n") # Output: Single row from X_train

#print(f"Y_train Shape: {y_train.shape}\n")
#print(f"Sample from y_train: {y_train[0]}\n")

# plot the data points
#plt.figure()
#plt.scatter(X[:, 2], X[:, 3], c=y, cmap=cmap, edgecolors='k', s=20)
#plt.savefig("./plot.png")

# create a object for KNN
model = KNN(k=5)

# fit the model with X_train, y_train
model.fit(X_train, y_train)

# model prediction with sample data
results = model.predict(X_test)

# calculate the accuracy
accuracy = np.sum(results == y_test) / len(y_test)

print(f"Accuracy of KNN Model: {accuracy}. ✅")
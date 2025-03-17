import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

# load the dataset
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# plot the dataset
#fig = plt.figure(figsize=(8, 6))
#plt.scatter(X[:, 0], y, color='b', marker='o', s=30)
#plt.savefig("./plot.png")

#print(f"Shape of X sample: {X_train.shape}")
#print(f"Shape of y sample: {y_train.shape}")

# error function
def mse(y_actual, y_predicted):
    return np.mean((y_actual - y_predicted) ** 2)

# create a object for LinearRegression
model = LinearRegression(lr=0.01)

# initialize the data to the model
model.fit(X_train, y_train)

# predict with the sample data
predictions = model.predict(X_test)

# calculate the accurayc (error)
# loss function
loss = mse(y_test, predictions)

print(f"Mean Square Error: {loss}. ✅")

y_predline = model.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_predline, color="black", linewidth=2, label="Prediction")

plt.savefig("Learning_rate_0.01.png")

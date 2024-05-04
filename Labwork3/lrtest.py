from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

X = [[2, -3], [7, 7], [9, 15], [4, 2], [1, -3], [5, 10], [8, 5], [3, -1], [6, 6], [10, 12]]
y = [0, 1, 1, 0, 0, 1, 0, 0, 1, 1]

# Train logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Get the coefficients and intercept
w = [model.intercept_[0], model.coef_[0][0], model.coef_[0][1]]

print (f"f(x) = {w[1]}*X1 + {w[2]}*X2 + {w[0]} = ")

def f(Xi, w):
    return -(w[0] + w[1]*Xi) / w[2]

# Plot the data points
plt.scatter(np.array(X)[:, 0], np.array(X)[:, 1], c=y, cmap=plt.cm.Paired, label='Data')

# Plot the decision boundary
x_values = np.linspace(min(np.array(X)[:, 0]), max(np.array(X)[:, 0]), 100)
plt.plot(x_values, f(x_values, w), label='Decision Boundary')

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Logistic Regression Decision Boundary')
plt.legend()
plt.show()
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

def load_csv(file_path):
    X, y = [], []
    with open(file_path, 'r') as file:
        for line in file:
            row = line.strip().split(',')
            X.append([float(row[0]),float(row[1])])
            y.append(float(row[2]))
    return X, y

X , y = load_csv("Labwork3\loan.csv")
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
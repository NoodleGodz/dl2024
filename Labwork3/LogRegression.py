import math

def f(Xi, w):
    return w[0] + w[1]*Xi[0] + w[2]*Xi[1]

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def diff_w0(Xi, yi, w):
    fx = f(Xi, w)
    y_hat = sigmoid(fx)
    return (y_hat - yi)

def diff_w1(Xi, yi, w):
    fx = f(Xi, w)
    y_hat = sigmoid(fx)
    return (y_hat - yi) * Xi[0]

def diff_w2(Xi, yi, w):
    fx = f(Xi, w)
    y_hat = sigmoid(fx)
    return (y_hat - yi) * Xi[1]

def L(Xi, yi, w):
    y_hat = sigmoid(f(Xi, w))
    return -(yi * math.log(y_hat) + (1 - yi) * math.log(1 - y_hat))

def loss(X, y, w):
    ld = [L(xi, yi, w) for xi, yi in zip(X, y)]
    return sum(ld) / len(ld)

def print_step(time, w, loss):
    print(f"{time}\t", end='')
    for weight in w:
        print(f"{weight:.3f}\t", end='')
    print(f"{loss:.3f}")

def GradientDescent3D(X, y, w, lr, stop):
    step = 0 
    old_loss = loss(X, y, w)
    print_step(step, w, old_loss)
    while True:
        for i in range(len(w)):
            if i == 0:
                dw = sum(diff_w0(X[j], y[j], w) for j in range(len(X))) / len(X)
            elif i == 1:
                dw = sum(diff_w1(X[j], y[j], w) for j in range(len(X))) / len(X)
            else:
                dw = sum(diff_w2(X[j], y[j], w) for j in range(len(X))) / len(X)
            w[i] -= lr * dw

        step += 1
        new_loss = loss(X, y, w)
        print_step(step, w, new_loss)
        
        if abs(old_loss - new_loss) <= stop:
            break
        old_loss = new_loss

    return w
def LogRegression(X, y):
    w = [0.1, 0.1, 0.1]  
    lr = 0.001  
    stop = 0.00001    
    w = GradientDescent3D(X, y, w, lr, stop)
    return w





X = [[2, -3], [7, 7], [9, 15], [4, 2], [1, -3], [5, 10], [8, 5], [3, -1], [1, 6], [10, 12]]
y = [0, 1, 1, 0, 0, 1, 0, 0, 1, 1]


w = LogRegression(X,y)



print (f"f(x) = {w[1]}*X1 + {w[2]}*X2 + {w[0]} = ")



from matplotlib import pyplot as plt
import numpy as np
def f(Xi, w):
    return -(w[0] + w[1]*Xi) / w[2]

# Plot the data points
plt.scatter(np.array(X)[:, 0], np.array(X)[:, 1], c=y, cmap=plt.cm.Paired, label='Data')

# Plot the decision boundary
x_values = np.linspace(min(np.array(X)[:, 0]), max(np.array(X)[:, 0]), 100)

plt.plot(x_values, f(x_values, w), label='Decision Boundary')

print()

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Logistic Regression Decision Boundary')
plt.legend()
plt.show()
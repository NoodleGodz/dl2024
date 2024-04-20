X = [1, 2, 3, 4, 5]
y = [ 14 , 41 , 98 ,197, 350] 

def deri_w3(X, y, w3, w2, w1, w0):

    dw3 = [(xi ** 3 * w3 + xi ** 2 * w2 + xi * w1 + w0 - yi) * xi ** 3 for xi, yi in zip(X, y)]
    dl = sum(dw3) / len(X)
    return dl

def deri_w2(X, y, w3, w2, w1, w0):

    dw2 = [(xi ** 3 * w3 + xi ** 2 * w2 + xi * w1 + w0 - yi) * xi ** 2 for xi, yi in zip(X, y)]
    dl = sum(dw2) / len(X)
    return dl

def deri_w1(X, y, w3, w2, w1, w0):

    dw1 = [(xi ** 3 * w3 + xi ** 2 * w2 + xi * w1 + w0 - yi) * xi for xi, yi in zip(X, y)]
    dl = sum(dw1) / len(X)
    return dl

def deri_w0(X, y, w3, w2, w1, w0):

    dw0 = [xi ** 3 * w3 + xi ** 2 * w2 + xi * w1 + w0 - yi for xi, yi in zip(X, y)]
    dl = sum(dw0) / len(X)
    return dl

def L(X, y, w3, w2, w1, w0):
    y_preds = [w3 * xi ** 3 + w2 * xi ** 2 + w1 * xi + w0 for xi in X]
    l = [(yi - y_pred) ** 2 for yi, y_pred in zip(y, y_preds)]
    lost = sum(l) / (2 * len(y))
    return lost

def print_step(time, w3, w2, w1, w0, lost):
    print(f"{time}\t{w3:.3f}\t{w2:.3f}\t{w1:.3f}\t{w0:.3f}\t{lost:.3f}")

def GradientDescent4D(X, y, w3, w2, w1, w0, L, deri_w3, deri_w2, deri_w1, deri_w0, lr, stop):
    step = 0
    old_loss = L(X, y, w3, w2, w1, w0)
    print_step(step, w3, w2, w1, w0, old_loss)
    
    while True:
        w3 -= lr * deri_w3(X, y, w3, w2, w1, w0)
        w2 -= lr * deri_w2(X, y, w3, w2, w1, w0)
        w1 -= lr * deri_w1(X, y, w3, w2, w1, w0)
        w0 -= lr * deri_w0(X, y, w3, w2, w1, w0)
        
        step += 1
        new_loss = L(X, y, w3, w2, w1, w0)
        print_step(step, w3, w2, w1, w0, new_loss)
        
        if abs(old_loss - new_loss) <= stop:
            break
        old_loss = new_loss
    
    return w3, w2, w1, w0

def PolynomialRegression(X, y):
    w3, w2, w1, w0 = 1, 1, 1, 0
    lr = 0.0001  
    stop = 0.0001  
    w3, w2, w1, w0 = GradientDescent4D(X, y, w3, w2, w1, w0, L, deri_w3, deri_w2, deri_w1, deri_w0, lr, stop)
    print(f"f(x) = {w3:.3f}x^3 + {w2:.3f}x^2 + {w1:.3f}x + {w0:.3f}")
    return w3,w2,w1,w0


w3,w2,w1,w0 = PolynomialRegression(X, y)

import numpy as np
import matplotlib.pyplot as plt


def polynomial_regression_function(x, w3, w2, w1, w0):
    return w3 * x**3 + w2 * x**2 + w1 * x + w0


x_values = np.linspace(min(X), max(X), 100)

y_values = polynomial_regression_function(x_values, w3, w2, w1, w0)


plt.scatter(X, y, color='blue', label='Data Points')

plt.plot(x_values, y_values, color='red', label='Polynomial Regression Curve')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polynomial Regression')
plt.legend()

# Show the plot
plt.show()

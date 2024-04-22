X = [1, 2, 3, 4, 5]
y = [ 14 ,28 , 128 ,197, 320] 

def f(x,w):
    res = 0
    for i,wi in enumerate(w):
        res+= x**i*wi
    return res
    
def deri_wn(X,y,w,deg):
    dwn = [(f(xi,w) - yi ) * xi ** deg for xi, yi in zip(X,y)]
    dl = sum(dwn) / len(X)
    return dl

def L(X, y, w):
    y_preds = [f(xi,w) for xi in X]
    l = [(yi - y_pred) ** 2 for yi, y_pred in zip(y, y_preds)]
    lost = sum(l) / (2 * len(y))
    return lost


def print_step(time, w, lost):
    print(f"{time}\t", end='')
    for i, weight in enumerate(w):
        print(f"{weight:.3f}\t", end='')
    print(f"{lost:.3f}")

def GradientDescentGen(X, y, w, L, deri_wn, lr, stop):
    step = 0 
    old_loss = L(X,y,w)
    while True:
        for i,v in enumerate(w):
            w[i] -= lr * deri_wn(X,y,w,i)

        step +=1
        new_loss = L(X,y,w)     

        print_step(step, w, new_loss)
        
        if abs(old_loss - new_loss) <= stop:
            break
        old_loss = new_loss
      
    return w

def PolynomialRegression(X, y, degree):
    w = [0]
    for i in range(degree):
        w.append(1)
    lr = 0.0000001  
    stop = 0.0001  
    w = GradientDescentGen(X, y, w, L, deri_wn, lr, stop)
    return w


w = PolynomialRegression(X, y,degree=5)
print(w)

import numpy as np
import matplotlib.pyplot as plt

x_range = np.linspace(min(X), max(X), 100)

y_pred = [f(xi,w) for xi in x_range ]

# Plot the original data points
plt.scatter(X, y, color='blue', label='Original data')

# Plot the predicted curve
plt.plot(x_range, y_pred, color='red', label='Predicted curve')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polynomial Regression')
plt.legend()
plt.grid(True)
plt.show()

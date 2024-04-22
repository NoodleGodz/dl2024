import math

from matplotlib import pyplot as plt
import numpy as np

def f(Xi,w):
    #print (f"{w[1]}*{Xi[0]} + {w[2]}*{Xi[1]} + {w[0]} = {w[1]*Xi[0] + w[2]*Xi[1] + w[0]}")
    return w[1]*Xi[0] + w[2]*Xi[1] + w[0]

def sigmond(x):
    return 1 / (1 + math.exp(-x))

def diff_w0(Xi,yi,w):
    fx= f(Xi,w)
    return (-yi/sigmond(fx) + (1-yi)/(1-sigmond(fx)))* sigmond(fx) * (1 - sigmond(fx) )

def diff_w1(Xi,yi,w):
    fx= f(Xi,w)
    return (-yi/sigmond(fx) + (1-yi)/(1-sigmond(fx)))* sigmond(fx) * (1 - sigmond(fx) ) * Xi[0]

def diff_w2(Xi,yi,w):
    fx= f(Xi,w)
    return (-yi/sigmond(fx) + (1-yi)/(1-sigmond(fx)))* sigmond(fx) * (1 - sigmond(fx) ) * Xi[1]

def L(Xi,yi,w):
    #print(f"{Xi} {yi} {w}")
    fx= sigmond(f(Xi,w))
    return -(yi * math.log(fx) + (1-yi)*math.log(1-fx))

def loss(X,y,w):
    ld = [L(xi,yi,w) for xi,yi in zip(X,y)]
    l = sum(ld)/len(ld)
    return l

def print_step(time, w, lost):
    print(f"{time}\t", end='')
    for i, weight in enumerate(w):
        print(f"{weight:.3f}\t", end='')
    print(f"{lost:.3f}")

def GradientDescent3D(X, y, w, L, lr, stop):
    step = 0 
    old_loss = loss(X,y,w)
    print_step(step, w, old_loss)
    while True:

            dwn = [diff_w0(Xi,yi,w) for Xi, yi in zip(X,y)]
            dw0 = sum(dwn) / len(dwn)

            dwn = [diff_w1(Xi,yi,w) for Xi, yi in zip(X,y)]
            dw1 = sum(dwn) / len(dwn)            
            
            dwn = [diff_w2(Xi,yi,w) for Xi, yi in zip(X,y)]
            dw2 = sum(dwn) / len(dwn)          

            w[0] -= lr*dw0
            w[1] -= lr*dw1
            w[2] -= lr*dw2


            step +=1
            new_loss = loss(X,y,w)     

            print_step(step, w, new_loss)
        
            if abs(old_loss - new_loss) <= stop:
                break
            old_loss = new_loss

    return w

def LogRegression(X, y):
    w = [0,1,1]
    lr = 0.00000000001
    stop = 0.00001  
    w = GradientDescent3D(X, y, w, L, lr, stop)
    return w


X = [[2,-3],[7,7],[9,15],[4,2],[1,-3]]
y = [0,1,1,0,0]

w = LogRegression(X,y)


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
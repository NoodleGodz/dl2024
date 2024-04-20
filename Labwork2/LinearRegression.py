

def L(X,y,w1,w0):
    y_preds = []

    for i in X:
        y_preds.append(w1*i+w0)

    l = []
    for i,y_pred in enumerate(y_preds):
        l.append((y[i]-y_pred)**2)
    #average lost
    lost = sum(l)/(2*len(y))
    return lost

def deri_w1(X,y,w1,w0):
    dw1=[]
    for xi,yi in zip(X,y):
        dw1.append((xi*w1 + w0 -yi)*xi)
    #average lost
    dl = sum(dw1)/len(dw1)
    return dl

def deri_w0(X,y,w1,w0):
    dw0=[]
    for xi,yi in zip(X,y):
        dw0.append((xi*w1 + w0 -yi))
    #average lost
    dl = sum(dw0)/len(dw0)
    return dl

def print_step(time,w1,w0, lost):
    print(f"{time}\t{w1:.3f}\t{w0:.3f}\t{lost:.3f}")

def GradientDescent2D(X, y, w1, w0, L, deri_w1, deri_w0, lr, stop):
    step = 0
    old_loss = L(X, y, w1, w0)
    print_step(step, w1, w0, L(X, y, w1, w0))
    
    while True:
        w1 -= lr * deri_w1(X, y, w1, w0)
        w0 -= lr * deri_w0(X, y, w1, w0)
        step += 1
        new_loss = L(X, y, w1, w0)
        print_step(step, w1, w0, new_loss)
        
        if abs(old_loss - new_loss) <= stop:
            break
        old_loss = new_loss
    
    return w1, w0

def LinearRegression(X,y):
    w1 = 1
    w0 = 0
    lr = 0.07
    stop = 0.000001
    w1, w0 = GradientDescent2D(X,y,w1,w0,L,deri_w1,deri_w0,lr,stop)
    print(f"f(x) = {w1:.3f}x + {w0:.3f}")

def load_csv(file_path):
    X, y = [], []
    with open(file_path, 'r') as file:
        for line in file:
            row = line.strip().split(',')
            X.append(float(row[0]))
            y.append(float(row[1]))
    return X, y


file_path = 'house_prices.csv' 
X, y = load_csv(file_path)
print("X:", X)
print("y:", y)

print(f"time\tw1\tw2\tL()")
LinearRegression(X,y)
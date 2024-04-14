
import csv
def f(x,w1,w0):
    return w1*x+w0

def load_csv(fp):
    X = []
    y = []
    with open(fp, 'r') as file:
        reader = csv.reader(file)
        next(reader)  
        for row in reader:
            X.append(float(row[0]))
            y.append(float(row[1]))
    return X, y

def L(X,y,w1,w0):
    y_preds = []
    l = []
    for i in X:
        y_preds.append(f(i,w1,w0))

    for i,y_pred in enumerate(y_preds):
        l.append((y[i]-y_pred)**2)
    
    lost = sum(l)*(1/(2*len(y)))
    return lost,y_preds

def update(X, y, y_preds):
    deri_w1=[]
    deri_w0=[]
    for x, y_true, y_pred in zip(X, y, y_preds):
        deri_w1.append((y_pred - y_true)*x)
        deri_w0.append((y_pred - y_true))

    meanw1 = sum(deri_w1)/len(y)
    meanw0 = sum(deri_w0)/len(y)
    return meanw1,meanw0

def print_step(time,w1,w0, lost):
    print(f"{time}\t{w1:.3f}\t{w0:.3f}\t{lost:.3f}")

def gradient_d(X,y,lr,stop):
    time = 0
    w1 = 1
    w0 = 0
    lost, y_preds = L(X,y,w1,w0)
    while lost>stop:
        print_step(time,w1,w0,lost)
        mw1,mw0 = update(X,y,y_preds)
        w1 = w1 - lr*mw1
        w0 = w0 - lr*mw0
        lost, y_preds = L(X,y,w1,w0)
        time=time +1


lr = 0.15
stop = 0.00001
print(f"time\tw1\tw2\tL()")

X,y = load_csv("Labwork1/house_prices.csv")
gradient_d(X,y,lr,stop)

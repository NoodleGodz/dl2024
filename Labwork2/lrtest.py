import csv
from sklearn.linear_model import LinearRegression

def load_csv(fp):
    X = []
    y = []
    with open(fp, 'r') as file:
        reader = csv.reader(file)
        next(reader)  
        for row in reader:
            X.append([float(row[0])])
            y.append(float(row[1]))
    return X, y

X, y = load_csv("Labwork1\\house_prices.csv")

model = LinearRegression()
model.fit(X, y)
w1 = model.coef_[0]  
w0 = model.intercept_  
print("Slope (w1):", w1)
print("Intercept (w0):", w0)

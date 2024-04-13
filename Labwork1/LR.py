
def f(x):
    return x**2

def f_(x):
    return 2*x



def gradient_d(x,L,stop):
    time = 0
    #print(f"{time}\t{x:.3f}\t{f(x):.3f}")
    while f(x)>stop:
        time = time +1
        x = x - L * f_(x)
        #print(f"{time}\t{x:.3f}\t{f(x):.3f}")
    

x0 = -2
L = 0.1
stop = 0.001
print(f"time\tx\tf(x)")
gradient_d(x0,L,stop)

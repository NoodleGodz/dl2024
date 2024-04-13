
def f(x):
    return x**2

def f_(x):
    return 2*x

def print_step(time,x):
    print(f"{time}\t{x:.3f}\t{f(x):.3f}")
    

def gradient_d(x,L,stop):
    time = 0
    print_step(time,x)
    while f(x)>stop:
        time = time +1
        x = x - L * f_(x)
        print_step(time,x)


x0 = -2
L = 10
stop = 0.001
print(f"time\tx\tf(x)")
gradient_d(x0,L,stop)

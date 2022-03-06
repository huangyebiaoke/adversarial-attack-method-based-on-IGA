import random
import numpy as np

def f_1(x):
    return np.sum(x**2)

def f_2(x):
    return np.sum(np.abs(x))+np.prod(np.abs(x))

def f_3(x):
    y=0.
    for i in range(1,x.shape[0]+1):
        temp=0.
        for j in range(1,i):
            temp+=x[j-1]
        y+=temp**2
    return y


def f_4(x):
    return np.max(np.abs(x))

def f_5(x):
    y=0.
    for i in range(1,x.shape[0]):
        y+=100*(x[i]-x[i-1]**2)**2+(x[i-1]-1)**2
    return y

def f_6(x):
    return np.sum(np.abs(x+.5)**2)

def f_7(x):
    y=0.
    for i in range(1,x.shape[0]+1):
        y+=i*x[i-1]**4
    return y+random.random()

def f_8(x):
    return -np.sum(x*np.sin(np.sqrt(np.abs(x))))

def f_9(x):
    return 10*x.shape[0]+np.sum(x**x.shape[0]-10*np.cos(2*np.pi*x))

def f_10(x):
    d=x.shape[0]
    return -20*np.exp(-.2*np.sqrt(np.sum(x**2)/d))-np.exp(np.sum(np.cos(2*np.pi*x))/d)+20+np.e

def f_11(x):
    y=np.sum(x**2)/4000
    temp=1.
    for i in range(1,x.shape[0]+1):
        temp*=np.cos(x[i-1]/np.sqrt(i))
    return y-temp+1



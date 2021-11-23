import numpy as np
test_num=10
# test functions link: https://en.wikipedia.org/wiki/Test_functions_for_optimization

# # Sphere function
# # search domain: -5~5
# # minimum: f(0,0)=0
# fun_index=1
# x1,x2,y1,y2=-5,5,-5,5
# def fun(x,y):
#     return x**2+y**2

# # Ackley's function
# # search domain: -5~5
# # minimum: f(0,0)=0
# fun_index=2
# x1,x2,y1,y2=-5,5,-5,5
# def fun(x,y):
#     return -20*np.exp(-.2*np.sqrt(.5*(x**2+y**2)))-np.exp(.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y)))+20+np.e

# # Beale function
# # search domain: -4.5~4.5
# # minimum: f(3,0.5)=0
# fun_index=3
# x1,x2,y1,y2=-4.5,4.5,-4.5,4.5
# def fun(x,y):
#     return (1.5-x+x*y)**2+(2.25-x+x*y**2)**2+(2.625-x+x*y**3)**2

# Eggholder function
# search domain: -512~512
# minimum: f(512,404.2319)=-959.6407
fun_index=4
x1,x2,y1,y2=-512,512,-512,512
def fun(x,y):
    return -(y+47)*np.sin(np.sqrt(np.abs(y+x/2+47)))-x*np.sin(np.sqrt(np.abs(x-(y+47))))

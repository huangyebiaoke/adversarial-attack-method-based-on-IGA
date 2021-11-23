from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rcParams['axes.unicode_minus']=False
# import seaborn as sns
import numpy as np
import pandas as pd
from fun import *

# GWO IGA PSO SGA
density=[
    [0.008130,0.040989,0.144532,2.209750],
    [0.044360,0.003750,0.112848,2.026469],
    [2.732986,1.584552,0.276638,3.469773],
    [385.347194,147.823100,284.187803,886.790405],
]

points=[[0,0],[0,0],[3,0.5],[512,404.2319]]
df1=pd.read_csv('./data/'+str(fun_index)+'SGA.csv',index_col=[0])
df2=pd.read_csv('./data/'+str(fun_index)+'PSO.csv',index_col=[0])
df3=pd.read_csv('./data/'+str(fun_index)+'GWO.csv',index_col=[0])
df4=pd.read_csv('./data/'+str(fun_index)+'IGA.csv',index_col=[0])
df1=df1[df1.iter==100]
df2=df2[df2.iter==100]
df3=df3[df3.iter==100]
df4=df4[df4.iter==100]
df=pd.concat([df1,df2,df3,df4],axis=0)

p=[0.001,0.002,0.01,1]
X = np.arange(np.min(df.x)-p[fun_index-1]*10, np.max(df.x)+p[fun_index-1]*10, p[fun_index-1])
Y = np.arange(np.min(df.y)-p[fun_index-1]*10, np.max(df.y)+p[fun_index-1]*10, p[fun_index-1])
X, Y = np.meshgrid(X, Y)
Z = fun(X,Y)
plt.contourf(X,Y,Z)
# cb = plt.colorbar()
# cb.set_label('meters')
# plt.contourf(X,Y,Z,cmap=plt.cm.hot)

plt.axvline(x=points[fun_index-1][0], ls='--', color='grey')
plt.axhline(y=points[fun_index-1][1], ls='--', color='grey')
a=1
plt.scatter(x="x", y="y",data=df1,label='SGA='+format(density[fun_index-1][3], ".2f"),alpha=a)
plt.scatter(x="x", y="y",data=df2,label='PSO='+format(density[fun_index-1][2], ".2f"),alpha=a)
plt.scatter(x="x", y="y",data=df3,label='GWO='+format(density[fun_index-1][0], ".2f"),alpha=a)
plt.scatter(x="x", y="y",data=df4,label='IGA='+format(density[fun_index-1][1], ".2f"),alpha=a)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.tight_layout()
plt.savefig('./images/'+str(fun_index)+'contourf_scatter_best.pdf')
# plt.show()
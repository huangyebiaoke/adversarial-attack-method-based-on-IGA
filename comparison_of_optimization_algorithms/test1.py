from matplotlib import colors, pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rcParams['axes.unicode_minus']=False
import numpy as np
import pandas as pd
from fun import *

# Locally optimal graph
df1=pd.read_csv('./data/'+str(fun_index)+'SGA.csv',index_col=[0])
df2=pd.read_csv('./data/'+str(fun_index)+'PSO.csv',index_col=[0])
df3=pd.read_csv('./data/'+str(fun_index)+'GWO.csv',index_col=[0])
df4=pd.read_csv('./data/'+str(fun_index)+'IGA.csv',index_col=[0])
df1=df1[df1.iter==100]
df2=df2[df2.iter==100]
df3=df3[df3.iter==100]
df4=df4[df4.iter==100]

fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(df1.x, df1.y, df1.z,label='SGA')
ax.scatter(df2.x, df2.y, df2.z,label='PSO')
ax.scatter(df3.x, df3.y, df3.z,label='GWO')
ax.scatter(df4.x, df4.y, df4.z,label='IGA')


points=[[0,0,0],[0,0,0],[3,0.5,0],[512,404.2319,-959.6407]]
ax.scatter(points[fun_index-1][0],points[fun_index-1][1],points[fun_index-1][2],color='gray',alpha=0.5)

df=pd.concat([df1,df2,df3,df4],axis=0)
p=[0.001,0.002,0.01,0.2]
X = np.arange(np.min(df.x), np.max(df.x), p[fun_index-1])
Y = np.arange(np.min(df.y), np.max(df.y), p[fun_index-1])
X, Y = np.meshgrid(X, Y)
Z = fun(X,Y)
ax.plot_surface(X, Y, Z,color='gray',alpha=0.5)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.legend()
plt.show()
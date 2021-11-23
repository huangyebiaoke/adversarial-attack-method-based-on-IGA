from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rcParams['axes.unicode_minus']=False
import seaborn as sns
import numpy as np
import pandas as pd
from fun import *

points=[[0,0],[0,0],[3,0.5],[512,404.2319]]
df1=pd.read_csv('./data/'+str(fun_index)+'SGA.csv',index_col=[0])
df2=pd.read_csv('./data/'+str(fun_index)+'PSO.csv',index_col=[0])
df3=pd.read_csv('./data/'+str(fun_index)+'GWO.csv',index_col=[0])
df4=pd.read_csv('./data/'+str(fun_index)+'IGA.csv',index_col=[0])
df1=df1[df1.iter==100]
df2=df2[df2.iter==100]
df3=df3[df3.iter==100]
df4=df4[df4.iter==100]
df1['algorithm']='SGA'
df2['algorithm']='PSO'
df3['algorithm']='GWO'
df4['algorithm']='IGA'
df=pd.concat([df1,df2,df3,df4],axis=0)

X = np.arange(0, np.max(df.x)+10, 1)
Y = np.arange(0, np.max(df.y)+10, 1)
X, Y = np.meshgrid(X, Y)
Z = fun(X,Y)
# plt.xlim(np.min(df.x), np.max(df.x))
# plt.ylim(np.min(df.y), np.max(df.y))
sns.heatmap(Z,annot=False).invert_yaxis()

plt.axvline(x=points[fun_index-1][0], ls='--', color='grey')
plt.axhline(y=points[fun_index-1][1], ls='--', color='grey')
sns.scatterplot(x="x", y="y",hue='algorithm',data=df)
plt.show()
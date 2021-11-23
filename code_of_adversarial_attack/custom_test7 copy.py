import os
import numpy as np
import matplotlib.pylab as plt
import random
import pandas as pd
plt.rc('font',family='Times New Roman')
def random_color():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color

colors=['#4A1F59', '#3FBC57', '#FD955F', '#8A2D8B', '#15DB4D', '#DD1951', '#ECA5A5', '#33A47F', '#68A462', '#F25DED']
# for i in range(10):
#     colors.append(random_color())
# print(colors)

df=pd.read_csv('./data/CNN/deep_net.csv')

for i in range(10):
    plt.plot(np.arange(0,100,1),df[str(i)],c=colors[i],label=str(i))
plt.xlabel('iteration')
plt.ylabel('confidence')
plt.legend()
plt.subplots_adjust(left=0.08, right=1, top=1, bottom=0.09)
# plt.savefig('CNN_result.svg')
plt.show()
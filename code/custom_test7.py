import os
import numpy as np
import matplotlib.pylab as plt
import random
plt.rc('font',family='Times New Roman')
def random_color():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color

def get_data(root=r'E:\Yangdingming\Desktop\论文相关2\遗传算法在神经网络中的应用\补充材料\NN',filename='0.txt'):
    with open(os.path.join(root,filename)) as f:
        lines = f.read()
    return [float(i) for i in lines.split()]

colors=['#4A1F59', '#3FBC57', '#FD955F', '#8A2D8B', '#15DB4D', '#DD1951', '#ECA5A5', '#33A47F', '#68A462', '#F25DED']
# for i in range(10):
#     colors.append(random_color())
# print(colors)

for i in range(10):
    plt.plot(np.arange(0,13,1),get_data(filename=str(i)+'.txt'),marker='.',c=colors[i],label=str(i))
plt.xlabel('iteration')
plt.ylabel('confidence')
plt.legend()
plt.subplots_adjust(left=0.08, right=1, top=1, bottom=0.09)
plt.savefig('NN_result.svg')
plt.show()
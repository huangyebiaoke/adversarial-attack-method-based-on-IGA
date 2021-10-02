import os
import numpy as np
import matplotlib.pylab as plt
plt.rc('font',family='Times New Roman')


def get_data(root=r'E:\Yangdingming\Desktop\论文相关2\遗传算法在神经网络中的应用\补充材料\zero_init_population',filename='GA.txt'):
    with open(os.path.join(root,filename)) as f:
        lines = f.read()
    return [float(i) for i in lines.split()]

plt.plot(np.arange(0,13,1),get_data(),marker='^',c='red',label='GA')
plt.plot(np.arange(0,13,1),get_data(filename='GA_A1.txt'),marker='o',c='orange',label='GA with A1')
plt.plot(np.arange(0,13,1),get_data(filename='GA_A1_A2.txt'),marker='s',c='green',label='GA with A1+A2')
plt.xlabel('iteration')
plt.ylabel('confidence')
# plt.xlim(1.0391408670466593e+113,5.9391408670466593e+113)
plt.legend()
plt.subplots_adjust(left=0.08, right=1, top=1, bottom=0.09)
# plt.savefig('zero_GA.svg')
plt.show()
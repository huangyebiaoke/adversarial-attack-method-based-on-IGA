from fun import *
import numpy as np
import pandas as pd
import timeit

# 目标函数定义
def ras(x):
    return fun(x[0], x[1])


# 参数初始化
w = 1.0
c1 = 1.49445
c2 = 1.49445

maxgen = 101   # 进化次数
sizepop = 50   # 种群规模

# 粒子速度和位置的范围
Vmax = 1
Vmin = -1

l1=[]
l2=[]
for test_n in range(test_num):
    # x和y的取值
    # 产生初始粒子和速度
    pop = x2 * np.random.uniform(-1, 1, (2, sizepop))
    # print('pop',pop)
    v = np.random.uniform(-1, 1, (2, sizepop))

    fitness = ras(pop)             # 计算适应度
    i = np.argmin(fitness)      # 找最好的个体
    gbest = pop                    # 记录个体最优位置
    zbest = pop[:, i]              # 记录群体最优位置
    fitnessgbest = fitness        # 个体最佳适应度值
    fitnesszbest = fitness[i]      # 全局最佳适应度值

    # 迭代寻优
    t = 0
    record = np.zeros(maxgen)
    # print('iter,x,y,z')
    while t < maxgen:
        start=timeit.default_timer()
        # 速度更新
        v = w * v + c1 * np.random.random() * (gbest - pop) + c2 * \
            np.random.random() * (zbest.reshape(2, 1) - pop)
        v[v > Vmax] = Vmax     # 限制速度
        v[v < Vmin] = Vmin
        # 位置更新
        pop = pop + 0.5 * v
        pop[pop > x2] = x2  # 限制位置
        pop[pop < x1] = x1
        '''
        # 自适应变异
        p = np.random.random()             # 随机生成一个0~1内的数
        if p > 0.8:                          # 如果这个数落在变异概率区间内，则进行变异处理
            k = np.random.randint(0,2)     # 在[0,2)之间随机选一个整数
            pop[:,k] = np.random.random()  # 在选定的位置进行变异 
        '''
        # 计算适应度值
        fitness = ras(pop)
        # 个体最优位置更新
        index = fitness < fitnessgbest
        fitnessgbest[index] = fitness[index]
        gbest[:, index] = pop[:, index]
        # 群体最优更新
        j = np.argmin(fitness)
        if fitness[j] < fitnesszbest:
            zbest = pop[:, j]
            fitnesszbest = fitness[j]
        # print(zbest)
        end=timeit.default_timer()
        l1.append({'test_n':test_n,'iter':t,'x':zbest[0],'y':zbest[1],'z':fitnesszbest,'time':end-start})
        # print(t, ',', zbest[0], ',', zbest[1], ',', fitnesszbest)
        record[t] = fitnesszbest  # 记录群体最优位置的变化
        t = t + 1
    for i in range(len(fitness)):
        xy=pop[:, i]
        l2.append({'test_n':test_n,'x':xy[0],'y':xy[1],'z':fitness[i]})
    print('test_n:',test_n)
pd.DataFrame(l1).to_csv('./data/'+str(fun_index)+'PSO.csv')
pd.DataFrame(l2).to_csv('./data/'+str(fun_index)+'PSO_final_population.csv')
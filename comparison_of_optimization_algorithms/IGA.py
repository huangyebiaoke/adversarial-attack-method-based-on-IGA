import math
import numpy as np
import random
import pandas as pd
import timeit
from fun import *

gen_len=30
population_size=50
mutate_rate=.05
iteration=101
p=math.pow(2,gen_len/2)

def decode(person):
    x=person[0:15]
    y=person[15:30]
    # 0~2^15
    # print(''.join(str(i) for i in x))
    x=int(''.join(str(i) for i in x),2)
    y=int(''.join(str(i) for i in y),2)
    # 0~2^15-->-1~1
    x = x*(x2-x1)/p+x1
    y = y*(y2-y1)/p+y1
    return x,y


def fitness(person):
    x,y=decode(person)
    return -fun(x,y)



def crossover(father,mother):
    best_fitness=float('-inf')
    child=np.zeros(30)
    # print('father_len',len(father))
    for i in range(father.size):
        current_child=np.zeros(30)
        current_child=np.append(father[0:i],mother[i:])
        # print('current_child:',current_child.shape)
        current_fitness=fitness(current_child)
        if current_fitness>best_fitness:
            best_fitness=current_fitness
            # change:add copy
            child=current_child.copy()
    return child

def muteChild(child):
    temp_child=child
    for i in range(temp_child.size):
        if(random.random()<mutate_rate):
            temp_child[i] = 0. if temp_child[i]==1 else 1.
    return temp_child



def getPopulation():
    population=np.zeros((population_size, gen_len),dtype=np.int)
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            if (random.random()>.5):
                population[i][j]=1
    return population

def selectParent(fitness_list):
    rand=random.random()
    current_fitness_sum=.0
    location=0
    fitness_sum=np.sum(fitness_list)
    for i in range(fitness_list.size):
        current_fitness_sum+=(fitness_list[i]/fitness_sum)
        if(rand<current_fitness_sum):
            location=i
            break
    return location



if __name__=='__main__':
    l1=[]
    l2=[]
    for test_n in range(test_num):
        fitness_list=np.zeros(population_size)
        best_fitness=float('-inf')
        best_person=np.zeros(gen_len)

        population=getPopulation()
        # print('iter,x,y,z')
        for iter in range(iteration):
            start=timeit.default_timer()
            for person in population:
                current_fitness=fitness(person)
                if current_fitness>best_fitness:
                    best_fitness=current_fitness
                    best_person=person.copy()
                # print('iteration:',iter,'fitness:',best_fitness)
                for i in range(population_size):
                    fitness_list[i]=fitness(population[i])
                    best_father=population[selectParent(fitness_list)]
                    best_mother=population[selectParent(fitness_list)]
                    child_from_best_parent=crossover(best_father, best_mother)
                    child_from_best_parent=muteChild(child_from_best_parent)
                    population[i]=child_from_best_parent
            x,y=decode(best_person)
            # print(iter,',',x,',',y,',',-best_fitness)
            end=timeit.default_timer()
            l1.append({'test_n':test_n,'iter':iter,'x':x,'y':y,'z':-best_fitness,'time':end-start})
        for i in range(population_size):
            xx,yy=decode(population[i])
            l2.append({'test_n':test_n,'x':xx,'y':yy,'z':fun(xx,yy)})
        # print('min_value:',-best_fitness)
        print('test_n:',test_n)
    pd.DataFrame(l1).to_csv('./data/'+str(fun_index)+'IGA.csv')
    pd.DataFrame(l2).to_csv('./data/'+str(fun_index)+'IGA_final_population.csv')

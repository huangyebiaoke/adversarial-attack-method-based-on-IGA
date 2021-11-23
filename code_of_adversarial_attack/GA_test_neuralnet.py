import sys, os
sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
import math,random
import cv2


# gene_length=4
iteration=10
population_size=1000
test_label=np.array([1,0,0,0,0,0,0,0,0,0])
test_label=np.expand_dims(test_label,axis=0)

# def decode(person):
#     image=np.zeros(784)
#     for i in range(image.size):
#         x=person[i*gene_length:(i+1)*gene_length]
#         num=int(''.join(map(lambda x: str(int(x)), x)),2)
#         image[i]=num/math.pow(2,gene_length)
#         return image

def calculateLoss(person,loss):
    person=np.expand_dims(person,axis=0)
    return loss(person,test_label)


# def calculateBestLoss(population,loss):
#     loss=10000.
#     for i in range(population.shape[0]):
#         current_loss=calculateLoss(population[i:i+1], loss)
#         if current_loss<loss:
#             loss=current_loss
#     return loss

def selectParent(loss_list):
    rand=random.random()
    current_loss_sum=.0
    location=0
    loss_sum=np.sum(loss_list)
    for i in range(loss_list.size):
        current_loss_sum+=(loss_list[i]/loss_sum)
        if(rand<current_loss_sum):
            location=i
            break
    return location

# under single cross, best loss
def crossover(father,mother,loss):
    best_loss=10000.
    child=np.zeros(784)
    # print('father_len',len(father))
    for i in range(father.size):
        current_child=np.zeros(784)
        current_child=np.append(father[0:i],mother[i:])
        # print('current_child:',current_child.shape)
        current_loss=calculateLoss(current_child, loss)
        if(current_loss<best_loss):
            best_loss=current_loss
            child=current_child
    return child

# under single cross, randomly loss
def crossover2(father,mother):
    location=int(random.random()*father.size)
    return np.append(father[0:location],mother[location:])

# under muti-cross, randomly loss
def crossover3(father,mother):
    child=[]
    for i in range(784):
        location=int(random.random()*gene_length)
        child.append(father[i*gene_length:i*gene_length+location]+mother[i*gene_length+location:(i+1)*gene_length])
    return child

# under muti-cross, best loss in specital loop
def crossover4(father,mother,loss):
    _loop=100
    loss=10000.
    child=[]
    for l in range(_loop):
        current_child=[]
        for i in range(784):
            location=int(random.random()*gene_length)
            current_child.append(father[i*gene_length:i*gene_length+location]+mother[i*gene_length+location:(i+1)*gene_length])
        current_loss=loss(child)
        if(current_loss<loss):
            loss=current_loss
            child=current_child
    return child

# all pixel to mutate
def muteChild(child):
    _mutate_rate=.05
    temp_child=child
    for i in range(temp_child.size):
        if(random.random()<_mutate_rate):
            temp_child[i] = 0. if temp_child[i]==1 else 1.
    return temp_child


# randomly single pixel to mutate
def muteChild2(child):
    _mutate_rate=.2
    temp_child=child
    if(random.random()<_mutate_rate):
        index=int(random.random()*len(temp_child))
        temp_child[index]= 0 if temp_child[index]==1 else 1
    return temp_child

def getPopulation():
    population=np.zeros((population_size, 784),dtype=np.float32)
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            if (random.random()>.5):
                population[i][j]=1
    return population



if __name__=='__main__':
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    network.load_params()
    print('train_loss:',network.loss(x_train[1:2],t_train[1:2]),'test_loss:',network.loss(x_test[0:1],t_test[0:1]))

    population=getPopulation()
    # print('population.shape:',population.shape)
    loss_list=np.zeros(population_size)
    best_loss=10000.
    best_person=np.zeros(784)
    for iter in range(iteration):
        for person in population:
            current_loss=calculateLoss(person, network.loss)
            if current_loss<best_loss:
                best_loss=current_loss
                # 坑，loss远大于best loss的bug已解决
                best_person=person.copy()
        #     print('calculateLoss1:',calculateLoss(person, network.loss))
        print('iteration:',iter,'loss:',best_loss)
        for i in range(population_size):
            # print('i:',i)
            loss_list[i]=calculateLoss(population[i], network.loss)
            best_father=population[selectParent(loss_list)]
            best_mother=population[selectParent(loss_list)]
            # print(best_father)
            child_from_best_parent=crossover(best_father, best_mother, network.loss)
            child_from_best_parent=muteChild(child_from_best_parent)
            population[i]=child_from_best_parent
    print('best_loss:',best_loss)
    # bug:最终测试结果的loss远大于best_loss
    print('test_loss:',calculateLoss(best_person,network.loss))
    image=best_person.reshape(28,28)
    cv2.imshow("winname", image)
    cv2.waitKey(0)




    # test_img=np.random.rand(784)
    # test_img=np.expand_dims(test_img,axis=0)
    # # print(x_test[2:3].shape,test_img.shape)

    # # test_label=[1,0,0,0,0,0,0,0,0,0]
    # # test_label=np.expand_dims(test_label,axis=0)
    # # print(t_test[2:3].shape,test_label.shape)

    # # loss = network.loss(test_img,test_label)
    # # 0.10070259023403888 0.07218919336829734
    # loss = network.loss(test_img,test_label)
    # # test_acc = network.accuracy(test_img,test_label)
    # print(loss)
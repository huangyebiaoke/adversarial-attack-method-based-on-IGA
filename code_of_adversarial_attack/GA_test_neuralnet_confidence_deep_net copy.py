import sys, os
sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from ch08.deep_convnet import DeepConvNet
import math,random
import cv2
from common.functions import softmax


# gene_length=4
iteration=1000
population_size=100
test_label=1

def calculateConfidence(person,predict):
    person=person.reshape(28,28)
    person=np.expand_dims(person,axis=0)
    person=np.expand_dims(person,axis=0)
    # print(person.shape)
    confidence=softmax(predict(person))[0,test_label]
    return confidence

# def calculateConfidence(person,label,predict):
#     person=np.expand_dims(person,axis=0)
#     confidence=softmax(predict(person))[0,np.argmax(label)]
#     return confidence


def selectParent(confidence_list):
    rand=random.random()
    current_confidence_sum=.0
    location=0
    confidence_sum=np.sum(confidence_list)
    for i in range(confidence_list.size):
        current_confidence_sum+=(confidence_list[i]/confidence_sum)
        if(rand<current_confidence_sum):
            location=i
            break
    return location

# under single cross, best loss
def crossover(father,mother,predict):
    best_confidence=-10000.
    child=np.zeros(784)
    # print('father_len',len(father))
    for i in range(father.size):
        current_child=np.zeros(784)
        current_child=np.append(father[0:i],mother[i:])
        # print('current_child:',current_child.shape)
        current_confidence=calculateConfidence(current_child, predict)
        if(current_confidence>best_confidence):
            best_confidence=current_confidence
            # change:添加copy
            child=current_child.copy()
    return child

# under single cross, randomly loss
def crossover2(father,mother,predict):
    confidence=-10000.
    child=np.zeros(father.size)
    for _ in range(10):
        location=int(random.random()*father.size)
        current_child=np.append(father[:location],mother[location:])
        current_confidence=calculateConfidence(current_child, predict)
        if(current_confidence>confidence):
            confidence=current_confidence
            # print('current_confidence in crossover2:',current_confidence)
            child=current_child.copy()
    return child

# under muti-cross, randomly loss
def crossover3(father,mother):
    child=np.zeros(784)
    for i in range(father.size):
        child[i]=father[i] if random.random()>.5 else mother[i]
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
    _mutate_rate=.5
    temp_child=child
    if(random.random()<_mutate_rate):
        index=int(random.random()*temp_child.size)
        temp_child[index]= 0 if temp_child[index]==1 else 1
    return temp_child

def getPopulation():
    population=np.zeros((population_size, 784),dtype=np.float32)
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    j=0
    for i in range(t_test.shape[0]):
        if np.argmax(t_test[i:i+1])==test_label:
            population[j]=x_test[i].copy()
            j+=1
            if(j==population.shape[0]-1):
                break
    return population



if __name__=='__main__':
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
    network = DeepConvNet()
    network.load_params("./ch08/deep_convnet_params.pkl")

    # print(calculateConfidence(x_train[3:4],t_train[3:4],network.predict))
    # print(t_train[3:4])
    # cv2.imshow("winname", x_train[3].reshape(28,28))
    # cv2.waitKey(0)

    population=getPopulation()
    confidence_list=np.zeros(population_size)
    best_confidence=-10000.
    best_person=np.zeros(784)
    for iter in range(iteration):
        for person in population:
            current_confidence=calculateConfidence(person, network.predict)
            # print('current_confidence:',current_confidence)
            if current_confidence>best_confidence:
                best_confidence=current_confidence
                best_person=person.copy()
        print('iteration:',iter,'confidence:',best_confidence)
        cv2.imwrite('./exp_images_deep_net/'+str(iter)+'.jpg',best_person.reshape(28,28)*255)
        for i in range(population_size):
            confidence_list[i]=calculateConfidence(population[i], network.predict)
            best_father=population[selectParent(confidence_list)]
            best_mother=population[selectParent(confidence_list)]
            child_from_best_parent=crossover2(best_father,best_mother,network.predict)
            child_from_best_parent=muteChild2(child_from_best_parent)
            population[i]=child_from_best_parent
    print('best_confidence:',best_confidence)
    print('test_confidence:',calculateConfidence(best_person,network.predict))
    # image=best_person.reshape(28,28)
    # cv2.imshow("winname", image)
    # cv2.waitKey(0)
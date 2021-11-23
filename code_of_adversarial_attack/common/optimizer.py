# coding: utf-8
import numpy as np
import random
import copy
class SGD:

    """随机梯度下降法（Stochastic Gradient Descent）"""

    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        for key in params.keys():
            # print(key,params[key].shape,params[key].ndim)
            """
            W1 (784, 50)
            b1 (50,)
            W2 (50, 10)
            b2 (10,)
            """
            params[key] -= self.lr * grads[key]

    # def update(self, params,grads, loss,x_batch,t_batch):
    #     # for i in range(10):
    #     for key in ('W1','W2'):
    #         params[key] -= random.random()*np.random.randn(params[key].shape[0], params[key].shape[1])
    #     print('update:',loss(x_batch,t_batch))
class GA:
    # Genetic Algorithm
    def __init__(self,gene_len=15,iteration=30,population_size=10,min_x=0,max_x=1):
        self.gene_len=gene_len
        self.iteration=iteration
        self.population_size=population_size
        self.min_x=min_x
        self.max_x=max_x

    def _get_population(self,params):
        population={}
        population.update(params)
        for key in population.keys():
            population[key]=np.expand_dims(population[key],axis=0).repeat(self.population_size,axis=0)
        # init the population with random number
        for i in range(self.population_size):
            population['W1'][i]=random.random()*np.random.randn(population['W1'].shape[1], population['W1'].shape[2])
            population['W2'][i]=random.random()*np.random.randn(population['W2'].shape[1], population['W2'].shape[2])
        return population
    
    def _get_population2(self,params):
        population={}
        for i in range(self.population_size):
            person={}
            for key,val in params.items():
                person[key]=np.zeros_like(val)
            population[i]=person
            # for key in population[i].keys():
                # population[i][key]+=random.random()
        population[0]['W1']=np.zeros_like(population[0]['W1'])
        population[0]['W2']=np.zeros_like(population[0]['W2'])
        population[1]['W1']=np.random.randn(population[1]['W2'].shape[0], population[1]['W2'].shape[1])
        population[1]['W2']=np.random.randn(population[1]['W2'].shape[0], population[1]['W2'].shape[1])
        return population

    def update(self,params,gradient,loss,x_batch,t_batch):
        generation=28
        current_fitness=loss(x_batch,t_batch)
        population=self._get_population2(
            
        )
        # W1 (10, 784, 50)
        # b1 (10, 50)
        # W2 (10, 50, 10)
        # b2 (10, 10)
        best_fitness=-100.
        best_person=''
        # print(population[0]['W1'][0],population[1]['W1'][0])
        while generation!=self.iteration:
            for i in range(self.population_size):
                # grad=gradient(x_batch,t_batch)
                params.update(population[i])
                # for key,val in population[i].items():
                #     # print(params[key][0])
                #     params[key]=val
                #     # print(params[key][0])
                current_fitness=loss(x_batch,t_batch)
                # 改变单层全连接层
                print('generation:',generation,'i:',i,'fitness:',loss(x_batch,t_batch),population[i]['W1'][0][0],params['W1'][0][0])


            generation+=1






class Momentum:

    """Momentum SGD"""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():                                
                self.v[key] = np.zeros_like(val)
                
        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key] 
            params[key] += self.v[key]


class Nesterov:

    """Nesterov's Accelerated Gradient (http://arxiv.org/abs/1212.0901)"""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.v[key] *= self.momentum
            self.v[key] -= self.lr * grads[key]
            params[key] += self.momentum * self.momentum * self.v[key]
            params[key] -= (1 + self.momentum) * self.lr * grads[key]


class AdaGrad:

    """AdaGrad"""

    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class RMSprop:

    """RMSprop"""

    def __init__(self, lr=0.01, decay_rate = 0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Adam:

    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            
            #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)

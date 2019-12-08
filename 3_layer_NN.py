# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 17:09:34 2019

@author: zaneh
"""
import numpy as np
import matplotlib.pyplot as plt

nn_architecture = [
    {"layer_size": 3, "activation": "none"}, 
    {"layer_size": 4, "activation": "sigmoid"},
    {"layer_size": 4, "activation": "sigmoid"},
    {"layer_size": 4, "activation": "sigmoid"},
    {"layer_size": 1, "activation": "sigmoid"}
]

def init_weights(nn_architecture, seed = 4):
    """
    nn_architecture: description of layers and neurons
    @ret parameters: dictionary where parameters[wi or bi] gives a np array corresponding to the i-th layers weight or biases
    """
    np.random.seed(seed)
    # python dictionary containing our parameters "W1", "b1", ..., "WL", "bL"
    parameters = {}
    number_of_layers = len(nn_architecture)
    for i in range(1, number_of_layers):#for weight layers 1-4
        parameters['W' + str(i)] = np.random.randn(nn_architecture[i]["layer_size"], nn_architecture[i-1]["layer_size"]) * 0.01
        parameters['b' + str(i)] = np.zeros((nn_architecture[i]["layer_size"], 1))
    return parameters        

def sigmoid(Z):
    S = 1 / (1 + np.exp(-Z))
    return S

def sigmoid_derivative(x):
    return x * (1.0 - x)


def feedForward(X, parameters, nn_architecture):
    forward_cache = {}
    A = X
    num_layers = len(nn_architecture)
    
    #iterate num_layers - 1 number of times calculating activations
    for i in range(1,num_layers):
        A_prev = A 
        #weights for between layer i-1 and i
        W = parameters['W' + str(i)] 
        #biases for neurons in layer i
        b = parameters['b' + str(i)]
        #check which fxn we're using
        activation = nn_architecture[i]["activation"]
        #do the math
        Z, A = linear_activation_forward(A_prev, W, b, activation)
        #save for later calcs in backprop stage
        forward_cache['Z' + str(i)] = Z
        forward_cache['A' + str(i)] = A
    return forward_cache
    
def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z = np.dot(W, A_prev.T)
        A = sigmoid(Z)
    elif activation == "relu":
        pass
        #not implemented yet
    return Z, A

def backProp(forward_cache, X, Y, parameters, nn_architecture):
    num_layers = len(nn_architecture)
    cost = [0 for i in range(1, num_layers + 1)] #first entry will never be set below
    for i in reversed(range(1, num_layers)):
        if i == num_layers - 1: #no previous cost[i] to include
            cost[i] =  np.dot(forward_cache['A'+str(i-1)], (2 * (Y - forward_cache['A' + str(i)])* sigmoid_derivative(forward_cache['A' + str(i)])).T)
        elif i-1 != 0:
            cost[i] = np.dot(forward_cache['A'+str(i-1)], cost[i+1]  * parameters['W' + str(i+1)] * sigmoid_derivative(forward_cache['A' + str(i)]))
        else: #need to use X
            cost[i] = np.dot(X.T, cost[i+1]  * parameters['W' + str(i+1)] * sigmoid_derivative(forward_cache['A' + str(i)]))
    #update weights
    for i in range(1, num_layers):
        parameters['W' + str(i)] += cost[i].T
    
    return parameters


def run(X, Y, nn_architecture, learning_rate = 0.0075, num_iterations = 30000, print_cost=False):
    np.random.seed(1)  
    # Parameters initialization.
    parameters = init_weights(nn_architecture)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        forward_cache = feedForward(X, parameters, nn_architecture)
        parameters = backProp(forward_cache, X, Y, parameters, nn_architecture)
    return forward_cache

def main():
    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
    #Y = np.array([[0],[1],[1],[0]])
    Y = np.array([1,1,1,0])
    result = run(X, Y, nn_architecture)
    print(result)

if __name__ == '__main__':
    main()

        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
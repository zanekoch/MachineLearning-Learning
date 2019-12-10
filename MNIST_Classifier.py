# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 09:49:53 2019

@author: zaneh
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from keras.utils import to_categorical
import pickle

#define activation functions
#for hidden layers
def relu(p):
    return np.maximum(0, p)
#output layer
def softmax(u):
    return np.exp(u) / np.sum(np.exp(u), axis=0, keepdims=True)
#derivate of relu for backprop
def drelu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def initialize_parameters(layer_dims):
    '''
    W1 = 50 x 784
    W2 = W3 = 50 x 50
    W4 = 10 x 50
    b1 = b2 = b3 = b4 = 50 x 1
    '''
    parameters = {}
    L = len(layer_dims)
    for i in range(1,L):
        #random small weights
        parameters["W" +str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1]) * (np.sqrt(2 / layer_dims[i - 1]))
        #biases to 0
        parameters["b" + str(i)] = np.zeros((layer_dims[i], 1))
    return parameters

def forward_prop(parameters, X_train, activation):
    outputs = {}
    #do input layer because its input is X, not previous layers activation
    outputs["Z" + str(1)] = np.dot(parameters["W1"], X_train) + parameters["b1"]
    activation["A" + str(1)] = relu(outputs["Z" + str(1)])
    #loop across hidden
    for i in range(2, 4):
        outputs["Z" + str(i)] = np.dot(parameters["W" +str(i)], activation["A" + str(i-1)]) + parameters["b" + str(i)]
        activation["A" + str(i)] = relu(outputs["Z" + str(i)])
    #do output layer seperately bc softmax
    outputs["Z4"] = np.dot(parameters["W4"], activation["A3"]) + parameters["b4"]
    activation["A4"] = softmax(outputs["Z4"])
    return outputs, activation

def compute_cost(activation, Y_train, m):
    '''
    use cross entropy loss to compute cost
    '''
    loss = - np.sum((Y_train * np.log(activation["A4"])), axis=0, keepdims=True)
    cost = np.sum(loss, axis=1) / m
    return cost

def grad_re(parameters, outputs, activation, Y_train, X_train, m):
    '''
    compute the amount to change each weight and bias (dW#, db#) using gradient descent
    '''
    grad_reg = {}
    #do output layer
    grad_reg["dZ4"] = (activation["A4"] - Y_train) / m
    #do hidden layers
    for i in reversed(range(1, 4)): #3,2,1
        grad_reg["dA" + str(i)] = np.dot(parameters["W" + str(i + 1)].T, grad_reg["dZ" + str(i+1)])
        grad_reg["dZ" + str(i)] = grad_reg["dA" + str(i)] * drelu(outputs["Z" + str(i)])
    #do input layer
    grad_reg["dW1"] = np.dot(grad_reg["dZ1"], X_train.T)
    grad_reg["db1"] = np.sum(grad_reg["dZ1"], axis=1, keepdims=True)
    for i in range(2,5):
        grad_reg["dW" + str(i)] = np.dot(grad_reg["dZ" + str(i)], activation["A" + str(i-1)].T)
        grad_reg["db" + str(i)] = np.sum(grad_reg["dZ" + str(i)], axis=1, keepdims=True)
    return parameters, outputs, activation, grad_reg

def learning(grad_reg, parameters, learning_rate=0.005):
    '''
    update parameters using gradient descent
    values computed in grad_re()
    '''
    for i in range(1, 5):
        parameters["W" + str(i)] -= (learning_rate * grad_reg["dW" + str(i)])
        parameters["b" + str(i)] -= (learning_rate * grad_reg["db" + str(i)])
    return parameters

def run(layer_dims, activation, Y_train, X_train, m, num_iterations=1000):
    costs = []
    parameters = initialize_parameters(layer_dims)
    for i in range(0, num_iterations):
        outputs, activation = forward_prop(parameters, X_train, activation)
        cost = compute_cost(activation, Y_train, m)
        parameters, outputs, activation, grad_reg = grad_re(parameters, outputs, activation, Y_train, X_train, m)
        parameters = learning(grad_reg, parameters)
        if i % 20 == 0:
            print("%f percent done" % (i/10))
        if i % 100 == 0:
            costs.append(cost)
            print("Cost after iteration %i: %f" % (i, cost))
    return costs, parameters

def test(parameters, X_test, Y_test):
    activation = {}
    #see what activations the model outputs using the test set
    #X_test is 784 x 10,000
    for i in range(0, 10000):
        activation = {}
        #get to activation of each of 10,000 X_train samples
        outputs, activation = forward_prop(parameters, X_test[0:,i], activation)
        #compare predicted value to actual value
        result = activation["A4"]
        print(result)


def main():
    #load data
    (X_train_orig, Y_train_orig), (X_test_orig, Y_test_orig) = mnist.load_data()
    #reformat from weird (60000,) and (10000,) shapes
    Y_tr_resh = Y_train_orig.reshape(60000, 1)
    Y_te_resh = Y_test_orig.reshape(10000, 1)
    #one hot encode to the 10 number classes
    Y_tr_T = to_categorical(Y_tr_resh, num_classes=10)
    Y_te_T = to_categorical(Y_te_resh, num_classes=10)
    #transpose data
    Y_train = Y_tr_T.T
    Y_test = Y_te_T.T
    #FLATTEN the 28 x 28 pixel images into 784 pixels
    #x_train_orig.shape is 60000 x 28 x 28 this turns it into 784, 60000
    X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
    X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
    #normalize pixel intensity to between 0 and 1
    X_train = X_train_flatten / 255
    X_test = X_test_flatten / 255
    #setup 4 layer network with 784 input neurons, 3 layers of 50, and an output layer of 10
    layer_dims = [X_train.shape[0], 50, 50, 50, 10]
    m = X_train.shape[1]

    '''
    to make sure weight and biase matrices are right shape
    for l in range(1, 5):
        print("W" + str(l) + " = " + str(parameters["W" + str(l)]))
        print("W" + str(l) + "shape" + " = " + str(parameters["W" + str(l)].shape))
        print("b" + str(l) + " = " + str(parameters["b" + str(l)]))
        print("b" + str(l) + "shape" + " = " + str(parameters["b" + str(l)].shape))
    print(len(parameters))
    parameters = initialize_parameters(layer_dims)
    outputs, activation = forward_prop(parameters, X_train, activation)
    parameters, outputs, activation, grad_reg = grad_re(parameters, outputs, activation, Y_train, X_train, m)
    parameters = learning(grad_reg, parameters)
    '''
    #train the model getting parameters
    activation = {}
    #costs, parameters = run(layer_dims, activation, Y_train, X_train, m)
    #pickle.dump(parameters, open("model.p", "wb+"))
    parameters = pickle.load(open("model.p", "rb"))
    test(parameters, X_test, Y_test)


if __name__ == '__main__':
    main()

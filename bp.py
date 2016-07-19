# coding: utf-8

import numpy as np
import random 
import mnist_loader

net = []
num_layers = 0
net_sizes  = []
weights = []
biases = []

#sizes is a list which contains the number of neural in every layer
# ex: [700, 30 , 10]
def network(sizes):
    global num_layers,weights,biases,net_sizes
    num_layers = len(sizes)
    net_sizes = sizes
    biases = [np.random.randn(x,1) for x in sizes[1:]]
    weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

def forward(a):
    global biases,weights
    for b, w in zip(biases, weights):
        a = sigmoid(np.dot(w, a)+b)
    return a
    


def SGD(training_data, epochs, mini_batch_size, eta, test_data):

    if test_data: n_test = len(test_data)
    train_len = len(training_data)
    for j in xrange(epochs):
        random.shuffle(training_data)
        
        mini_batches = [
            training_data[k:k+mini_batch_size]
            for k in xrange(0, train_len, mini_batch_size)]

        for mini_batch in mini_batches:
           update(mini_batch, eta)

        if test_data :
            print "Epoch {0}: {1} / {2}".format(
                j, evaluate(test_data), n_test)
        else:
            print "no test data"
                

def evaluate(test_data):
    test_results = [(np.argmax(forward(x)), y)
                    for (x, y) in test_data]
    return sum(int(x == y) for (x, y) in test_results)
    


def update(mini_batch_data,eta):
    global biases,weights
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_w = [np.zeros(w.shape) for w in weights]
    for x,y in mini_batch_data:
        delta_nabla_b, delta_nabla_w = backprop(x,y)
        nabla_b = [nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
        nabla_w = [nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]

    weights = [w - (eta/len(mini_batch_data))* nw 
                        for w,nw in zip(weights,nabla_w)]
    biases = [b - (eta/len(mini_batch_data)*nb) 
                        for b,nb in zip(biases,nabla_b)]

def backprop(x,y):
    global biases,weights,num_layers
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_w = [np.zeros(w.shape) for w in weights]
    #feedforward
    activation = x
    # list to store all the activations, layer by layer
    activations = [x]
    zs = [] # list to store all the z vectors, layer by layer
    for b,w in zip(biases,weights):
        z = np.dot(w, activation)+b
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)

    # backward pass
    delta = cost_derivative(activations[-1],y) * \
            sigmoid_prime(zs[-1])

    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta ,activations[-2].transpose())

    for l in xrange(2,num_layers):
        z = zs[-l]
        sp = sigmoid_prime(z)
        delta = np.dot(weights[-l+1].transpose(),delta) * sp
        nabla_b[-l] = delta
        nabla_w[-l] = np.dot(delta,activations[-l-1].transpose())
    return (nabla_b,nabla_w)
    
    

def cost_derivative(output_activations,y):
    return (output_activations - y)

def sigmoid(z):
    return 1.0/(np.exp(-z)+1)

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
    

training_data, validation_data, test_data =mnist_loader.load_data_wrapper()

network([784,100,10])

SGD(training_data, 30 , 20, 3.0, test_data)
    
        


    












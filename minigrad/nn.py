from minigrad.engine import Value
from typing import List
import random

class Module: 
    def zero_grad(self): 
        for p in self.parameters(): 
            p.grad = 0.0

    def parameters(): 
        return []
    

class Neuron(Module): 
    def __init__(self, nin: int, activation: callable = None):

        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.activation = activation

    def __call__(self, x): 
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act if self.activation is None else self.activation(act)
    
    def parameters(self): 
        return self.w + [self.b]
    
    
class Layer(Module): 
    def __init__(self, nin: int, nout: int, activation: callable = None):
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]

    def __call__(self, x): 
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return[p for n in self.neurons for p in n.parameters()]
    
class MLP(Module): 
    def __init__(self, nin: int, nouts: List[int], activation: callable = None): 

        """
        nin: number of inputs(input features)

        nouts: list of numbers representing the size of each layer

        activation: activation function to be used in each neuron

        sz: creates a list of layer sizes including input size. For example: 
                if nin = 3 and nouts = [4, 2], then sz = [3, 4, 2]
                This means 3 inputs, 01 layer of 04 neurons, and 01 layer of 02 neurons
        """ 
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], activation) for i in range(len(nouts))] #Defining the input and output of each layer sequentially...

    def __call__(self, x): # forward pass...
        for layer in self.layers: 
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
        

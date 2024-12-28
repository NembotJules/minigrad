from minigrad.engine import Value
from typing import List
from graphviz import Digraph
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
    

class Optimizer: 
    def __init__(self, parameters):
        self.parameters = parameters

class SGD(Optimizer): 
    def __init__(self, parameters, lr = 0.01):
        super().__init__(parameters)
        self.lr = lr
    
    def step(self): 
        for p in self.parameters: 
            p.data -= self.lr * p.grad

# class Adam(Optimizer):
#     def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
#         super().__init__(parameters)
#         self.lr = lr
#         self.betas = betas
#         self.eps = eps
#         self.m = [0.0 for _ in parameters]  # First moment
#         self.v = [0.0 for _ in parameters]  # Second moment
#         self.t = 0  # Timestep
    
#     def step(self):
#         self.t += 1
#         b1, b2 = self.betas
        
#         for i, p in enumerate(self.parameters):
#             g = p.grad
#             self.m[i] = b1 * self.m[i] + (1 - b1) * g
#             self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            
#             # Bias correction
#             m_hat = self.m[i] / (1 - b1**self.t)
#             v_hat = self.v[i] / (1 - b2**self.t)
            
#             p.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
        

class NeuralNetwork: 
    def __init__(self, nin: int, architecture: List[int], activation: callable = None) -> None:
        self.model = MLP(nin, architecture, activation)
        self.optimizer = None
        self.last_output = None # tracking the last output

    def forward(self, x): 
        out = self.model(x)
        self.last_output = out  #storing the output
        return out
    
    def backward(self): 
        if self.last_output is None: 
            raise ValueError("No forward pass perform yet")
        
        self.last_output.backward()

    def zero_grad(self): 
        self.model.zero_grad()

    def set_optimizer(self, optimizer_name = 'sgd', **kwargs): 
        if optimizer_name.lower() == 'adam': 
            pass
           # self.optimizer = Adam(self.model.parameters(), **kwargs)
        elif optimizer_name.lower() =='sgd': 
            self.optimizer = SGD(self.model.parameters(), **kwargs)

    def step(self): 
        if self.optimizer is None: 
            raise ValueError("No Optimizer set. Call set_optimizer() first")
        self.optimizer.step()

    def trace(self, root):
        # builds a set of all nodes and edges in a graph
        nodes, edges = set(), set()
        
        def build(v):
            if v not in nodes:
                nodes.add(v)
                for child in v._prev:
                    edges.add((child, v))
                    build(child)
        build(root)
        return nodes, edges

    def draw_nn(self, root, format='svg', rankdir='LR'):
        """
        Draw the Neural Network with calculated gradients for each weights.
        You should pass the last output element, and it will draw the neural network, back from that point.
        format: png | svg | ...
        rankdir: TB (top to bottom graph) | LR (left to right)
        """
        dot = Digraph(format=format, 
                    graph_attr={
                        'rankdir': rankdir,
                        'bgcolor': '#ffffff',  # White background
                        'ratio': 'expand',
                        'width': '100',
                        'height': '50',
                        'margin': '0.1',
                        'nodesep': '0.5',     # Increased space between nodes
                        'ranksep': '0.5'      # Increased rank separation
                    })
        
        nodes, edges = self.trace(root)
        
        for n in nodes:
            uid = str(id(n))
            
            # Get the variable name if it exists in locals/globals
            var_name = None
            for name, value in globals().items():
                if value is n:
                    var_name = name
                    break
            if var_name is None:
                for name, value in locals().items():
                    if value is n:
                        var_name = name
                        break
            
            # Enhanced node formatting
            label = f"""{{
                {var_name if var_name else ''}
                |data: {n.data:.4f}
                |grad: {n.grad:.4f}
            }}"""
            
            dot.node(name=uid, 
                    label=label, 
                    shape='record',
                    style='filled',
                    fillcolor='#e8f3ff',  # Light blue background
                    color='#2878b5',      # Darker blue border
                    fontname='Arial',
                    fontsize='12')
            
            if n._op:
                # Operation node styling
                dot.node(name=uid + n._op, 
                        label=n._op,
                        shape='circle',
                        style='filled',
                        fillcolor='#ff9999',  # Light red for operations
                        color='#cc4444',      # Darker red border
                        fontname='Arial Bold',
                        fontsize='12',
                        width='0.5',
                        height='0.5')
                dot.edge(uid + n._op, uid, color='#666666')
        
        # Edge styling
        for n1, n2 in edges:
            dot.edge(str(id(n1)), 
                    str(id(n2)) + n2._op, 
                    color='#666666',
                    penwidth='1.5')
        
        return dot
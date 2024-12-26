import random
import math
from typing import List, Tuple, Callable
from graphviz import Digraph

class Value: 
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0  # Initialize gradient to zero
        self._backward = lambda: None  # Default backward function
        self._prev = set(_children)  # Track dependencies
        self._op = _op  # Operation that created this value
    
    def __repr__(self) -> str:
        return f"Value(data={self.data})"
    
    def __add__(self, other): 
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        
        return out
    
    def __mul__(self, other): 
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out
    
    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)  # Reuse add by negating other
    
    def __neg__(self):  # for negative numbers
        return self * -1
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')
        
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        
        return out
    

    
    def backward(self):
        # Topological sort all children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        # Go one variable at a time and apply the chain rule
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

def trace(root):
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

def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})
    
    nodes, edges = trace(root)
    
    for n in nodes:
        # unique node id (needed for graphviz)
        uid = str(id(n))
        
        # for any value in the graph, create a rectangular node for it
        dot.node(name=uid, 
                label=f"{{ data: {n.data:.4f} | grad: {n.grad:.4f} }}", 
                shape='record')
        
        if n._op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name=uid + n._op, label=n._op)
            # and connect this node to it
            dot.edge(uid + n._op, uid)
    
    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
    
    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, nin: int, activation: Callable = None):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.activation = activation
    
    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act if self.activation is None else self.activation(act)
    
    def parameters(self):
        return self.w + [self.b]

class Layer(Module):
    def __init__(self, nin: int, nout: int, activation: Callable = None):
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP(Module):
    def __init__(self, nin: int, nouts: List[int], activation: Callable = None):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], activation) for i in range(len(nouts))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    

class Optimizer:
    def __init__(self, parameters):
        self.parameters = parameters
        
class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01):
        super().__init__(parameters)
        self.lr = lr
    
    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad

class Adam(Optimizer):
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(parameters)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.m = [0.0 for _ in parameters]  # First moment
        self.v = [0.0 for _ in parameters]  # Second moment
        self.t = 0  # Timestep
    
    def step(self):
        self.t += 1
        b1, b2 = self.betas
        
        for i, p in enumerate(self.parameters):
            g = p.grad
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            
            # Bias correction
            m_hat = self.m[i] / (1 - b1**self.t)
            v_hat = self.v[i] / (1 - b2**self.t)
            
            p.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)


class NeuralNetwork:
    def __init__(self, nin: int, architecture: List[int], activation: Callable = None):
        self.model = MLP(nin, architecture, activation)
        self.optimizer = None
        self.last_output = None  # Add this to track the last output
    
    def forward(self, x):
        out = self.model(x)
        self.last_output = out  # Store the output
        return out
    
    def backward(self):
        if self.last_output is None:
            raise ValueError("No forward pass performed yet")
        self.last_output.backward()
    
    def zero_grad(self):
        self.model.zero_grad()
    
    def set_optimizer(self, optimizer_name='adam', **kwargs):
        if optimizer_name.lower() == 'adam':
            self.optimizer = Adam(self.model.parameters(), **kwargs)
        elif optimizer_name.lower() == 'sgd':
            self.optimizer = SGD(self.model.parameters(), **kwargs)
    
    def step(self):
        if self.optimizer is None:
            raise ValueError("No optimizer set. Call set_optimizer() first.")
        self.optimizer.step()
    
    def visualize(self):
        if self.last_output is None:
            raise ValueError("No forward pass performed yet")
        dot = draw_dot(self.last_output)
        return dot.render('neural_network', view=True)
    



# Example usage
if __name__ == "__main__":
    # Create a neural network with 2 inputs, [3, 1] means 3 neurons in hidden layer and 1 in output
    nn = NeuralNetwork(2, [3, 1], activation=lambda x: x.tanh())

    # Training data
    X = [[2.0, 3.0], [3.0, -1.0], [0.5, 1.0], [1.0, 1.0]]
    y = [1.0, -1.0, -1.0, 1.0]

    # Set optimizer
    nn.set_optimizer('adam', lr=0.01)

    # Training loop
   # Training loop
for epoch in range(100):
    total_loss = 0
    for x_i, y_i in zip(X, y):
        # Convert inputs to Value objects
        x = [Value(x) for x in x_i]
        
        # Forward pass
        pred = nn.forward(x)
        loss = (pred - Value(y_i))**2
        nn.last_output = loss  # Store the loss for visualization
        
        # Backward pass
        nn.zero_grad()
        loss.backward()
        
        # Optimize
        nn.step()
        
        total_loss += loss.data
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {total_loss/len(X)}')
        nn.visualize()  # Now this will visualize the loss computation graph
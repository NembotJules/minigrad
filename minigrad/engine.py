import math

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
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
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


    def __truediv__(self, other): 
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** -1
    
    def __rtruediv__(self, other): 
        other = other if isinstance(other, Value) else Value(other)
        return other * self ** -1

# Example usage
if __name__ == "__main__":

    a = Value(2.0)
    b = Value(3.0)

    c = b + a
    print(f"c.data: {c.data}")  # Should be 5.0

    d = c ** 2
    print(f"d.data: {d.data}")  # Should be 25.0

    e = d * 2
    print(f"e.data: {e.data}")  # Should be 50.0

    f = e**2 + 5 

    print(f"f.data: {f.data}")  # Should be 25005

    g = f * f 

    print(f"g.data: {g.data}")  # Should be 65025

    h = g * 2 - 3

    print(f"h.data: {h.data}")  # Should be 4228250622

    h.backward()

    print("\nGradients:")
    print(f"h.grad: {h.grad}")
    print(f"g.grad: {g.grad}")
    print(f"f.grad: {f.grad}")
    print(f"e.grad: {e.grad}")
    print(f"d.grad: {d.grad}")
    print(f"c.grad: {c.grad}")
    print(f"b.grad: {b.grad}")
    print(f"a.grad: {a.grad}")
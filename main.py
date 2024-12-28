from minigrad.engine import Value
from visualization import draw_nn



a = Value(2.0, name = 'a')
b = Value(3.0, name = 'b')

c = b + a
c.name = 'c'
print(f"c.data: {c.data}")  # Should be 5.0

d = c ** 2
d.name = 'd'
print(f"d.data: {d.data}")  # Should be 25.0

e = d * 2
e.name = 'e'
print(f"e.data: {e.data}")  # Should be 50.0

f = e**2 + 5 
f.name = 'f'

print(f"f.data: {f.data}")  # Should be 2505

g = f * f 
g.name = 'g'

print(f"g.data: {g.data}")  # Should be 6275025.0

h = g * 2 - 3
h.name = 'h'

print(f"h.data: {h.data}")  # Should be 12550047.0

h.backward()

print("\nGradients:")
print(f"h.grad: {h.grad}")
print(f"g.grad: {g.grad}")
print(f"f.grad: {f.grad}")
print(f"e.grad: {e.grad}")
print(f"d.grad: {d.grad}")
print(f"c.grad: {c.grad}")
print(f"b.grad: {b.grad}") #20040000.0
print(f"a.grad: {a.grad}") #20040000.0

draw_nn(h)
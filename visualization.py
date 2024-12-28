from graphviz import Digraph

def trace(root):
    """Build a set of all nodes and edges in the graph."""
    nodes, edges = set(), set()
    
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges


def draw_nn(root, filename='neural_network', format='svg', rankdir='LR'):
    """
    Draw and save the Neural Network computation graph.
    """
    dot = Digraph(format=format, 
                  graph_attr={
                      'rankdir': rankdir,
                      'bgcolor': '#ffffff',
                      'splines': 'ortho',        # Orthogonal lines for cleaner look
                      'nodesep': '0.8',          # Increased space between nodes
                      'ranksep': '1.0',          # Increased rank separation
                      'fontsize': '12',
                      'concentrate': 'true',      # Merge edges when possible
                  })
    
    nodes, edges = trace(root)
    
    for n in nodes:
        uid = str(id(n))
        
        # Get the variable name if it exists
        var_name = n.name if n.name else ''
        
        # Enhanced node formatting with HTML-like label
        label = f'''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
            <TR><TD>{var_name}</TD></TR>
            <TR><TD>data: {n.data:.4f}</TD></TR>
            <TR><TD>grad: {n.grad:.4f}</TD></TR>
        </TABLE>>'''
        
        dot.node(name=uid, 
                 label=label, 
                 shape='none',           # Using none shape for HTML-like labels
                 style='rounded',        # Rounded corners
                 fontname='Arial')
        
        if n._op:
            # Operation node styling
            dot.node(name=uid + n._op, 
                     label=n._op,
                     shape='circle',
                     style='filled',
                     fillcolor='#ff9999',
                     color='#cc4444',
                     fontname='Arial Bold',
                     fontsize='10',
                     width='0.5',
                     height='0.5')
            dot.edge(uid + n._op, uid, 
                     color='#666666',
                     penwidth='1.2',
                     arrowsize='0.8')
    
    # Edge styling
    for n1, n2 in edges:
        dot.edge(str(id(n1)), 
                 str(id(n2)) + n2._op, 
                 color='#666666',
                 penwidth='1.2',
                 arrowsize='0.8')
    
    # Save the graph
    dot.render(filename, view=True, cleanup=True)
    return dot
# utility functions for graph-vizualizations using graphviz
# used ChatGPT for this code...

from graphviz import Digraph

def _tensor_label(t):
    req = "req_grad" if getattr(t, "requires_grad", False) else "no_grad"
    shape = getattr(t, "shape", None)
    grad_shape = getattr(t, "grad", None)
    grad_shape = grad_shape.shape if hasattr(grad_shape, "shape") else None
    return f"Tensor\nshape={tuple(shape)}, {req}, {grad_shape}"

def _tensor_id(t):  # stable-ish node id
    return f"T{id(t)}"

def _fn_id(fn):
    return f"F{id(fn)}"

def make_dot(
    output_tensor,
    params: dict = None,
    filename: str = "autograd_graph",
    directory: str = None,
    format: str = "png",
    graph_attr=None,
    node_attr=None,
    edge_attr=None,
):
    """
    **Experimental**, this function is likely to change in later updates.
    
    Draws trace for output_tensor.

    Args:
        output_tensor: the final Tensor (e.g., `out`) you called backward on.
        params: optional dict {name: tensor} to highlight parameters.
    """
    params = params or {}
    name_for_tensor = {id(v): k for k, v in params.items()}

    g = Digraph(name="AutogradGraph", format=format, directory=directory)
    g.attr(rankdir="LR", **(graph_attr or {}))
    g.node_attr.update(dict(shape="record", fontsize="10"), **(node_attr or {}))
    g.edge_attr.update(dict(arrowsize="0.7"), **(edge_attr or {}))

    seen_tensors = set()
    seen_fns = set()

    def add_tensor_node(t):
        tid = _tensor_id(t)
        if tid in seen_tensors:
            return
        seen_tensors.add(tid)

        label_name = name_for_tensor.get(id(t))
        main = _tensor_label(t)
        if label_name:
            label = f"{label_name}|{main}"
        else:
            label = main

        style = "filled"
        color = "#b7e1cd" if getattr(t, "is_leaf", True) and getattr(t, "requires_grad", False) else "#dddddd"
        g.node(tid, label=label, style=style, fillcolor=color)

    def add_fn_node(fn):
        fid = _fn_id(fn)
        if fid in seen_fns:
            return
        seen_fns.add(fid)
        opname = fn.__class__.__name__
        g.node(fid, label=opname, shape="ellipse", style="filled", fillcolor="#cfe2f3")

    def build(t):
        # current tensor node
        add_tensor_node(t)
        fn = getattr(t, "grad_fn", None)
        if fn is None:
            return  # leaf
        # function node
        add_fn_node(fn)
        # edge: fn -> this tensor
        g.edge(_fn_id(fn), _tensor_id(t))
        # parents
        parents = getattr(fn, "_parents", []) or []
        for p in parents:
            add_tensor_node(p)
            # edge: parent tensor -> fn
            g.edge(_tensor_id(p), _fn_id(fn))
            build(p)

    build(output_tensor)

    g.render(filename=filename, cleanup=True)
    return g

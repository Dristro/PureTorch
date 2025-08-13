import weakref
import numpy as np
from typing import Union, Optional, Callable, Tuple

from .context import Context
from .function import Function
from .ops import (
    Add,
    Mul,
    Sub,
    Div,
    Neg,
    MatMul,
    Mean,
    Transpose,
    Reshape,
    ReLU,
    Pow,
    VariableSum,
    Exp,
    Log,
)

class Variable:
    def __init__(
            self,
            data: Union[int, float, list, tuple, np.ndarray, 'Variable'],
            requires_grad: bool = False,
            grad_fn: Optional[Function] = None,
            is_leaf: bool = True,
        ):
        self.data: np.ndarray = data if isinstance(data, np.ndarray) else np.array(data)
        self.shape = self.data.shape
        self.grad: Optional[np.ndarray] = None
        self.grad_fn = grad_fn  # function that produced this tensor (None if leaf tensor)
        self.requires_grad = requires_grad
        self.is_leaf = is_leaf
        self._backward_hooks = []
        self._self_weakref = weakref.ref(self)  # weakref to self for use in saved_tensors to avoid cycles

    def __repr__(self):
        return f"tensor({self.data}, requres_grad={self.requires_grad})"

    def zero_grad(self):
        self.grad = None
    
    def detach(self):
        return Variable(self.data.copy(), requires_grad=False, grad_fn=None, is_leaf=True)

    def requres_grad_(self, val=True):
        self.requires_grad = val
        return self

    def backward(self, gradient: Optional[np.ndarray] = None):
        r"""
        Compute gradients of the graph w.r.t current tensor.
        If `gradient = None`, then its assumed to be one.

        For custom gradients, ensure that: `gradient.shape == tensor.shape`

        Args:
            gradient (Optional[np.ndarray]): gradient of tensor
        """
        if not self.requires_grad:
            raise RuntimeError("Called .backward() on a tensor that doesn't require grad.")
        
        # assume grad is one if not provided
        if gradient is None:
            gradient = np.ones_like(self.data)
        
        # build topo sort
        topo = []
        visited = set()
        def build(_tensor: 'Variable'):
            if id(_tensor) in visited:
                return
            visited.add(id(_tensor))
            if _tensor.grad_fn is not None:
                for child in _tensor.grad_fn._parents:
                    build(child)
            topo.append(_tensor)
        build(self)

        # init the grads
        grads = {id(self): gradient.copy()}

        # use topo to propagate
        gen = (_t for _t in reversed(topo) if _t.requires_grad)  # reduce number of indent-blocks
        for _tensor in gen:
            grad = grads.get(id(_tensor))
            if grad is None:
                continue
            # processing a leaf
            if _tensor.is_leaf:
                if _tensor.grad is None:
                    _tensor.grad = grad   # init grad
                else:                     # (or)
                    _tensor.grad += grad  # accumilate grad
                for hook in _tensor._backward_hooks:
                    hook(_tensor)  # call all hooks (if given)
            # processing intermediate tensors
            else:
                grad_out = _tensor.grad_fn.backward(_tensor.grad_fn._ctx, grad)  # op-wise backward
                # logic for any function (even custom ones, defined by user)
                if not isinstance(grad_out, tuple):
                    grad_out = (grad_out,)
                for parent, g_out in zip(_tensor.grad_fn._parents, grad_out):  # number of params need-not be 2, thus loop
                    if g_out is None:
                        continue
                    # inverse-brodcasing, reduce gradient to parent shape (if needed)
                    g_out = _unbroadcast(g_out, parent.data.shape)
                    if id(parent) not in grads:
                        grads[id(parent)] = g_out  # put new tensor with grad in dict
                    else:
                        grads[id(parent)] += g_out # accumulate grad, if tensor alr in dict

    # hooks
    def register_hook(self, fn: Callable[['Variable'], None]):
        self._backward_hooks.append(fn)

    # general functions
    def reshape(self, shape: tuple):
        return reshape(self, shape=shape)

    # math operators
    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(other, self)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return sub(other, self)

    def __truediv__(self, other):
        return div(self, other)

    def __rtruediv__(self, other):
        return div(other, self)

    def __neg__(self):
        return neg(self)

    def __matmul__(self, other):
        return matmul(self, other)
    
    def __pow__(self, exp: Union[int, float]):
        return pow(self, exp)

    @property
    def T(self):
        return transpose(self)
    
    def sum(self, dim: Optional[Union[int, tuple]] = None, keepdims: bool = False):
        return tensor_sum(a=self, dim=dim, keepdims=keepdims)
    
    def mean(self, dim: Optional[Union[int, tuple]] = None, keepdims: bool = False):
        return mean(a=self, dim=dim, keepdims=keepdims)
    
    def exp(self):
        return exp(a=self)
    
    def log(self):
        return log(a=self)
    
# REMOVE: later
#def tensor(shape: Tuple, requires_grad=False):
#    """Generate random tensor instance of given shape."""
#    return Variable(np.random.randn(shape, dtype=float), requires_grad=requires_grad, is_leaf=True)


# Helpers for the tensor class (not accessed outside this file.)

def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Reduce grad to the target shape (inverse of broadcasting)."""
    if grad.shape == shape:
        return grad
    # sum over leading dims
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)
    # sum dims of size 1
    for i, (gdim, sdim) in enumerate(zip(grad.shape, shape)):
        if sdim == 1 and gdim != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)

def enforce_tensor(x) -> Variable:
    """
    Converts 'x' into a Tensor instance if not already.
    Used for tensor-ops.
    """
    if isinstance(x, Variable):
        return x
    return Variable(np.array(x, dtype=float), requires_grad=False, is_leaf=True)

def _wrap_forward(fn_cls: type, *parents: Variable, **kwargs) -> Variable:
    """
    Wraps the forward-operation of a function (like: Add, Sub, etc)
    Used to setup optimized .backward() call.
    Sets up ctx, parents, grad_fn, etc
    """
    ctx = Context()
    parents_data = [parent.data for parent in parents]
    # forward
    out_data = fn_cls.forward(ctx, *parents_data, **kwargs)
    out = Variable(out_data, requires_grad=any(parent.requires_grad for parent in parents), grad_fn=None, is_leaf=False)
    # attach grad_fn instance with context and parents for backward
    fn = fn_cls()
    fn._ctx = ctx
    fn._parents = parents
    out.grad_fn = fn
    return out

def add(a, b):
    a = enforce_tensor(a)
    b = enforce_tensor(b)
    return _wrap_forward(Add, a, b)

def sub(a,b):
    a = enforce_tensor(a)
    b = enforce_tensor(b)
    return _wrap_forward(Sub, a, b)

def mul(a,b):
    a = enforce_tensor(a)
    b = enforce_tensor(b)
    return _wrap_forward(Mul, a, b)

def div(a,b):
    a = enforce_tensor(a)
    b = enforce_tensor(b)
    return _wrap_forward(Div, a, b)

def neg(a):
    a = enforce_tensor(a)
    return _wrap_forward(Neg, a)

def matmul(a,b):
    a = enforce_tensor(a)
    b = enforce_tensor(b)
    return _wrap_forward(MatMul, a, b)

def tensor_sum(a, dim: Optional[Union[int, tuple]] = None, keepdims: bool = False):
    a = enforce_tensor(a)
    return _wrap_forward(VariableSum, a, dim=dim, keepdims=keepdims)

def mean(a, dim=None, keepdims=False):
    a = enforce_tensor(a)
    return _wrap_forward(Mean, a, dim=dim, keepdims=keepdims)

def transpose(a):
    a = enforce_tensor(a)
    return _wrap_forward(Transpose, a)

def reshape(a, shape: tuple):
    a = enforce_tensor(a)
    return _wrap_forward(Reshape, a, shape=shape)

def relu(a):
    a = enforce_tensor(a)
    return _wrap_forward(ReLU, a)

def pow(a, exponent):
    a = enforce_tensor(a)
    return _wrap_forward(Pow, a, exponent=exponent)

def exp(a):
    a = enforce_tensor(a)
    return _wrap_forward(Exp, a)

def log(a):
    a = enforce_tensor(a)
    return _wrap_forward(Log, a)

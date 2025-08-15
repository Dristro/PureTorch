import numpy as np
from typing import Union, Optional

from .context import Context
from .function import Function

class Add(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray):
        ctx.save_for_backward(None, None)  # we will not need the tensors
        return a + b

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        return grad_output, grad_output

class Sub(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray):
        ctx.save_for_backward(None, None)  # we will not need the tensors
        return a - b

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        return grad_output, -grad_output

class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray):
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        a, b = ctx.saved_tensors
        return grad_output * b, grad_output * a

class Div(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray):
        ctx.save_for_backward(a, b)
        return a / b

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        a, b = ctx.saved_tensors
        return grad_output / b, -grad_output * a / (b ** 2)

class Neg(Function):
    @staticmethod
    def forward(ctx: Context, a):
        return -a

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        return -grad_output

class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray):
        ctx.save_for_backward(a, b)
        return a @ b

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        a, b = ctx.saved_tensors
        da = grad_output @ b.T
        db = a.T @ grad_output
        return da, db

class Transpose(Function):
    @staticmethod
    def forward(ctx: Context, a):
        ctx.save_for_backward(a.shape)
        return a.T

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        #(orig_shape,) = ctx.saved_tensors  # DEBUG
        return grad_output.T

class Reshape(Function):
    @staticmethod
    def forward(ctx: Context, a, shape):
        ctx.save_for_backward(a.shape)
        return a.reshape(shape)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        (orig_shape,) = ctx.saved_tensors
        return grad_output.reshape(orig_shape)

class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, a):
        ctx.save_for_backward(a)
        return np.maximum(a, 0)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        (a,) = ctx.saved_tensors
        grad = grad_output * (a > 0).astype(float)
        return grad

class Pow(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, exponent: Union[int, float]):
        assert isinstance(exponent, (int, float)), f"Exponent must be int or float, got: {exponent}"
        ctx.save_for_backward(a, exponent)
        return a ** exponent

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        a, exponent = ctx.saved_tensors
        return grad_output * exponent * (a ** (exponent - 1))
    
class VariableSum(Function):
    ###
    # Used chatgpt to fix bugs in this code, leading to issues with ce-loss
    ###
    @staticmethod
    def forward(ctx, a, dim=None, keepdims=False):
        """
        Args:
            a: np.ndarray
            dim: int or tuple of ints or None
            keepdims: if True, result shape matches 'a' in rank (PyTorch-style)
        """
        # Normalize dim to a sorted tuple of positive axes or None
        if dim is None:
            norm_dim = None
        else:
            if isinstance(dim, int):
                dim = (dim,)
            nd = a.ndim
            norm_dim = tuple(sorted(d if d >= 0 else d + nd for d in dim))
        ctx.save_for_backward(a.shape, norm_dim, keepdims)
        return np.sum(a, axis=norm_dim, keepdims=keepdims)

    @staticmethod
    def backward(ctx, grad_output):
        shape, dim, keepdims = ctx.saved_tensors

        # Sum over all elements
        if dim is None:
            # grad_output shape is () if keepdims=False, or shape of ones if keepdims=True
            return np.ones(shape, dtype=grad_output.dtype) * grad_output

        # We reduced some axes. If keepdims=False, reinsert singleton dims at reduced positions
        if not keepdims:
            shp = list(shape)
            for d in dim:
                shp[d] = 1
            grad_output = grad_output.reshape(shp)  # now broadcastable to 'shape'

        return np.ones(shape, dtype=grad_output.dtype) * grad_output

class Mean(Function):
    ###
    # Used chatgpt to fix bugs in this code, leading to issues with ce-loss
    ###
    @staticmethod
    def forward(ctx, a, dim=None, keepdims=False):
        if dim is None:
            norm_dim = None
            denom = a.size
        else:
            if isinstance(dim, int):
                dim = (dim,)
            nd = a.ndim
            norm_dim = tuple(sorted(d if d >= 0 else d + nd for d in dim))
            denom = 1
            for d in norm_dim:
                denom *= a.shape[d]
        ctx.save_for_backward(a.shape, norm_dim, keepdims, denom)
        return np.mean(a, axis=norm_dim, keepdims=keepdims)

    @staticmethod
    def backward(ctx, grad_output):
        shape, dim, keepdims, denom = ctx.saved_tensors

        if dim is None:
            # grad_output is scalar if keepdims=False
            return (np.ones(shape, dtype=grad_output.dtype) * grad_output) / denom

        if not keepdims:
            shp = list(shape)
            for d in dim:
                shp[d] = 1
            grad_output = grad_output.reshape(shp)

        return (np.ones(shape, dtype=grad_output.dtype) * grad_output) / denom

class Exp(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray):
        out = np.exp(a)
        ctx.save_for_backward(out)
        return out
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        out = ctx.saved_tensors
        return grad_output * out

class Log(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray):
        ctx.save_for_backward(a)
        return np.log(a)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        a = ctx.saved_tensors
        return grad_output/a

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

class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a, axis=None, keepdims=False):
        ctx.save_for_backward(a.shape, axis, keepdims)
        return np.sum(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        shape, axis, keepdims = ctx.saved_tensors
        if not keepdims and axis is not None:
            # need to reshape grad_output to have singleton dims for broadcast
            if isinstance(axis, int):
                axis_t = (axis,)
            else:
                axis_t = tuple(axis)
            #new_shape = list(grad_output.shape)  # DEBUG
            # build target shape
            shape_with_keep = list(shape)
            for ax in sorted(axis_t):
                shape_with_keep[ax] = 1
            reshaped = grad_output.reshape(shape_with_keep)
            return np.ones(shape) * reshaped
        else:
            return np.ones(shape) * grad_output

class Mean(Function):
    @staticmethod
    def forward(ctx: Context, a, dim=None, keepdims=False):
        ctx.save_for_backward(a.shape, dim, keepdims)
        return np.mean(a, axis=dim, keepdims=keepdims)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        shape, dim, keepdims = ctx.saved_tensors
        denom = np.prod([shape[i] for i in range(len(shape))]) if dim is None else np.prod([shape[i] for i in (dim if isinstance(dim, tuple) else (dim,))])
        if not keepdims and dim is not None:
            if isinstance(dim, int):
                dim = (dim,)
            else:
                dim = tuple(dim)
            shape_with_keep = list(shape)
            for ax in sorted(dim):
                shape_with_keep[ax] = 1
            reshaped = grad_output.reshape(shape_with_keep)
            return np.ones(shape) * reshaped / denom
        else:
            return np.ones(shape) * grad_output / denom

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
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, dim: Optional[Union[int, tuple]] = None, keepdims: bool = False):
        """
        Args:
            a: np.ndarray
            dim: int or tuple of ints or None
            keepdims: if True, result shape matches 'a' in rank (PyTorch-style)
        """
        ctx.save_for_backward(a.shape, dim, keepdims)
        return np.sum(a, axis=dim, keepdims=keepdims)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        shape, dim, keepdims = ctx.saved_tensors

        # Grad of sum is just ones, broadcasted to input shape
        if not keepdims and dim is not None:
            # reshape grad_output so it can broadcast back to original shape
            if isinstance(dim, int):
                dim = (dim,)
            grad_output = np.reshape(grad_output, [
                grad_output.shape[i] if i in dim else 1
                for i in range(len(shape) - (0 if keepdims else len(dim)))
            ])

        return np.ones(shape, dtype=grad_output.dtype) * grad_output

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

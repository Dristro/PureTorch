import numpy as np

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
    def forward(ctx: Context, a, b):
        ctx.save_for_backward(None, None)  # we will not need the tensors
        return a - b

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        return grad_output, -grad_output

class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a, b):
        ctx.save_for_backward(a, b)
        print(f"[DEBUG @ ops/Mul.forward] ctx: {ctx} | ctx.saved_tensors: {ctx.saved_tensors}")
        return a * b

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        print(f"[DEBUG @ ops/Mul.backward] ctx: {ctx} | ctx.saved_tensors: {ctx.saved_tensors}")
        a, b = ctx.saved_tensors
        return grad_output * b, grad_output * a

class Div(Function):
    @staticmethod
    def forward(ctx: Context, a, b):
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
    def forward(ctx: Context, a, b):
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
    def forward(ctx: Context, a, axis=None, keepdims=False):
        ctx.save_for_backward(a.shape, axis, keepdims)
        return np.mean(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        shape, axis, keepdims = ctx.saved_tensors
        denom = np.prod([shape[i] for i in range(len(shape))]) if axis is None else np.prod([shape[i] for i in (axis if isinstance(axis, tuple) else (axis,))])
        if not keepdims and axis is not None:
            if isinstance(axis, int):
                axis_t = (axis,)
            else:
                axis_t = tuple(axis)
            shape_with_keep = list(shape)
            for ax in sorted(axis_t):
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
    def forward(ctx: Context, a, exponent):
        assert isinstance(exponent, (int, float)), f"Exponent must be int or float, got: {exponent}"
        ctx.save_for_backward(a, exponent)
        return a ** exponent

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        a, exponent = ctx.saved_tensors
        return grad_output * exponent * (a ** (exponent - 1))

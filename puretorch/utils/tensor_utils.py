import numpy as np
from typing import Tuple
from ..tensor import Tensor

def tensor(*shape, **kwargs) -> Tensor:
    """Creates Tensor of given shape."""
    return Tensor(np.random.randn(*shape), **kwargs)

def zeros_like(a: Tensor) -> Tensor:
    """
    Returns a Tensor containing zeros, shaped like input (a).
    """
    return Tensor(
        data=np.zeros(shape=a.shape),
        requires_grad=a.requires_grad,
        device=a.device
    )

def allclose(a: Tensor, b: Tensor, **kwargs) -> bool:
    """
    Return True if two tensors are elementwise equal within a tolerance.
    kwargs are passed to np.allclose (rtol, atol, equal_nan).
    """
    return np.allclose(a.numpy(), b.numpy(), **kwargs)

def all(x: Tensor) -> bool:
    """
    Return True if all elements of the tensor evaluate to True.
    Equivalent to np.all(x).
    """
    return np.all(x.numpy())

def equal(a: Tensor, b: Tensor) -> bool:
    """
    Return True if two tensors have the same shape and elements.
    Equivalent to np.array_equal(a, b).
    """
    return np.array_equal(a.numpy(), b.numpy())

def linspace(*args, **kwargs) -> Tensor:
    """
    Creates np.ndarray and initializes puretorch.Tensor instance.
    """
    return Tensor(
        data=np.linspace(*args, **kwargs),
    )

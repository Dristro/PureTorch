import numpy as np
from typing import Optional, Union

from puretorch import Tensor

def _as_cls_const(x: np.ndarray, like: Tensor) -> Tensor:
    """Create a constant (no-grad) tensor of the same class as `like`."""
    return type(like)(x, requires_grad=False, is_leaf=True)

def _broadcast_class_weight(weight: Optional[Tensor], logits: Tensor) -> Optional[Tensor]:
    """
    Take a class weight of shape [C] and reshape it to broadcast over logits [..., C].
    Returns a constant Tensor with no grad.
    """
    if weight is None:
        return None
    if isinstance(weight, Tensor):
        w_np = np.asarray(weight.data)
    elif isinstance(weight, (list, np.ndarray)):
        w_np = np.asarray(weight)
    else:
        raise TypeError(f"weight must be Tensor, list, or ndarray; got {type(weight)}")
    if w_np.ndim != 1 or w_np.shape[0] != logits.shape[-1]:
        raise ValueError(f"weight must have shape [C={logits.shape[-1]}], got {w_np.shape}")
    # reshape to [1, 1, ..., 1, C] to broadcast across batch/extra dims
    new_shape = (1,) * (len(logits.shape) - 1) + (logits.shape[-1],)
    return _as_cls_const(w_np.reshape(new_shape), logits)

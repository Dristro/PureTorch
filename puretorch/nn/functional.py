import numpy as np
from typing import (
    List,
    Union,
    Literal,
    Optional,
)

from puretorch import Tensor
from .utils import _as_cls_const, _broadcast_class_weight

# activations
def relu(x: Tensor) -> Tensor:
    return x.relu()

def tanh(x: Tensor) -> Tensor:
    pos = x.exp()
    neg = x.__neg__().exp()
    return (pos - neg) / (pos + neg)

def softmax(logits: Tensor, dim: int = -1) -> Tensor:
    # Numerically stable softmax: subtract per-row max
    shift_np = logits.data.max(axis=dim, keepdims=True)
    shift = _as_cls_const(shift_np, logits)
    exps = (logits - shift).exp()
    denom = exps.sum(dim=dim, keepdims=True)
    return exps / denom


def log_softmax(logits: Tensor, dim: int = -1) -> Tensor:
    return softmax(logits, dim=dim).log()


# losses
def cross_entropy(
    logits: Tensor,
    targets: Union[np.ndarray, Tensor],
    ignore_idx: int = -100,
    weight: Optional[Union[np.ndarray, List, Tensor]] = None,
    reduction: Optional[Literal["mean", "sum"]] = "mean",
) -> Tensor:
    """
    Cross-entropy over logits [..., C] and targets:
      - class indices [...] or
      - one-hot / soft probs [..., C]

    Args:
        logits (Tensor): tensor of size [B,C]
        targets (Tensor): tensor of size [B,]
        ignore_idx (int): target value that is ignored during loss calculation
        weight (Tensor, optional): a manual rescaling weight given to each class. If given, it has to be a Tensor of size C. Otherwise, it is treated as if having all ones.
        reduction (str, optional): reduction applied to output
    Returns:
        Tensor with loss as float
    """
    # Shapes
    C = logits.shape[-1]
    ls = log_softmax(logits, dim=-1)  # [..., C]
    w = _broadcast_class_weight(weight, logits)  # [..., C] or None

    # Convert targets to Tensor data for shape/branching (we'll build constants for masks/onehots)
    if isinstance(targets, Tensor):
        t_data = targets.data
    else:
        t_data = np.asarray(targets)

    # targets are distributions / one-hot [..., C]
    if t_data.ndim == ls.data.ndim and t_data.shape[-1] == C:
        t_const = _as_cls_const(t_data.astype(ls.data.dtype, copy=False), ls)  # no-grad constant
        if w is not None:
            per_class_term = t_const * ls * w
        else:
            per_class_term = t_const * ls
        loss_map = -per_class_term.sum(dim=-1)
        
        count_np = np.prod(loss_map.data.shape, dtype=np.int64)
        count = _as_cls_const(np.array(count_np, dtype=ls.data.dtype), ls)

    # targets are class indices [...]
    else:
        # build one-hot mask for selected classes, with ignore support
        if t_data.dtype.kind not in "iu":  # ints only
            raise ValueError(f"Index targets must be integer-like, got dtype {t_data.dtype} and shape {t_data.shape}")
        if t_data.shape != ls.data.shape[:-1]:
            raise ValueError(f"Index targets must have shape {ls.data.shape[:-1]}, got {t_data.shape}")

        # mask of valid (non-ignored) positions [...]
        valid_mask_np = (t_data != ignore_idx).astype(ls.data.dtype, copy=False)
        valid_mask = _as_cls_const(valid_mask_np, ls)  # no-grad

        # one-hot [..., C] only for valid indices
        # build flattened one-hot
        flat_targets = t_data.reshape(-1)
        flat_valid = (flat_targets != ignore_idx)
        onehot_np = np.zeros((flat_targets.size, C), dtype=ls.data.dtype)
        if flat_valid.any():
            onehot_np[np.arange(flat_targets.size)[flat_valid], flat_targets[flat_valid]] = 1.0
        onehot = _as_cls_const(onehot_np.reshape(*t_data.shape, C), ls)  # no-grad

        if w is not None:
            per_class = onehot * (ls * w)
        else:
            per_class = onehot * ls

        picked = per_class.sum(dim=-1)
        loss_map = -(picked * valid_mask)
        # Count valid positions
        count_np = valid_mask_np.sum(dtype=np.int64)
        count = _as_cls_const(np.array(count_np, dtype=ls.data.dtype), ls)

    # out-logic
    num = loss_map.sum()
    if reduction is None or reduction == "mean":
        if count.data == 0:
            return num * _as_cls_const(np.array(0.0, dtype=ls.data.dtype), ls)
        return num / count
    elif reduction == "sum":
        return num
    else:
        raise ValueError(f"Got unexpected reduction: {reduction}. Use 'mean', 'sum', or None.")  # better error msg...

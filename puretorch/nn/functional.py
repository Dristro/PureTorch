import numpy as np
from typing import Union, Literal

from puretorch import Tensor

def _as_cls_const(x: np.ndarray, like: Tensor) -> Tensor:
    """Create a constant (no-grad) tensor of the same class as `like`."""
    return type(like)(x, requires_grad=False, is_leaf=True)

def softmax(logits: Tensor, dim: int = -1) -> Tensor:
    # Numerically stable softmax: subtract per-row max
    shift_np = logits.data.max(axis=dim, keepdims=True)
    shift = _as_cls_const(shift_np, logits)
    exps = (logits - shift).exp()
    denom = exps.sum(dim=dim, keepdims=True)
    return exps / denom

def log_softmax(logits: Tensor, dim: int = -1) -> Tensor:
    # log(softmax(x))
    return softmax(logits, dim=dim).log()

def cross_entropy_loss(
    logits: Tensor,
    targets: Union[np.ndarray, Tensor],
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Tensor:
    """
    Cross-entropy for integer class targets.
    logits: [B, C] (unnormalized scores)
    targets: int indices [B] (np.ndarray or Tensor)
    """
    # Normalize targets to np.int64 1-D array
    if isinstance(targets, Tensor):
        tgt = np.array(targets.data, dtype=np.int64).reshape(-1)
    else:
        tgt = np.array(targets, dtype=np.int64).reshape(-1)

    B, C = logits.shape
    assert tgt.shape == (B,), f"targets shape must be (B,), got {tgt.shape}"

    # Log-probabilities
    log_probs = log_softmax(logits, dim=-1)  # [B, C]

    # One-hot to select the correct class; avoids needing a gather op
    one_hot = np.zeros((B, C), dtype=logits.dtype)
    one_hot[np.arange(B), tgt] = 1.0
    one_hot_t = _as_cls_const(one_hot, logits)  # same class as logits

    # per-sample -ve log-likelihood: -sum(one_hot * log_probs, dim=-1)
    nll_per_sample = -(log_probs * one_hot_t).sum(dim=-1)  # [B]

    if reduction == "mean":
        return nll_per_sample.mean()
    elif reduction == "sum":
        return nll_per_sample.sum()
    elif reduction == "none":
        return nll_per_sample
    else:
        raise ValueError("reduction must be 'mean' | 'sum' | 'none'")


def cross_entropy(
    logits: Tensor,
    targets: Union[np.ndarray, Tensor],
    ignore_idx: int = -100,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Tensor:
    """
    Cross entropy loss over logits and targets.

    Args:
        logits (Tensor): tensor of size [B,C]
        targets (Tensor): tensor of size [B,]
        ignore_idx (int): target value that is ignored during loss calculation
        reduction (str, optional): reduction applied to output
    Returns:
        Tensor with loss as float
    """
    assert logits.shape[0] == targets.shape[0], f"batch dim not matching bw logits ({logits.shape[0]}) and targets ({targets.shape[0]})"
    reduction = reduction.lower()
    if reduction == "mean" or reduction == "none":
        pass
    elif reduction == "sum":
        pass
    else:
        raise ValueError(f"Got unexpected reduction: {reduction}, please read docstring for proper value of reduction.")

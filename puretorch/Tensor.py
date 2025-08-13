import numpy as np
from typing import Union, Optional
from autograd import Variable, Function

class Tensor(Variable):
    def __init__(
        self,
        data: Union[int, float, list, tuple, np.ndarray, Variable],
        requires_grad: bool = False,
        grad_fn: Optional[Function] = None,
        is_leaf: bool = True,
        device: str = "cpu",
    ):
        """
        Creates a Tensor instance with data.

        Args:
            data: data in tensor
            requires_grad: tracks gradient if `True`
            grad_fn: function used to arrive to current tensor
            is_leaf: is tensor a leaf-tensor
            device: device the tensor lives on (cpu only for now)
        """
        super().__init__(
            data=data,
            requires_grad=requires_grad,
            grad_fn=grad_fn,
            is_leaf=is_leaf,
        )
        self._device = device

    @property
    def device(self):  # for future gpu support
        return self._device

    @property
    def item(self):
        return self.data
        
    def __repr__(self):
        return f"tensor({self.data}, requires_grad={self.requires_grad}, device={self.device}, dtype={self.dtype})"

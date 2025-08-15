from typing import List
import puretorch
from .optimizer import Optimizer


class SGD(Optimizer):
    def __init__(
            self,
            params: list[puretorch.Tensor],
            lr: float = 1e-3,
            momentum: float = 0.9
        ):
        """
        Initialize the SGD optimizer for given parameters.
        Args:
            params: parameters to be optimized (list only for now)
            lr: learning rate for the optimizer
            momentum: momentum for the optimizer

        Note:
            Momentum functionality is not implemented yet
        """
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum

    def step(self):
        """
        Performs a single optimization step using vanilla SGD (no momentum).
        Updates the parameters with new values using the learning rate and their gradients.
        Formula:
            param.data -= lr * param.grad
            Where, param.data is the parameter's value and param.grad is the parameters gradient w.r.t. the loss.
        """
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None: # ensure param has a grad
                    param._data -= self.lr * param.grad

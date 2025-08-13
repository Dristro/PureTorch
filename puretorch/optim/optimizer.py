import puretorch
from typing import DefaultDict, List, Dict, Any, cast
from collections import defaultdict
from abc import abstractmethod
# Base class for all optimizer classes, be it custom or defaults

class Optimizer:
    """
    Base class for all optimizers
    Args:
        params: iterable of PureTorch.Tensor's. Parameters to optimized
    """
    def __init__(self, params):
        if not isinstance(params, (tuple, list)) and not hasattr(params, "__iter__"):
            raise TypeError("Parameters must be an iterable with Tensors or dicts, got: "
                            + params.__class__.__name__)

        self.state: DefaultDict[puretorch.Tensor, Any] = defaultdict(dict)
        self.param_groups: List[Dict[str, Any]] = []

        param_groups = list(params)
        if not param_groups or len(param_groups) == 0:  # Check for no parameters
            raise ValueError("No parameters to optimize.")
        if not isinstance(param_groups[0], dict):
            param_groups = [{"params": param_groups}]
        self.param_groups = param_groups # List of Dicts, where key = "params"

    def zero_grad(self, set_to_none: bool = False):
        """
        Sets the gradients of all the parameters given to the optimizer
        Args:
            set_to_none: sets the grads to None if True, sets the grads to zero otherwise.
        """
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    if set_to_none:
                        param.grad = None
                    else:
                        try:
                            param.zero_grad()
                        except AttributeError:
                            raise TypeError("Parameter does not support 'zero_grad'; ensure all parameters are "
                                            "PureTorch.Tensors.")

    # The step method must be defined for each optimizer
    @abstractmethod
    def step(self):
        """
        Performs a single optimization step on all given params (updates the parameters)
        """
        raise NotImplementedError

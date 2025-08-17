from puretorch import Tensor

class Parameter(Tensor):
    """Simple alias to mark a Tensor as trainable parameter."""
    def __init__(self, data, requires_grad: bool = True):
        super().__init__(
            data,
            requires_grad=requires_grad,
            is_leaf=True
        )

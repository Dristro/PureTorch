class SGD:
    def __init__(self, params: list,
                 lr: float = 1e-3,
                 momentum: float = 0.9):
        """
        Initialize the SGD optimizer for given parameters.

        Args:
            params:
            lr:
            momentum:
        """
        self.params = params
        self.lr = lr
        self.momentum = momentum

    def zero_grad(self):
        """
        Zero gradient for all the parameters provided
        """
        for param in self.params:
            if param.requires_grad:
                param.zero_grad()

    def update(self):
        """
        Updates the parameters data using the lr provided.
        Updation done using gradient descent.
        formula:
            data -= learning_rate * gradient

        ** Not implemented yet, will be added soon
        """
        pass

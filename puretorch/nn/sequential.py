from puretorch import Tensor, nn

class Sequential(nn.Module):
    def __init__(self,
                 *modules: nn.Module,
                 name: str = "Sequential model"):
        """
        Creates a new Sequential instance with the given layers.

        Args:
            layers: List of layers to propagate during the forward pass
            name: Name of the model

        Returns:
            None, creates a model instance
        """
        super().__init__()
        for i,m in enumerate(modules):
            self.add_module(f"{str(m.__class__)}{str(i)}", m)
        #self.layers = layers
        #self.name = name

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs a forward pass on the input Tensor and returns a Tensor with the output of the model
        Args:
            x: Tensor to propagate on.

        Returns:
            A Tensor with the output of the model on the input Tensor (x).
        """
        for m in self.children():
            x = m(x)
        return x
        #for layer in self.layers:
        #    x = layer.forward(x)
        #return x

    #def parameters(self):
    #    """
    #    Runs through each layer in the Sequential instance and gets the parameters from each layer and returns a list
    #    with the parameters of the model.
    #
    #    NOTE: if you are using custom layers, you need to add your own `.parameters()` function to your layer. Make
    #    sure to return a list of parameters Returns: A list of the parameters of the model.
    #    """
    #    parameters = []
    #    for layer in self.layers:
    #        yield from layer.parameters()

    #def __call__(self, x):
    #    return self.forward(x)
    
    def __repr__(self):
        return f"[INFO] add this later, not that important"

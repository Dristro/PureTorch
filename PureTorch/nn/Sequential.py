from PureTorch import Tensor


class Sequential():
    def __init__(self,
                 layers: list,
                 name: str = "Sequential model"):
        """
        Creates a new Sequential instance with the given layers.

        Args:
            layers: List of layers to propagate during the forward pass
            name: Name of the model

        Returns:
            None, creates a model instance
        """
        self.layers = layers
        self.name = name

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs a forward pass on the input Tensor and returns a Tensor with the output of the model
        Args:
            x: Tensor to propagate on.

        Returns:
            A Tensor with the output of the model on the input Tensor (x).
        """
        for layer in self.layers:
            x = layer.forward()
        return x

    def parameters(self):
        """
        Runs through each layer in the Sequential instance and gets the parameters from each layer and returns a list
        with the parameters of the model.

        NOTE: if you are using custom layers, you need to add your own `.parameters()` function to your layer. Make
        sure to return a list of parameters Returns: A list of the parameters of the model.
        """
        parameters = []
        for layer in self.layers:
            param = layer.parameters() if hasattr(layer, "parameters") else []
            parameters.append(param)
        return parameters

    def summary(self):
        """
        Prints the model's layer name along with the total parameter count.
        """
        print(f"Model: {self.name}\n")
        print(f"Layer name\t| Num params")
        print(f"---------------\t| ---------")
        sum = 0
        for layer in self.layers:
            print(f"{layer.__class__.__name__}\t\t| {len(layer.parameters())}")
            sum += layer.parameters()
        print(f"Total params: {sum}")

Sequential([]).parameters()
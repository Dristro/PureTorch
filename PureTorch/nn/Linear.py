from PureTorch.nn import Perceptron

class Linear():
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True):
        """
        NOTE: This class is likely to not work the updated version of Tensor.
              This method works for the version of `PureTorch.Tensor` that doesn't use NumPy.

              

        Creates a layers of `Perceptrons` (found at PureTorch.nn.Perceptron)

        Args:
            in_features   - number of inputs the layer expects
            out_features  - number of perceptrons in layer (number of outputs of the layer)
            bias          - adds a bias factor per-perceptron in layer if `True`

        Returns:
            None
        """
        self.perceptrons = [Perceptron(num_inputs = in_features, bias = bias) for _ in range(out_features)]

    def forward(self, x):
        """
        Performs a forward pass using the layers weights and bias

        Args:
            x  - input to layer
        
        Returns:
            list of outputs of the layer
        """
        logits = [n(x) for n in self.perceptrons]
        return logits
        
    def parameters(self):
        return [p for perceptron in self.perceptrons for p in perceptron.parameters()]
    
    def __call__(self, x):
        return self.forward(x)
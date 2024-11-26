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
        self.in_features = in_features
        self.out_features = out_features

        # Weights (normalized, -1 to 1)
        weights_data = np.random.randn(out_features, in_features)
        weights_norm = np.linalg.norm(weights_data)
        epsilon = 1e-8
        if weights_norm != 0:
            weights_data /= (weights_norm + epsilon)
        self.weights = Tensor(weights_data, requires_grad=True)

        # Bias (normalized, -1 to 1)
        if bias:
            bias_data = np.random.randn(out_features)
            bias_norm = np.linalg.norm(bias_data)
            if bias_norm != 0:
                bias_data /= (bias_norm + epsilon)
            self.bias = Tensor(bias_data, requires_grad=True)
        else:
            self.bias = None

    def forward(self, x):
        """
        Performs a forward pass using the layers weights and bias

        Args:
            x  - input to layer

        Returns:
            list of outputs of the layer
        """
        # assert x.ndim >= 1, f"Add a batch-dim to the data, currently: {x.shape}"
        out = Tensor(
            np.dot(x.data, self.weights.data.T),
            _children=(x, self.weights),
            requires_grad=x.requires_grad or self.weights.requires_grad,
        )
        if self.bias is not None:
            out.data += self.bias.data
            out._prev.add(self.bias)
        return out

    def parameters(self):
        yield self.weights
        if self.bias is not None:
            yield self.bias

    def __call__(self, x):
        return self.forward(x)
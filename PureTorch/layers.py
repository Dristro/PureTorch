class Linear():
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 seed: int = None):
        import numpy as np
        ### Create the weights and biases
        np.random.seed(seed)
        self.weights = np.random.rand(out_features, in_features) * 0.01
        self.bias = np.zeros(out_features) if bias else None
        ### Track the grad's of the weights and the biases
        self.weights_grad = np.zeros_like(self.weights)
        self.bias_grad = np.zeros_like(self.bias) if self.bias is not None else None
        
    def forward(self, x):
        """
        Performs a forward pass on the linear layer,
        and returns the output of the layer.
        """
        #print(f"Linear layer input shape: {x.shape}")
        self.input = x
        x = x @ self.weights.T
        #print(f"Linear layer output shape: {x.shape}")
        if self.bias is not None:
            x = x + self.bias
        return x

    def backward(self,
                 grad_output):
        """
        Performs the backward pass for the linear layer

        Args:
            grad_output - grad of the loss with respect to this layer's output
        """
        import numpy as np
        ### Error handling
        if grad_output is None:
            raise ValueError("grad_output cannot be None")
        ### dL/dW = (dL/dY).T @ X
        #self.weights_grad = grad_output[:, None] @ self.input[None, :]
        self.weights_grad = grad_output.T @ self.input
        ### dL/db = (dL/dY).sum
        if self.bias is not None:
            self.bias_grad = np.sum(grad_output, axis = 0)
        ### dL/dX = dL/dY @ W
        grad_input = grad_output @ self.weights
        return grad_input
        
    def parameters(self):
        num_weights = self.weights.size
        num_bias = self.bias.size
        return num_weights + num_bias
    
class Sequential():
    def __init__(self,
                 layers: list,
                 lr: float = 1e-3,
                 name: str = "Sequential model"):
        """
        Initializes the layers of the model
        """
        self.layers = layers
        self.lr = lr
        self.name = name
        
    def forward(self, x):
        """
        Performs forward propagation and returns the output of the model
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self,
                 loss_grad):
        """
        Performs the backward pass on the model's parameters,
        and updates the weights of the model
        """
        ### Perform the backward pass
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update_params(self):
        ### Update weights and biases using gradient descent
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.weights = layer.weights - (self.lr * layer.weights_grad)
                if layer.bias is not None:
                    layer.bias = layer.bias - (self.lr * layer.bias_grad)
            
        
    def summary(self):
        """
        Prints the model's layers and their parameters
        """
        print(f"Model: {self.name}\n")
        print(f"Layer name\t| Num params")
        print(f"---------------\t| ---------")
        sum = 0
        for layer in self.layers:
            print(f"{layer.__class__.__name__}\t\t| {layer.parameters()}")
            sum += layer.parameters() if layer.parameters() is not None else 0
        print(f"Total params: {sum}")
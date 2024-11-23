from PureTorch import Tensor
import numpy as np

class Perceptron:
    def __init__(self,
                 num_inputs: int,
                 bias: bool = True,
                 requires_grad: bool = True):
        """
        Creates a single Perceptron (neuron/unit) with weights and bias.
        Note, each weight and bias is creating using the standard normal distribution.
        - random.gauss(0, 1)

        Args:
            num_inputs    - number of inputs the neuron expects
            bias          - adds a bias if `True`
            requires_grad - gradient tracking for weights and bias
        
        Returns:
            None
        """
        self.weights = Tensor(np.random.randn(num_inputs), requires_grad=requires_grad)
        self.bias = Tensor(np.random.randn(1), requires_grad=requires_grad) if bias else Tensor(np.zeros(1), requires_grad=requires_grad)

    def forward(self, x):
        """
        Performs the forward pass on the input 'x'

        Args:
            x - data input (type: Tensor or np.array)
        
        Returns:
            Tensor after forward pass
        """
        x = x if isinstance(x, Tensor) else Tensor(data=x, requires_grad=True)
        x = np.array(x.data) # convert to np array
        logits = np.dot(self.weights.data, x) + self.bias.data # perform forward pass
        return Tensor(logits, _children=(self.weights, self.bias), requires_grad=True) # return Tensor(logits)
    
    def parameters(self):
        params = [self.weights] + [self.bias]
        return params[0]

    def __call__(self, x):
        return self.forward(x)
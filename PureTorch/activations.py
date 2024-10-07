### Creating a few activation functions
class ReLU():
    def forward(self, x):
        import numpy as np
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        grad_input = grad_output * (self.input > 0)
        return grad_input
    
    def parameters(self):
        return None

class Softmax():
    def forward(self, x):
        import numpy as np
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))  # Numerical stability trick
        self.output = exps / np.sum(exps, axis=1, keepdims=True)  # Ensure sum is over correct axis
        return self.output

    def backward(self, grad_output):
        return grad_output  # Softmax doesn't change gradient during back-propagation

    def parameters(self):
        return None
    
### Testing the model using the TanH activation
class Tanh():
    def forward(self, x):
        import numpy as np
        ex = np.exp(x)
        e_x = 1/np.exp(x)
        self.output = (ex - e_x)/(ex + e_x)
        return self.output

    def backward(self, grad_output):
        grad_input = grad_output * (1 - self.output ** 2)
        return grad_input
        
    def parameters(self):
        return None
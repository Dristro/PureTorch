### Linear - feed forward layer (aka dense layer)
class Linear():
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 seed: int = None):
        """
        Note: this layer expects to get a batch-dim first for any operation
        """
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

        Expected to have batch-dim first.
        """
        #print(f"Linear layer input shape: {x.shape}")
        self.input = x
        self.batch_size = x.shape[0]
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
    
    def update_params(self, lr: float):
        """
        Updates the parameters of the Linear layer
        """
        self.weights -= lr*self.weights_grad
        if self.bias is not None:
            self.bias -= lr*self.bias_grad
        
    def parameters(self):
        num_weights = self.weights.size
        num_bias = self.bias.size if self.bias is not None else 0
        return num_weights + num_bias

class Flatten():
    def __init__(self,
                 input_shape):
        """
        Flattens the input from input_shape into a 'flat' single dimension output_shape.
        How it works:
            input_shape = (x, y, z)
            output_shape = x * y * z
        """
        from functools import reduce
        import operator
        self.input_shape = input_shape
        self.output_shape = reduce(operator.mul, self.input_shape)

    def forward(self, x):
        """
        Takes in a batch of inputs, and returns it with output_shape
        """
        import numpy as np
        return x.reshape(-1, self.output_shape)
    
    def backward(self, grad_output):
        """
        No gradients to track/change here, only converts the shape of the inputs
        """
        return grad_output
    
    def update_params(self, lr: float):
        """
        No updates are done here
        """
        pass

    def parameters(self):
        """
        Returns a "f" string with the expected input and output shapes
        """
        return f"Input: {self.input_shape}, Output: {self.output_shape}"

class Conv2D():
    def __init__(self,
                 kernels: int,
                 kernel_size: int,
                 padding: int = 0,
                 stride: int = 1,
                 bias: bool = True):
        """
        Creates a Conv2D layer instance with 'kernels' number of filters of size,
        (kernel_size, kernel_size).

        Args:
            kernels - number of filters (output channels)
            kernel_size - dimension of each kernel (e.g., kernel_size = 3  -> (3, 3))
            padding - padding added to the border of the input
            stride - the 'jump' after each convolution
            bias - adds a learnable bias to the output if `True`
        """
        import numpy as np
        self.kernels = kernels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        # Weight initialization (shape will be initialized during forward pass)
        self.weights = None  # Initialized during forward pass based on input channels
        self.bias = np.zeros(kernels) if bias else None

    def forward(self, x):
        """
        Performs the forward pass in the Conv2D layer

        Args:
            x: numpy array of shape (batch_size, channels, height, width)
        
        Returns:
            The output of the Conv2D layer
        """
        import numpy as np
        self.input = x  # Store the input for backward pass
        b, c, h, w = x.shape  # Get (batch_size, input_channels, height, width)

        if self.weights is None:
            # Initialize weights once we know the number of input channels (shape: kernels, input_channels, height, width)
            self.weights = np.random.randn(self.kernels, c, self.kernel_size, self.kernel_size) * 1e-2

        # Unpack weight shape
        k, in_channels, k_size, _ = self.weights.shape

        # Calculate output dimensions
        h_out = (h - k_size + 2 * self.padding) // self.stride + 1
        w_out = (w - k_size + 2 * self.padding) // self.stride + 1

        # Initialize the output tensor (batch_size, kernels, output_height, output_width)
        output = np.zeros((b, self.kernels, h_out, w_out))

        # Apply padding to the input (if required)
        if self.padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode="constant")

        # Perform the convolution
        for i in range(h_out):
            for j in range(w_out):
                for l in range(self.kernels):
                    # Extract region from input (taking a patch the size of the filter)
                    region = x[:, :, i*self.stride:i*self.stride+k_size, j*self.stride:j*self.stride+k_size]
                    # Convolve with the l-th filter (element-wise multiplication and sum across all input channels)
                    output[:, l, i, j] = np.sum(region * self.weights[l, :, :, :], axis=(1, 2, 3))
                    # Add bias (if any)
                    if self.bias is not None:
                        output[:, l, i, j] += self.bias[l]

        return output

    def backward(self, grad_out):
        """
        Performs the backward pass on the Conv2D layer

        Args:
            grad_out: Gradient of the loss w.r.t the output of the layer (batch_size, kernels, height_out, width_out)

        Returns:
            Gradient of the loss w.r.t the input of the layer
        """
        import numpy as np
        ### Getting things ready
        b, c, h, w = self.input.shape  # (batch_size, input_channels, height, width)
        _, f, h_out, w_out = grad_out.shape  # (batch_size, filters, height_out, width_out)
        k_size = self.kernel_size

        ### Init the gradients
        self.d_weights = np.zeros_like(self.weights)
        self.d_bias = np.zeros_like(self.bias) if self.bias is not None else None
        d_input = np.zeros_like(self.input)

        ### Apply padding to input and gradient input
        if self.padding > 0:
            padded_input = np.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
            d_input_padded = np.pad(d_input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        else:
            padded_input = self.input
            d_input_padded = d_input

        ### Perform the backward pass
        for i in range(h_out):
            for j in range(w_out):
                for l in range(self.kernels):
                    # Extract region from input (similar to forward pass)
                    region = padded_input[:, :, i*self.stride:i*self.stride+k_size, j*self.stride:j*self.stride+k_size]
                    # Gradient with respect to weights
                    self.d_weights[l] += np.sum(region * grad_out[:, l, i, j][:, np.newaxis, np.newaxis, np.newaxis], axis=0)
                    # Gradient with respect to bias
                    if self.d_bias is not None:
                        self.d_bias[l] += np.sum(grad_out[:, l, i, j], axis=0)
                    # Gradient with respect to input
                    d_input_padded[:, :, i*self.stride:i*self.stride+k_size, j*self.stride:j*self.stride+k_size] += self.weights[l] * grad_out[:, l, i, j][:, np.newaxis, np.newaxis, np.newaxis]

        ### Remove padding from d_input if padding was applied
        if self.padding > 0:
            d_input = d_input_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return d_input

    def update_params(self, lr: float):
        """
        Updates the parameters (weights and bias) of the Conv2D layer using SGD
        """
        self.weights -= lr * self.d_weights
        if self.bias is not None:
            self.bias -= lr * self.d_bias

    def parameters(self):
        """
        Returns the total number of parameters (weights and biases) in the Conv2D layer
        """
        num_weights = self.weights.size
        num_bias = self.bias.size if self.bias is not None else 0
        return num_weights + num_bias

class Sequential():
    def __init__(self,
                 layers: list,
                 name: str = "Sequential model"):
        """
        Initializes the layers of the model
        """
        self.layers = layers
        self.name = name
        self.compiled = False
        
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

    def update_params(self, lr):
        ### Update weights and biases using gradient descent
        self.lr = lr
        for layer in self.layers:
            ### New update method, all layers should have them from v0.1.3
            layer.update_params(lr = self.lr)
            ### REMOVE THE BELOW CODE IF EVERYTHING WORKS WELL
            #if isinstance(layer, Linear):
            #    layer.weights = layer.weights - (self.lr * layer.weights_grad)
            #    if layer.bias is not None:
            #        layer.bias = layer.bias - (self.lr * layer.bias_grad)

    def __accuracy_fn(self, y_pred, y_true):
        import numpy as np
        return np.sum(y_pred == y_true) / len(y_pred)
    
    def compile(self, input_shape: tuple):
        """
        Sets up the model, ensures that hte input/output shapes are correct for every layer.

        Args:
            input_shape - expected input shape for the first layer of the model
        """
        import numpy as np
        current_shape = input_shape
        for layer in self.layers:
            if isinstance(layer, Conv2D):
                layer.input_shape = current_shape
                output_height = (current_shape[1] - layer.kernel_size + 2 * layer.padding) // layer.stride + 1
                output_width = (current_shape[2] - layer.kernel_size + 2 * layer.padding) // layer.stride + 1
                current_shape = (layer.kernels, output_height, output_width)  # Update shape for next layer
            
            elif isinstance(layer, Flatten):
                layer.input_shape = current_shape
                layer.output_shape = np.prod(current_shape)
                current_shape = (layer.output_shape,)
            
            elif isinstance(layer, Linear):
                layer.input_shape = current_shape[0]
                layer.output_shape = (layer.weights.shape[0],)
        self.compiled = True
        print(f"Model: {self.name} | Compiled: {self.compiled} | Input shape: {input_shape}")

    def train(self,
              X_train,
              y_train,
              X_test,
              y_test,
              loss_fn,
              epochs: int,
              lr: float = 1e-3,
              batch_size: int = 32,
              print_freq: int = 1,
              track_acc: bool = False):
        """
        Trains the model for "epochs" and returns the losses of the model

        Args:
            X_train - training feature matrix
            y_train - training label vector
            X_test - testing feature matrix
            y_test - testing label vector
            loss_fn - loss function (from PureTorch.loss.X)
            epochs - number of epochs to train
            lr - learning rate (aka step_size)
            print_freq - how frequently the train/test losses and/or accuracy is printed
            track_acc - track the accuracy of the model's predictions (for classification)
        
        Returns:
            A Python dict with the training and testing losses and accuracies (if train_acc = True) of the model per epoch.
        
        Example usage:
            model = Sequential([
                Linear(10, 5),
                ReLU(),
                Linear(5, 2),
                Softmax()
            ], name = "example model")

            model_train_results = model.train(X_train = X_train, y_train = y_train,
                                              X_test = X_test, y_test = y_test,
                                              epochs = 5, lr = 1e-3, print_freq = 1, track_acc = True)
            
            # Here the model_train_results will contain the model's train and testing metrics as a Python dict.
        """
        import numpy as np
        ### Check for compilation status
        assert self.compiled, f"Compile the model before training it, this is will ensure that the correct input/output shapes are initialized for each layer"
        self.batch_size = batch_size

        results = {"train_loss": [],
                   "train_acc": [],
                   "test_acc": []} if track_acc else {"train_loss": []}
        
        ### Getting things ready
        

        ### Train for "epochs" number of epochs
        for epoch in range(epochs):
            ### Permute the train dataset
            permutation = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]

            for i in range(0, len(X_train), self.batch_size):
                ### Get batches from the permuted datasets
                X_batch = X_train_shuffled[i:i+self.batch_size]
                y_batch = y_train_shuffled[i:i+self.batch_size].flatten()
                
                ### Model steps
                y_pred = self.forward(X_batch) # Forward pass
                loss = loss_fn(y_pred=y_pred, y_true=y_batch) # Get loss
                loss_grad = loss_fn.backward() # Get the loss grad for backward pass
                self.backward(loss_grad) # Perform backward pass
                self.update_params(lr = lr) # Update the weights and biases where applicable
            
            if track_acc: # Track the accuracy scores in the 'results' Dict if user specifies
                y_preds = []
                # Calculate the accuracies
                for img in X_test:
                    img = img.reshape(1, -1)
                    pred = self.forward(img)
                    pred = pred.argmax(axis = 1)
                    y_preds.append(pred[0])
                test_acc = self.__accuracy_fn(y_pred = y_preds, y_true = y_test)
                train_acc = self.__accuracy_fn(np.argmax(y_pred, axis = 1), y_batch)
                # Add the accuracies to the 'results' Dict
                results["train_acc"].append(train_acc)
                results["test_acc"].append(test_acc)
                # Print the scores with the accuracies
                if (epoch+1) % print_freq == 0:
                    print(f"Epoch: {epoch + 1} | Train loss: {loss:.4f} | Train acc: {train_acc*100:.2f} % | Test acc: {test_acc*100:.2f} %")
            # Print the scores with the loss ONLY
            else:
                if (epoch+1) % print_freq == 0:
                    print(f"Epoch: {epoch + 1} | Train loss: {loss:.4f}")
            # Add the training loss to the 'results' Dict
            results["train_loss"].append(loss)
        
        return results
            
        
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
            sum += layer.parameters() if isinstance(layer.parameters(), (int)) else 0
        print(f"Total params: {sum}")
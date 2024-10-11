### Linear - feed forward layer (aka dense layer)
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
        
    def parameters(self):
        num_weights = self.weights.size
        num_bias = self.bias.size
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

    def parameters(self):
        """
        Returns a "f" string with the expected input and output shapes
        """
        return f"Input: {self.input_shape}, Output: {self.output_shape}"

class Sequential():
    def __init__(self,
                 layers: list,
                 name: str = "Sequential model"):
        """
        Initializes the layers of the model
        """
        self.layers = layers
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

    def update_params(self, lr):
        ### Update weights and biases using gradient descent
        self.lr = lr
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.weights = layer.weights - (self.lr * layer.weights_grad)
                if layer.bias is not None:
                    layer.bias = layer.bias - (self.lr * layer.bias_grad)

    def __accuracy_fn(self, y_pred, y_true):
        import numpy as np
        return np.sum(y_pred == y_true) / len(y_pred)

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
        self.batch_size = batch_size

        results = {"train_loss": [],
                   "train_acc": [],
                   "test_acc": []} if track_acc else {"train_loss": []}
        
        ### Train for "epochs" number of epochs
        for epoch in range(epochs):
            ### Permute the datasets
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
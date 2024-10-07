### Loss functions
class CrossEntropyLoss():
    def __init__(self,
                 batch_size: int = None):
        """
        Make sure to use one-hot encoded outputs (y_true)
        """
        self.y_pred = None
        self.y_true = None
        self.batch_size = batch_size
        
    def __call__(self, y_pred, y_true):
        """
        Calculates the loss given the True and Predicted values,
        and returns it.
        """
        import numpy as np
        self.y_pred = y_pred
        self.y_true = y_true
        ### Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, 1e-10, 1.0-1e-10)
        ### Calculate cross-entropy loss
        loss = -np.sum(y_true * np.log(y_pred))
        if self.batch_size is not None:
             loss /= self.batch_size
        return loss

    def backward(self):
        """
        Calculates the loss with respect to the True and Predicted values,
        and returns it.
        """
        grad = (self.y_pred - self.y_true)
        if self.batch_size is not None:
            grad /= self.batch_size
        return grad
    
    def parameters(self):
        return None

class SparseCrossEntropyLoss():
    def __init__(self, batch_size: int = None):
        """
        Use for continuous outputs. eg - 1, 2, ...
        """
        self.y_pred = None
        self.y_true = None
        self.batch_size = batch_size
        
    def __call__(self, y_pred, y_true):
        import numpy as np
        self.y_pred = y_pred
        self.y_true = y_true
        y_pred = np.clip(y_pred, 1e-10, 1.0 - 1e-10)
        
        # Calculate sparse cross-entropy loss
        loss = -np.log(y_pred[np.arange(len(y_true)), y_true])
        
        if self.batch_size is not None:
            loss = np.sum(loss) / self.batch_size
        else:
            loss = np.sum(loss) / len(y_true)
        
        return loss

    def backward(self):
        import numpy as np
        grad = self.y_pred.copy()
        grad[np.arange(len(self.y_true)), self.y_true] -= 1
        
        if self.batch_size is not None:
            grad /= self.batch_size
        
        return grad
    
    def parameters(self):
        return None
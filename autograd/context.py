class Context:
    """Context passed to forward to save variables for backward."""
    def __init__(self):
        self.saved_tensors = ()
        self.need_grad = True

    def save_for_backward(self, *tensors):
        self.save_tensors = tensors

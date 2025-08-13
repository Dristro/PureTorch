class Context:
    """Context passed to forward to save variables for backward."""
    def __init__(self):
        self._saved_tensors = ()

    def save_for_backward(self, *tensors):
        self._saved_tensors = tensors

    @property
    def saved_tensors(self):
        return self._saved_tensors

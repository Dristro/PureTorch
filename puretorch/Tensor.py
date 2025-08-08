import numpy as np

class Tensor():
    def __init__(self,
                 data,
                 _children: tuple = (),
                 _operand: str = "",
                 label: str = "",
                 requires_grad: bool = False):
        """
        Creates a Tensor instance with data.

        Args:
            data          - data in tensor (accessible with Tensor.data)
            _children     - instance's children (internal purpose)
            _operand      - operation on tensor (internal purpose)
            label         - name of tensor (used for drawing compute graph)
            requires_grad - tracks gradient if `True`

        Returns:
            None
        """
        data = data.data if isinstance(data, Tensor) else data
        data = data if isinstance(data, (np.ndarray, np.generic)) else np.array(data)
        # self.data = np.array(data.data, dtype=float).squeeze() if isinstance(data, Tensor) else np.array(data, dtype=float).squeeze()
        self.data = data
        self._prev = set(_children)
        self._operand = _operand
        self.label = label
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data, dtype=float)
        self._backward = lambda: None

        # Numpy attributes
        self.ndim = self.data.ndim
        self.shape = self.data.shape

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            data=self.data + other.data,
            _children=(self, other),
            _operand="+",
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            data=self.data * other.data,
            _children=(self, other),
            _operand="*",
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(
            data=np.exp(self.data),
            _children=(self,),
            _operand="exp",
            requires_grad=self.requires_grad,
        )

        def _backward():
            if self.requires_grad:
                self.grad += np.exp(self.data) * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only supporting int/float powers for now"
        base = self.data.astype(float) if np.issubdtype(self.data.dtype, np.integer) else self.data
        out = Tensor(
            data=base ** (other),
            _children=(self,),
            _operand=f"**{other}",
            requires_grad=self.requires_grad,
        )

        def _backward():
            if self.requires_grad:
                self.grad += (other * base ** (other - 1)) * out.grad
            else:
                self.grad += out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        out = self * (other ** -1)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad / other.data
            if other.requires_grad:
                other.grad += -self.data / (other.data ** 2) * out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        assert isinstance(other, Tensor), "Matrix multiplication requires 2 Tensor arguments"
        # assert self.data.ndim == 2 and other.data.ndim == 2, f"Both tensors need to be 2D, got: {self.data.ndim} and {other.data.ndim}"
        out = Tensor(
            data=self.data @ other.data,
            _children=(self, other),
            _operand="matmul",
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward():
            ### DEBUGGING
            # print("")
            # print(f"SELF.data: {self.data.shape}")# | {self.data.shape}")
            # print(f"OTHER.data: {other.data.shape}")# | {other.data.shape}")
            # print(f"OUT.grad: {out.grad.shape}")
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def sum(self):
        """
        Sums all the data in the Tensor instance

        Args:
            None (applies summation on the tensor)
        Returns:
            Tensor(tenor.data.sum)

        Example useage:
        ```Python
        # example-1
        t1 = Tensor([1, 2, 3, 4, 5])
        t2 = t1.sum()
        print(t2)

        # example-2
        t1 = Tensor(np.random.randn(5), requires_grad=True)
        t2 = t1.sum()
        print(t2)
            ```
        """
        return tensor_sum(self)

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        if np.all(self.grad == 0):
            self.grad = np.ones_like(self.data) if self.data.ndim > 0 else np.array(1.0)
        for node in reversed(topo):
            if node.requires_grad:
                node._backward()

    def zero_grad(self):
        """
        Sets the gradient of the Tensor object to zero
        """
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    @property
    def T(self):
        """
        Transpose of the Tensor if its 2D.
        """
        # assert self.ndim == 2, "Transpose only supported for 2D matrix"
        out = Tensor(
            data=self.data.T,
            _children=(self,),
            _operand="T",
            requires_grad=self.requires_grad
        )

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.T

        out._backward = _backward
        return out

    def __radd__(self, other):  # Reverse add
        return self + other

    def __sub__(self, other):  # Normal sub
        return self + (-other)

    def __rsub__(self, other):  # Reverse substitution
        return -(self - other)

    def __neg__(self):  # Negation
        return self * -1

    def __rmul__(self, other):  # Reverse multiplication
        return self * other

    def __rtruediv__(self, other):  # Reverse division
        return (self ** -1) * other

    def __repr__(self):
        formatted_data = np.array2string(
            self.data,
            precision=4,
            suppress_small=True,
            separator=', ',
            prefix=' ' * 7
        )
        statement = f"tensor({formatted_data}, requires_grad={self.requires_grad})" if self.requires_grad else f"tensor({formatted_data})"
        return statement


def tensor_sum(tensor: Tensor) -> Tensor:
    """
    Sums all the data in the input_tensor

    Args:
        tensor - a tensor with data

    Returns:
        Tensor(tenor.data.sum)
    """
    # Unpack the tensor
    data = tensor.data
    requires_grad = tensor.requires_grad
    out = Tensor(
        data=data.sum(),
        requires_grad=requires_grad,
    )

    def _backward():
        tensor.grad += out.grad

    out._backward = _backward
    return out
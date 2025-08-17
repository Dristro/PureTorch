from typing import List, Iterator, Any

from puretorch import nn

class Module:
    """
    Base module class.
    Handles module's train/eval modes only (as of now).
    """
    def __init__(self) -> None:
        self._modules = {}
        self._parameters = {}
        self._training = True

    def add_module(self, name: str, module: nn.Parameter):
        self._modules[name] = module

    def register_param(self, name: str, param: nn.Parameter):
        self._parameters[name] = param
    
    def children(self) -> Iterator["Module"]:
        return self._modules.values()
    
    def parameters(self) -> Iterator[nn.Parameter]:
        # module's params
        for p in self._parameters.values():
            yield p
        
        # child module(s) params
        # doing that recursively
        for m in self.children():
            yield from m.parameters()

    # forward/call
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # modes
    def train(self, mode: bool = True):
        """
        Current behavior: toggle requires_grad for all parameters (recursively)
        and set ._training flag. (You can later specialize for dropout/BN.)
        """
        self._training = mode
        for p in self.parameters():
            p.requires_grad = self._training

    def eval(self, mode: bool = True):
        self.train(not mode)

    def __repr__(self) -> str:
        sub = []
        for k, m in self._modules.items():
            sub.append(f"({k}): {repr(m)}")
        body = "\n  ".join(sub)
        rep = f"{self.__class__.__name__}({self.extra_repr()})"
        return rep if not sub else f"{rep}\n  {body}"

    def __setattr__(self, key: str, value: Any):
        if isinstance(value, Module):
            self._modules[key] = value
        elif isinstance(value, nn.Parameter):
            self._parameters[key] = value
        super().__setattr__(key, value)

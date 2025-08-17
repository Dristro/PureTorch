import puretorch
import puretorch.nn as nn
import puretorch.nn.functional as F

class TanH(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: puretorch.Tensor):
        return F.tanh(x)

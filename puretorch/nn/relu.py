import puretorch
import puretorch.nn as nn
import puretorch.nn.functional as F

class ReLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: puretorch.Tensor) -> puretorch.Tensor:
        return F.relu(x)

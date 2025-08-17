import puretorch
from puretorch import nn

def summary(model: nn.Module):
    """
    Prints the model's layer name along with the total parameter count.
    """
    raise NotImplementedError(f"Will add this in a future version...")
    # ignore this code, needs to be updated...
    print(f"Model: {[x]}\n")
    print(f"Layer name\t| Num params")
    print(f"---------------\t| ---------")
    sum = 0
    for layer in layers:
        print(f"{layer.__class__.__name__}\t\t| {len(layer.parameters())}")
        sum += len(layer.parameters())
    print(f"Total params: {sum}")

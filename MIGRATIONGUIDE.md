# Migration Guide: v0.x.x to v1.x.x-beta

This guide explains how to migrate your PureTorch projects from `v0.x.x` to `v1.x.x-beta`.\
The `v1.x.x-beta` release introduces breaking changes that make it incompatible with the `v0.x.x` versions.

> **Note:** Some of this content may change by the time this is pushed to the main branch.

---

## Breaking Changes

### 1. Module Structure
- The `PureTorch.layers` module has been replaced by `puretorch.nn`. Update your imports as follows:
    ```python
    # Before (in v0.x.x)
    from PureTorch.layers import Linear

    # After (in v1.x.x+dev)
    from puretorch.nn import Linear
    ```


### 2. Gradient Tracking
- Use the new `puretorch.Tensor` class for tensor operations requiring automatic differentiation.  
  Replace NumPy arrays in your code as follows:
    ```python
    # Before (in v0.x.x)
    import numpy as np
    x = np.array([1.0, 2.0])

    # After (in v1.x.x+dev)
    from puretorch import Tensor
    x = Tensor([1.0, 2.0], requires_grad = True)
    ```

### 3. Using .backward() to calculate gradients
- Use `puretorch.Tensor.backward()` to perform backpropagation on a tensor that requires-grad. Unlike v0.x.x, you can call `.backward` on tensors rather than complete models.
    ```python
    # Before (in v0.x.x)
    # No support for tensor-wise backward passes

    # After (in v1.x.x+dev)
    from puretorch import Tensor
    t1 = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    t2 = Tensor([4.0, 5.0, 6.0], requires_grad=False)
    t3 = t2 - t1
    t3.backward()  # backpropagation
    print(t1.grad)
    print(t2.grad)
    print(t3.grad)
    ```


### 3. Temporarily Unavailable Features
The following features have been removed in `v1.1.0+dev` but will be reintroduced in future updates:

- **Layers**: Flatten

---

## Migration Checklist

- Replace imports from `PureTorch.layers` with `puretorch.nn`.
- Update tensor creation to use `puretorch.Tensor` instead of NumPy arrays.
- Adjust your model architecture to use the new `nn.Linear` and `nn.Sequential`.
- Feel free to use `autograd.function`-base class to define custom tensor operations.

---

## Additional Notes

For further assistance, examples, or to report issues, visit the [GitHub repository](https://github.com/Dristro/PureTorch/issues) or open an issue.

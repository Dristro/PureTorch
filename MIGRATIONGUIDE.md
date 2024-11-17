# Migration Guide: v0.x.x to v1.0.0-beta

This guide explains how to migrate your PureTorch projects from `v0.x.x` to `v1.0.0-beta`.  
The `v1.0.0-beta` release introduces breaking changes that make it incompatible with the `v0.x.x` versions.

**Note:** Some of this content may change by the time this is pushed to the main branch.

---

## Breaking Changes

### 1. Module Structure
- The `PureTorch.layers` module has been replaced by `PureTorch.nn`. Update your imports as follows:
    ```python
    # Before (v0.x.x)
    from PureTorch.layers import Linear

    # After (v1.0.0-beta)
    from PureTorch.nn import Linear
    ```


### 2. Gradient Tracking
- Use the new `PureTorch.Tensor` class for tensor operations requiring automatic differentiation.  
  Replace NumPy arrays in your code as follows:
    ```python
    # Before (v0.x.x)
    import numpy as np
    x = np.array([1.0, 2.0])

    # After (v1.0.0-beta)
    from PureTorch import Tensor
    x = Tensor([1.0, 2.0])
    ```


### 3. Temporarily Unavailable Features
The following features have been removed in `v1.0.0-beta` but will be reintroduced in future updates:

- **Activations** (e.g., ReLU, Softmax)
- **Loss Functions**
- **Optimizers**
- **Layers** (e.g., Flatten)

---

## Migration Checklist

- Replace imports from `PureTorch.layers` with `PureTorch.nn`.
- Update tensor creation to use `PureTorch.Tensor` instead of NumPy arrays.
- Adjust your model architecture to use the new `Linear` and `Perceptron` layers.

---

## Additional Notes

For further assistance, examples, or to report issues, visit the [GitHub repository](https://github.com/Dristro/PureTorch/issues) or open an issue.

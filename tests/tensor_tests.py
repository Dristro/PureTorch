# tests/tensor_tests.py
import os
import sys
sys.path.append(os.getcwd())

import math
import pytest
import numpy as np

import puretorch

from puretorch import Tensor
from puretorch import make_dot
from puretorch.nn import functional as F


# same tests as tests/autograd_tests.py
def test_add():
    t1 = Tensor(np.random.randn(3, 4), requires_grad=True, is_leaf=True)
    t2 = Tensor(np.random.randn(3, 4), requires_grad=True, is_leaf=True)

    out = t1 + t2
    out.backward()

    expected_grad = np.ones_like(out.data)
    assert np.array_equal(t1.grad, expected_grad)
    assert np.array_equal(t2.grad, expected_grad)


def test_sub():
    t1 = Tensor(np.random.randn(3, 4), requires_grad=True, is_leaf=True)
    t2 = Tensor(np.random.randn(3, 4), requires_grad=True, is_leaf=True)

    out = t1 - t2
    out.backward()

    assert np.array_equal(t1.grad, np.ones_like(out.data))
    assert np.array_equal(t2.grad, -np.ones_like(out.data))


def test_mul():
    t1 = Tensor(np.random.randn(3, 4), requires_grad=True, is_leaf=True)
    t2 = Tensor(np.random.randn(3, 4), requires_grad=True, is_leaf=True)

    out = t1 * t2
    out.backward()

    assert np.allclose(t1.grad, t2.data)
    assert np.allclose(t2.grad, t1.data)


def test_div():
    t1 = Tensor(np.random.randn(3, 4), requires_grad=True, is_leaf=True)
    t2 = Tensor(np.random.randn(3, 4) + 1.5, requires_grad=True, is_leaf=True)

    out = t1 / t2
    out.backward()

    expected_t1_grad = 1 / t2.data
    expected_t2_grad = -t1.data / (t2.data ** 2)
    assert np.allclose(t1.grad, expected_t1_grad)
    assert np.allclose(t2.grad, expected_t2_grad)


def test_pow():
    t1 = Tensor(np.random.randn(3, 4) + 2.0, requires_grad=True, is_leaf=True)
    exponent = 3.0
    out = t1 ** exponent
    out.backward()

    expected_grad = exponent * (t1.data ** (exponent - 1))
    assert np.allclose(t1.grad, expected_grad)


def test_neg():
    t1 = Tensor(np.random.randn(3, 4), requires_grad=True, is_leaf=True)
    out = -t1
    out.backward()

    assert np.allclose(t1.grad, -np.ones_like(t1.data))


def test_scalar_ops():
    t1 = Tensor(np.random.randn(3, 4), requires_grad=True, is_leaf=True)

    out = t1 * 2.5 + 5
    out.backward()

    assert np.allclose(t1.grad, np.ones_like(t1.data) * 2.5)


def test_matmul():
    t1 = Tensor(np.random.randn(3, 4), requires_grad=True, is_leaf=True)
    t2 = Tensor(np.random.randn(4, 3), requires_grad=True, is_leaf=True)

    out = t1 @ t2
    out.backward()

    expected_t1_grad = np.ones_like(out.data) @ t2.data.T
    expected_t2_grad = t1.data.T @ np.ones_like(out.data)

    assert out.shape == (3, 3)
    assert np.allclose(t1.grad, expected_t1_grad)
    assert np.allclose(t2.grad, expected_t2_grad)


def test_sum():
    t1 = Tensor(np.random.randn(3, 4), requires_grad=True, is_leaf=True)
    out = t1.sum()
    out.backward()

    assert np.allclose(t1.grad, np.ones_like(t1.data))


def test_sum_dim():
    t1 = Tensor(np.random.randn(3, 4), requires_grad=True, is_leaf=True)
    out = t1.sum(dim=0)
    out.backward(np.ones((4,)))  # upstream gradient shape matches reduced shape

    assert np.allclose(t1.grad, np.ones_like(t1.data))


def test_broadcasting_add():
    t1 = Tensor(np.random.randn(3, 4), requires_grad=True, is_leaf=True)
    t2 = Tensor(np.random.randn(4,), requires_grad=True, is_leaf=True)

    out = t1 + t2
    out.backward(np.ones_like(out.data))

    assert np.allclose(t1.grad, np.ones_like(t1.data))
    assert np.allclose(t2.grad, np.ones_like(t2.data) * 3)  # summed over axis 0


def test_gradient_accumulation():
    t1 = Tensor(np.random.randn(3, 4), requires_grad=True, is_leaf=True)

    out1 = t1 * 2
    out1.backward(np.ones_like(t1.data))

    out2 = t1 + 5
    out2.backward(np.ones_like(t1.data))

    expected_grad = (np.ones_like(t1.data) * 2) + np.ones_like(t1.data)
    assert np.allclose(t1.grad, expected_grad)


def test_requires_grad_false():
    t1 = Tensor(np.random.randn(3, 4), requires_grad=False)
    t2 = Tensor(np.random.randn(3, 4), requires_grad=True, is_leaf=True)

    out = t1 + t2
    out.backward(np.ones_like(out.data))

    assert t1.grad is None
    assert np.allclose(t2.grad, np.ones_like(t2.data))


def test_complex_graph():
    t1 = Tensor(np.random.randn(10, 100), requires_grad=True, is_leaf=True)
    t2 = Tensor(np.random.randn(100, 5), requires_grad=True, is_leaf=True)
    t3 = Tensor(np.random.randn(10, 5), requires_grad=True, is_leaf=True)
    t4 = Tensor(np.random.randn(5, 10), requires_grad=True, is_leaf=True)
    t5 = Tensor(np.random.randn(10, 1), requires_grad=True, is_leaf=True)

    out = t1 @ t2
    out = out + t3
    out = out @ t4
    out = out @ t5
    out.backward()

    assert all(g.grad is not None for g in [t1, t2, t3, t4, t5])
    np.testing.assert_allclose(
        out.data, (((t1.data @ t2.data) + t3.data) @ t4.data) @ t5.data
    )
    assert out.shape == (10, 1)


def test_chain_operations():
    t1 = Tensor(np.random.randn(3, 4), requires_grad=True, is_leaf=True)
    t2 = Tensor(np.random.randn(4,), requires_grad=True, is_leaf=True)

    out = ((t1 + 2) * 3 - t2) / 2
    out = out.sum()
    out.backward()

    assert t1.grad.shape == t1.data.shape
    assert t2.grad.shape == t2.data.shape


def test_multiple_outputs_and_branches():
    t1 = Tensor(np.random.randn(3, 3), requires_grad=True, is_leaf=True)

    out1 = t1 * 2
    out2 = t1 + 3
    final = (out1 + out2).sum()
    final.backward()

    expected_grad = np.ones_like(t1.data) * (2 + 1)
    assert np.allclose(t1.grad, expected_grad)


def test_mean():
    t1 = Tensor(np.random.randn(3, 4), requires_grad=True, is_leaf=True)
    out = t1.mean()
    out.backward()

    expected = np.ones_like(t1.data) / t1.data.size
    assert np.allclose(t1.grad, expected)


def test_advanced_broadcasting_mul():
    t1 = Tensor(np.random.randn(5, 1, 4), requires_grad=True, is_leaf=True)
    t2 = Tensor(np.random.randn(1, 3, 1), requires_grad=True, is_leaf=True)

    out = t1 * t2  # broadcast to (5, 3, 4)
    out.backward(np.ones_like(out.data))

    assert t1.grad.shape == t1.data.shape
    assert t2.grad.shape == t2.data.shape


def test_grad_non_leaf_tensor():
    a = Tensor(np.random.randn(3, 4), requires_grad=True, is_leaf=True)
    b = a * 2
    c = b * 3
    out = c.sum()
    out.backward()
    
    # b is not a leaf, so the gradient is never 'stored'
    assert np.allclose(a.grad, np.ones_like(a.data) * 6)
    assert b.grad is None
    assert c.grad is None
    assert out.grad is None


def test_scalar_tensor():
    a = Tensor(np.array(5.0), requires_grad=True, is_leaf=True)
    b = a * a + 2 * a + 1  # (a + 1)^2
    b.backward()

    expected_grad = 2 * (a.data + 1)
    assert np.allclose(a.grad, expected_grad)


def test_exp_log():
    # keep inputs strictly positive for log()
    rng = np.random.default_rng(0)
    x_data = rng.random((3, 4)) + 0.5  # (0.5, 1.5)

    x = Tensor(x_data, requires_grad=True, is_leaf=True)

    y = x.exp() + x.log()
    out = y.sum()
    out.backward()
    
    expected_grad = np.exp(x_data) + 1.0 / x_data
    assert np.allclose(x.grad, expected_grad, rtol=1e-6, atol=1e-6)

    eps = 1e-6
    num_grad = np.zeros_like(x_data)
    for k in range(x_data.size):
        idx = np.unravel_index(k, x_data.shape)
        x_pos = x_data.copy(); x_pos[idx] += eps
        x_neg = x_data.copy(); x_neg[idx] -= eps
        f_pos = np.sum(np.exp(x_pos) + np.log(x_pos))
        f_neg = np.sum(np.exp(x_neg) + np.log(x_neg))
        num_grad[idx] = (f_pos - f_neg) / (2 * eps)

    assert np.allclose(x.grad, num_grad, rtol=1e-4, atol=1e-4)


def test_gradient_accumulation_multiple_backward():
    t1 = Tensor(np.random.randn(2, 2), requires_grad=True, is_leaf=True)
    out1 = t1 * 2
    out2 = t1 + 4

    out1.backward(np.ones_like(t1.data))
    out2.backward(np.ones_like(t1.data))

    expected = np.ones_like(t1.data) * 3
    assert np.allclose(t1.grad, expected)


def test_clone():
    t1 = Tensor(np.random.randn(3, 4), requires_grad=True, is_leaf=True)
    t2 = t1 + 0  # clone via noop

    out = t2 * 3
    out.sum().backward()

    assert np.allclose(t1.grad, np.ones_like(t1.data) * 3)


def test_zero_grad_behavior():
    t1 = Tensor(np.random.randn(3, 3), requires_grad=True, is_leaf=True)

    # First backward
    out = t1 * 2
    out.sum().backward()
    first_grad = t1.grad.copy()
    assert np.allclose(first_grad, 2 * np.ones_like(t1.data))

    # Zero grad should set grad to None
    t1.zero_grad()
    assert t1.grad is None

    # Backward again with a new forward
    out = t1 * 3
    out.sum().backward()
    assert np.allclose(t1.grad, 3 * np.ones_like(t1.data))


def test_reshape_and_transpose():
    t1 = Tensor(np.random.randn(2, 6), requires_grad=True, is_leaf=True)

    t2 = t1.reshape((3, 4))
    t3 = t2.T  # transpose
    out = t3.sum()
    out.backward()

    # The derivative of sum wrt every element is 1
    assert np.allclose(t1.grad, np.ones_like(t1.data))

def test_inplace_breaks_ctx_snapshot():
    a = Tensor(np.random.randn(3,3), requires_grad=True, is_leaf=True)
    out = (a * 2).sum()        # snapshots a._version
    a.add_(1.0)                # bump a._version
    with pytest.raises(RuntimeError):
        out.backward()

def test_inplace_on_nonleaf_disallowed():
    a = Tensor(np.random.randn(3,3), requires_grad=True, is_leaf=True)
    b = a * 2                  # non-leaf that requires grad
    with pytest.raises(RuntimeError):
        b.add_(1.0)

def test_inplace_update_bw_out():
    a = Tensor(np.random.randn(3,3), requires_grad=True, is_leaf=True)
    out = (a * 2).sum()        # snapshots a._version
    a += a                # bump a._version
    with pytest.raises(RuntimeError):
        out.backward()

def test_iadd_bumps_version_and_breaks_backward():
    a = Tensor(np.ones((2,2)), requires_grad=True, is_leaf=True)
    out = (a * 2).sum()
    a += 1.0                 # __iadd__ -> bump version
    import pytest
    with pytest.raises(RuntimeError):
        out.backward()

def test_setitem_bumps_version():
    a = Tensor(np.zeros((3,)), requires_grad=True, is_leaf=True)
    out = (a + 1).sum()
    a[1] = 5                 # __setitem__ -> bump version
    import pytest
    with pytest.raises(RuntimeError):
        out.backward()

def test_data_write_is_blocked():
    a = Tensor(np.zeros((2,)), requires_grad=True, is_leaf=True)
    try:
        a.data += 1          # read-only view -> should raise
        raised = False
    except ValueError:
        raised = True
    assert raised

def test_data_reassign_is_tracked():
    a = Tensor(np.zeros((2,)), requires_grad=True, is_leaf=True)
    out = (a + 1).sum()
    a.data = np.ones((2,))   # goes through setter -> bump version
    import pytest
    with pytest.raises(RuntimeError):
        out.backward()


# new tests for Tensor
def test_no_grad_tensor():
    with puretorch.no_grad():
        t1 = Tensor(data=[1.0], requires_grad=True)
        t2 = Tensor(data=[1.0], requires_grad=True)
    out1 = t1 * 5
    out2 = t2 * 5
    with pytest.raises(RuntimeError):
        out1.backward()
        out2.backward()
    assert t1.grad is None
    assert t2.grad is None

def test_enable_grad():
    with puretorch.enable_grad():
        t1 = Tensor(data=[1.0], requires_grad=True)
        t2 = Tensor(data=[1.0], requires_grad=False)
    out1 = t1 * 5
    out2 = t2 * 5
    out1.backward()
    with pytest.raises(RuntimeError):
        out2.backward()
    assert t1.grad is not None
    assert t2.grad is None

def test_cross_entropy_grad_matches_softmax_minus_onehot():
    rng = np.random.default_rng(0)
    B, C = 5, 7
    x = Tensor(rng.standard_normal((B, C)), requires_grad=True, is_leaf=True)
    tgt = rng.integers(0, C, size=(B,))

    # forward/backward
    loss = F.cross_entropy(x, tgt, reduction="mean")
    loss.backward()

    # expected grad: (softmax - onehot)/B
    s = F.softmax(x).numpy()
    oh = np.zeros((B, C)); oh[np.arange(B), tgt] = 1.0
    expected = (s - oh) / B
    assert np.allclose(x.grad, expected, rtol=1e-6, atol=1e-6)


def test_relu_positive_values():
    x = Tensor([1.0, 2.5, 100.0])
    y = F.relu(x)
    expected = Tensor([1.0, 2.5, 100.0])
    assert puretorch.equal(y, expected)


def test_relu_negative_values():
    x = Tensor([-1.0, -5.0, -0.001])
    y = F.relu(x)
    expected = puretorch.zeros_like(x)
    assert puretorch.equal(y, expected)


def test_relu_zero_boundary():
    x = Tensor([0.0])
    y = F.relu(x)
    expected = Tensor([0.0])
    assert puretorch.equal(y, expected)


@pytest.mark.parametrize("val", [0.0, 1.0, -1.0, 2.5, -2.5])
def test_tanh_scalar_against_math(val):
    x = Tensor(val)
    y = F.tanh(x).item()
    expected = math.tanh(val)  # reference definition
    assert math.isclose(y, expected, rel_tol=1e-6, abs_tol=1e-6)


def test_tanh_symmetry():
    # tanh is odd: tanh(-x) == -tanh(x)
    x = puretorch.linspace(-3, 3, num=10)
    y_pos = F.tanh(x)
    y_neg = F.tanh(-x)
    assert puretorch.allclose(y_neg, -y_pos, atol=1e-6)


def test_tanh_range():
    x = puretorch.linspace(-100, 100, num=50)
    y = F.tanh(x)
    # all outputs must be strictly between -1 and 1
    #print(f"[DEBUG @ tests/tensor_tests.py] y: {y}")  # passed, remove in next commit
    #print(f"[DEBUG @ tests/tensor_tests.py] y.shape: {y.shape}")
    assert np.all(y >= -1.0)
    assert np.all(y <= 1.0)


def test_tanh_zero_boundary():
    x = Tensor([0.0])
    y = F.tanh(x)
    expected = Tensor([0.0])
    assert puretorch.allclose(y, expected, atol=1e-8)


def my_func():
    t1 = Tensor(np.random.randn(3, 3), requires_grad=True, is_leaf=True)

    out1 = t1 * 2
    out2 = t1 + 3
    final = (out1 + out2).sum()
    final.backward()

    print(f"[FUN @ tests/tensor_tests.py] t1:\n{t1}")
    print("---"*5)
    print(f"[FUN @ tests/tensor_tests.py] out1:\n{out1}")
    print("---"*5)
    print(f"[FUN @ tests/tensor_tests.py] out2:\n{out2}")
    print("---"*5)
    print(f"[FUN @ tests/tensor_tests.py] final:\n{final}")
    print("---"*5)

    rng = np.random.default_rng(0)
    B, C = 5, 7
    x = Tensor(rng.standard_normal((B, C)), requires_grad=True, is_leaf=True)
    tgt = rng.integers(0, C, size=(B,))
    print(f"[FUN @ tests/tensor_tests.py] tgt.shape: {tgt.shape}")
    print(f"[FUN @ tests/tensor_tests.py] tgt:\n{tgt}")
    print("---"*5)

if __name__ == "__main__":
    my_func()
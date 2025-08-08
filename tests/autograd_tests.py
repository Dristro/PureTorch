import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from autograd import Tensor

def test_sum():
    t1 = Tensor(data=np.random.randn(3, 4), requires_grad=True, grad_fn=None, is_leaf=True)
    t2 = Tensor(data=np.random.randn(3, 4), requires_grad=True, grad_fn=None, is_leaf=True)
    
    out = t1 + t2
    out.backward(gradient=None)  # assumes grad = 1 | shape: (3, 4)
    
    t1_expected_grad = np.ones_like(out.data)
    t2_expected_grad = np.ones_like(out.data)
    
    assert np.array_equal(t1.grad, t1_expected_grad)
    assert np.array_equal(t2.grad, t2_expected_grad)

def test_mul():
    t1 = Tensor(data=np.random.randn(3, 4), requires_grad=True, grad_fn=None, is_leaf=True)
    t2 = Tensor(data=np.random.randn(3, 4), requires_grad=True, grad_fn=None, is_leaf=True)
    
    out = t1 * t2
    out.backward(gradient=None)  # assumes grad = 1 | shape: (3, 4)
    
    t1_expected_grad = t2.data
    t2_expected_grad = t1.data
    
    assert np.array_equal(t1.grad, t1_expected_grad)
    assert np.array_equal(t2.grad, t2_expected_grad)

def test_sub():
    t1 = Tensor(data=np.random.randn(3, 4), requires_grad=True, grad_fn=None, is_leaf=True)
    t2 = Tensor(data=np.random.randn(3, 4), requires_grad=True, grad_fn=None, is_leaf=True)

    out = t1 - t2
    out.backward(gradient=None)

    t1_expected_grad = np.ones_like(out.data)
    t2_expected_grad = -np.ones_like(out.data)
    
    assert np.array_equal(t1.grad, t1_expected_grad)
    assert np.array_equal(t2.grad, t2_expected_grad)

def test_matmul():
    t1 = Tensor(data=np.random.randn(3, 4), requires_grad=True, grad_fn=None, is_leaf=True)
    t2 = Tensor(data=np.random.randn(4, 3), requires_grad=True, grad_fn=None, is_leaf=True)

    out = t1 @ t2
    out.backward(gradient=None)

    t1_expected_grad = np.ones_like(out.data) @ t2.data.T
    t2_expected_grad = t1.data.T @ np.ones_like(out.data)
    
    assert out.shape == (3, 3)
    assert np.array_equal(t1.grad, t1_expected_grad)
    assert np.array_equal(t2.grad, t2_expected_grad)

def test_complex_graph():
    t1 = Tensor(data=np.random.randn(10, 100), requires_grad=True, is_leaf=True)
    t2 = Tensor(data=np.random.randn(100, 5), requires_grad=True, is_leaf=True)
    t3 = Tensor(data=np.random.randn(10, 5), requires_grad=True, is_leaf=True)
    t4 = Tensor(data=np.random.randn(5, 10), requires_grad=True, is_leaf=True)
    t5 = Tensor(data=np.random.randn(10, 1), requires_grad=True, is_leaf=True)

    out = t1 @ t2     # (10, 100) @ (100, 5) = (10, 5)
    out = out + t3    # (10, 5) + (10, 5) = (10, 5)
    out = out @ t4    # (10, 5) @ (5, 10) = (10, 10)
    out = out @ t5    #(10, 10) @ (10, 1) = (10, 1)
    out.backward(gradient=None)

    assert all([
        t1.grad is not None,
        t2.grad is not None,
        t3.grad is not None,
        t4.grad is not None,
        t5.grad is not None,
        ])
    assert np.array_equal(out.data, (((t1.data @ t2.data) + t3.data) @ t4.data) @ t5.data)
    assert out.shape == (10, 1)
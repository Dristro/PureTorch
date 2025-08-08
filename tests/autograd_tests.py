import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from autograd import Tensor

def test_sum():
    t1 = Tensor(data=np.random.randn(3, 4), requires_grad=True, grad_fn=None, is_leaf=True)
    t2 = Tensor(data=np.random.randn(3, 4), requires_grad=True, grad_fn=None, is_leaf=True)
    
    sum = t1 + t2
    sum.backward(gradient=None)  # assumes grad = 1 | shape: (3, 4)
    
    t1_expected_grad = np.ones_like(sum.data)
    t2_expected_grad = np.ones_like(sum.data)
    
    assert np.array_equal(t1.grad, t1_expected_grad)
    assert np.array_equal(t2.grad, t2_expected_grad)

def test_mul():
    t1 = Tensor(data=np.random.randn(3, 4), requires_grad=True, grad_fn=None, is_leaf=True)
    t2 = Tensor(data=np.random.randn(3, 4), requires_grad=True, grad_fn=None, is_leaf=True)
    
    sum = t1 * t2
    sum.backward(gradient=None)  # assumes grad = 1 | shape: (3, 4)
    
    t1_expected_grad = t2.data
    t2_expected_grad = t1.data
    
    assert np.array_equal(t1.grad, t1_expected_grad)
    assert np.array_equal(t2.grad, t2_expected_grad)
# context manager for gradients #

import contextlib
_grad_enabled = True

def is_grad_enabled() -> bool:
    return _grad_enabled

@contextlib.contextmanager
def no_grad():
    global _grad_enabled
    old = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = old

@contextlib.contextmanager
def enable_grad():
    global _grad_enabled
    old = _grad_enabled
    _grad_enabled = True
    try:
        yield
    finally:
        _grad_enabled = old

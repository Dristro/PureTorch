from .context import Context

class Function:
    """Base class for autograd functions."""
    def forward(ctx: Context, *args, **kwargs):
        raise NotImplementedError

    def barkward(ctx: Context, grad_output):
        raise NotImplementedError

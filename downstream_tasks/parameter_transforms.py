"""In-place transforms of parameter values."""


def clamp(psi, min_, max_):
    psi.clamp_(min=min_, max=max_)

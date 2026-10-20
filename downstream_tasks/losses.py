import torch


def squared_error(x, x_hat, reduction, feature_tranform, weight=1.0):

    if feature_tranform is not None:
        x = feature_tranform(x)
        x_hat = feature_tranform(x_hat)

    e2 = x_hat - x
    if torch.is_complex(x):
        e2 = (e2.conj() * e2).real
    else:
        e2 = e2 ** 2

    if reduction == 'mean':
        return weight * e2.mean()
    elif reduction == 'sum':
        return weight * e2.sum()
    elif reduction == 'sum_over_frequency_+_mean_over_rest':
        return weight * e2.sum(dim=-2).mean()
    else:
        raise ValueError

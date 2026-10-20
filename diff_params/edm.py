import torch

from diff_params.shared import SDE


class EDM(SDE):
    """EDM parameterisation (Karras et al., 2022), with sigma = t."""

    def __init__(self, type, sde_hp):
        super().__init__(type, sde_hp)
        self.sigma_data = self.sde_hp.sigma_data  # variance of the training data
        self.sigma_min = self.sde_hp.sigma_min
        self.sigma_max = self.sde_hp.sigma_max
        self.rho = self.sde_hp.rho

    # training
    def sample_time_training(self, batch_size):
        """Noise levels for training: uniform in the EDM schedule's rho-warped space between sigma_max and sigma_min."""
        i = torch.rand(batch_size)
        t = (self.sigma_max ** (1 / self.rho) + i * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
        return t

    def sample_prior(self, shape):
        return torch.randn(shape)

    def prepare_train_preconditioning(self, x, t, n=None):
        """Noisy network input cin * (x + t n), training target (x - cskip (x + t n)) / cout, and cnoise."""
        mu, sigma = self._mean(x, t), self._std(t).unsqueeze(-1)
        sigma = sigma.view(*sigma.size(), *(1,) * (x.ndim - sigma.ndim))
        if n is None:
            n = self.sample_prior(x.shape).to(x.device)
        x_perturbed = mu + sigma * n

        cskip = self.cskip(sigma)
        cout = self.cout(sigma)
        cin = self.cin(sigma)
        cnoise = self.cnoise(sigma.squeeze())

        if len(cnoise.shape) == 0:
            cnoise = cnoise.repeat(x.shape[0],)
        else:
            cnoise = cnoise.view(x.shape[0],)

        target = 1 / cout * (x - cskip * x_perturbed)
        return cin * x_perturbed, target, cnoise

    # preconditioning
    def cskip(self, sigma):
        return self.sigma_data**2 * (sigma**2 + self.sigma_data**2)**-1

    def cout(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data**2 + sigma**2)**(-0.5)

    def cin(self, sigma):
        return (self.sigma_data**2 + sigma**2)**(-0.5)

    def cnoise(self, sigma):
        return (1 / 4) * torch.log(sigma)

    def Tweedie2score(self, tweedie, xt, t, *args, **kwargs):
        return (tweedie - self._mean(xt, t)) / self._std(t)**2

    def _mean(self, x, t):
        return x

    def _std(self, t):
        return t

    def _ode_integrand(self, x, t, score):
        return -t * score


class EDM_cond(EDM):
    """EDM with a conditioning input (the DoA) to the denoiser."""

    def denoiser(self, xn, net, t, cond, *args, **kwargs):
        """Preconditioned denoiser D(xn; t) = cskip * xn + cout * net(cin * xn, cnoise, cond)."""
        sigma = self._std(t).unsqueeze(-1)
        sigma = sigma.view(*sigma.size(), *(1,) * (xn.ndim - sigma.ndim))

        cskip = self.cskip(sigma)
        cout = self.cout(sigma)
        cin = self.cin(sigma)
        cnoise = self.cnoise(sigma.squeeze())

        if len(cnoise.shape) == 0:
            cnoise = cnoise.repeat(xn.shape[0],)
        else:
            cnoise = cnoise.view(xn.shape[0],)

        return cskip * xn + cout * net(cin * xn, cnoise, cond=cond)

    def loss_fn(self, net, x, n=None):
        """
        Denoising score matching loss for a batch dict(features=(B, 4, F), context=DoA (B, 3)): squared error between
        the network output and the preconditioned target, per element. Returns (error ** 2, sigma).
        """
        features, context = x['features'], x['context']
        t = self.sample_time_training(features.shape[0]).to(features.device)
        input, target, cnoise = self.prepare_train_preconditioning(features, t, n=n)
        estimate = net(input, cnoise, cond=context)
        error = estimate - target
        return error ** 2, self._std(t)

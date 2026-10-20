from functools import partial

import torch
from tqdm import tqdm

from testing.SamplerWithSimpleSetup import SamplerWithSimpleSetup as Sampler


class EulerHeunSampler(Sampler):
    """
    Euler / Heun sampler of Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models", 2022
    (Algorithm 2, referred to as [1] below).
    """

    def __init__(self, model, diff_params, T, Schurn, Snoise, Stmin, Stmax, order, schedule_mode):
        super().__init__(model, diff_params, T, schedule_mode)
        # stochasticity parameters
        self.Schurn = Schurn
        self.Snoise = Snoise
        self.Stmin = Stmin
        self.Stmax = Stmax
        self.order = order

    @staticmethod
    def initialize_x(shape, device, schedule):
        x = schedule[0] * torch.randn(shape).to(device)
        return x

    @staticmethod
    def get_gamma(t, Stmin, Stmax, Schurn):
        """Stochasticity gamma_i for each timestep in t (non-zero only for Stmin < t < Stmax)."""
        N = t.shape[0]
        gamma = torch.zeros(t.shape).to(t.device)
        indexes = torch.logical_and(t > Stmin, t < Stmax)
        gamma[indexes] = gamma[indexes] + torch.min(torch.Tensor([Schurn / N, 2**(1 / 2) - 1]))
        return gamma

    @staticmethod
    def stochastic_timestep(x, t, gamma, Snoise=1):
        """Steps 4, 5 and 6 of Algorithm 2 [1]."""
        epsilon = torch.randn(x.shape).to(x.device) * Snoise
        t_hat = t + gamma * t
        x_hat = x + ((t_hat ** 2 - t ** 2) ** (1 / 2)) * epsilon
        return x_hat, t_hat

    @staticmethod
    def step(x_i, t_i, t_iplus1, gamma_i, stochastic_timestep, get_Tweedie_estimate, Tweedie2score, ode_integrand, order):
        # steps 4, 5 and 6
        x_hat, t_hat = stochastic_timestep(x_i, t_i, gamma_i)

        # step 7: denoised estimate D(x_hat; t_hat) and ODE derivative d
        x_den = get_Tweedie_estimate(x_hat, t_hat)
        score = Tweedie2score(x_den, x_hat, t_hat)
        ode_integrand_ = ode_integrand(x_hat, t_hat, score)

        # step 8
        dt = t_iplus1 - t_hat

        if t_iplus1 != 0 and order == 2:
            # steps 9-11: second order correction
            x_prime = x_hat + dt * ode_integrand_
            x_den = get_Tweedie_estimate(x_prime, t_iplus1)
            score = Tweedie2score(x_den, x_prime, t_iplus1)
            ode_integrand_next = ode_integrand(x_prime, t_iplus1, score)
            ode_integrand_midpoint = .5 * (ode_integrand_ + ode_integrand_next)
            x_iplus1 = x_hat + dt * ode_integrand_midpoint
        else:
            x_iplus1 = x_hat + dt * ode_integrand_

        return x_iplus1, x_den

    def predict(self, shape, device, x_T=None):
        t = self.create_schedule().to(device)

        if x_T is None:
            assert shape is not None, 'specify shape when x_T is not given'
            x = EulerHeunSampler.initialize_x(shape, device, t)
        else:
            assert shape is None, 'x_T is set, no need to specify shape'
            x = x_T

        gamma = EulerHeunSampler.get_gamma(t, Stmin=self.Stmin, Stmax=self.Stmax, Schurn=self.Schurn).to(device)

        step = partial(EulerHeunSampler.step,
                       stochastic_timestep=EulerHeunSampler.stochastic_timestep,
                       get_Tweedie_estimate=self.get_Tweedie_estimate,
                       Tweedie2score=self.Tweedie2score,
                       ode_integrand=self.diff_params._ode_integrand,
                       order=self.order)

        for i in tqdm(range(0, self.T, 1)):
            x, x_den = step(x, t[i], t[i + 1], gamma[i])

        return x

    def predict_unconditional(self, device, shape=None, x_T=None):
        with torch.no_grad():
            return self.predict(shape, device, x_T=x_T)


class EulerHeunSampler_doa_cond(EulerHeunSampler):
    """EulerHeunSampler conditioned on the direction of arrival (DoA)."""

    def load_DoA(self, doa):
        self.doa = doa.detach().clone()

    def get_Tweedie_estimate(self, x, t_i):
        assert self.doa is not None, 'DoA is not set, call load_DoA first'
        if x.ndim == 2 and x.shape[0] == 1:
            x_hat = self.diff_params.denoiser(x.unsqueeze(1), self.model, t_i, self.doa).squeeze(1)
        else:
            assert x.ndim == 3, 'expecting shape (batch, channel, frequency)'
            x_hat = self.diff_params.denoiser(x, self.model, t_i, self.doa)
        return x_hat

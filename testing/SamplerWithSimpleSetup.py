import torch


class SamplerWithSimpleSetup():

    def __init__(self, model, diff_params, T, schedule_mode):
        assert not model.training, 'Model must be set to eval mode prior to building the Sampler.'
        self.model = model
        self.diff_params = diff_params
        self.T = T
        self.schedule_mode = schedule_mode

    def create_schedule(self, sigma_min=None, sigma_max=None, rho=None, T=None):
        if sigma_min is None:
            sigma_min = self.diff_params.sde_hp.sigma_min
        if sigma_max is None:
            sigma_max = self.diff_params.sde_hp.sigma_max
        if rho is None:
            rho = self.diff_params.sde_hp.rho
        if T is None:
            T = self.T

        if self.schedule_mode == "edm":
            # Eq. (5) of Karras et al., 2022
            i = torch.arange(0, T + 1)
            sigma_t = (sigma_max ** (1 / rho) + i / (T - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
            sigma_t[-1] = 0
            return sigma_t
        else:
            raise ValueError(f"unknown schedule_mode {self.schedule_mode}")

    def Tweedie2score(self, tweedie, xt, t):
        return self.diff_params.Tweedie2score(tweedie, xt, t)

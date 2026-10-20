"""
1D U-Net denoiser with DoA conditioning (FiLM on random Fourier feature embeddings of the noise level and the DoA).

Adapted from https://github.com/01tot10/neural-tape-modeling/blob/master/code/networks/unet_1d.py
Authors: Eloi Moliner, Otto Mikkonen
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

torch.pi = torch.acos(torch.zeros(1)).item() * 2  # 3.1415927410125732


class UNet1D(nn.Module):

    def __init__(self, in_features, hidden_features, resampling_factors, kernel_size, use_bias, use_norm,
                 in_features_context, n_fourier_features, n_fourier_frequencies):
        super(UNet1D, self).__init__()

        assert len(resampling_factors) == len(hidden_features)

        self.use_norm = use_norm
        self.hidden_features = hidden_features
        self.resampling_factors = resampling_factors

        # embeddings of the conditioning inputs (DoA and noise level), one per input
        self.embeddings = nn.ModuleList([RFF_MLP_Block(emb_dim=n_fourier_features, n_fourier_frequencies=n_fourier_frequencies) for _ in range(in_features_context)])

        self.init_conv = nn.Conv1d(in_features,
                                   self.hidden_features[0],
                                   kernel_size=kernel_size,
                                   padding="same",
                                   padding_mode="zeros",
                                   bias=use_bias)

        # the combiners are unused in forward() but part of the checkpoint
        self.downs = nn.ModuleList([])
        self.middle = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        features_in = hidden_features[:-1]
        features_out = hidden_features[1:]
        self.n_blocks = len(features_in)
        for i, (d_in, d_out) in enumerate(zip(features_in, features_out)):

            if i < self.n_blocks - 1:
                self.downs.append(
                    nn.ModuleList([
                        ResnetBlock(d_in,
                                    d_out,
                                    self.use_norm,
                                    emb_dim=in_features_context*n_fourier_features,
                                    bias=use_bias,
                                    kernel_size=kernel_size),
                        Downsample(self.resampling_factors[i]),
                        CombinerDown("sum", 1, d_out, bias=use_bias)
                    ]))

            else:
                # no downsampling in last layer
                assert i == self.n_blocks - 1
                self.downs.append(
                    nn.ModuleList([
                        ResnetBlock(d_in,
                                    d_out,
                                    self.use_norm,
                                    bias=use_bias,
                                    emb_dim=in_features_context*n_fourier_features),
                    ]))

        self.middle.append(
            nn.ModuleList([
                ResnetBlock(d_out,
                            d_out,
                            self.use_norm,
                            bias=use_bias,
                            emb_dim=in_features_context*n_fourier_features)
            ]))

        features_in, features_out = reversed(features_out), reversed(features_in)
        for i, (d_in, d_out) in enumerate(zip(features_in, features_out)):

            if i < self.n_blocks - 1:
                self.ups.append(
                    nn.ModuleList([
                        ResnetBlock(d_in * 2,
                                    d_out,
                                    use_norm=self.use_norm,
                                    bias=use_bias,
                                    emb_dim=in_features_context*n_fourier_features),
                        Upsample(self.resampling_factors[i]),
                        CombinerUp("sum", 1, d_out, bias=use_bias)
                    ]))

            else:
                # no upsampling in last layer
                assert i == self.n_blocks - 1
                self.ups.append(
                    nn.ModuleList([
                        ResnetBlock(d_in * 2,
                                    d_out,
                                    use_norm=self.use_norm,
                                    bias=use_bias,
                                    emb_dim=in_features_context*n_fourier_features),
                    ]))

        self.out_conv = nn.Conv1d(d_out,
                                  in_features,
                                  kernel_size=kernel_size,
                                  padding="same",
                                  padding_mode="zeros",
                                  bias=use_bias)

    def forward(self, x, sigma, cond=None):
        if x.ndim == 2:
            # add a singleton batch dimension
            x_ = x.unsqueeze(0)
            if sigma.ndim == 0:
                sigma = sigma.unsqueeze_(0)
        else:
            assert x.ndim == 3, 'expecting input with shape (batch, channel, spatial dimension e.g. time or frequency)'
            assert x.shape[0] == sigma.shape[0], 'expecting matching batch dimensions'
            if sigma.ndim == 1:
                sigma = sigma.unsqueeze(-1)  # channel dimension
            x_ = x

        if cond is not None:
            # concatenate the conditioning (DoA) to sigma
            if cond.shape[0] != sigma.shape[0] and cond.shape[0] == 1:
                cond = cond.expand(sigma.shape[0], -1)
            sigma = torch.cat((cond, sigma), axis=-1)

        z = list()
        for i, embedding in enumerate(self.embeddings):
            z.append(embedding(sigma[:, i:i + 1]))
        sigma = torch.cat(z, dim=-1)
        del z

        x_ = self.init_conv(x_)

        hs = []
        for i, modules in enumerate(self.downs):

            if i < self.n_blocks - 1:
                assert len(modules) == 3
                resnet, downsample, combiner = modules
                x_ = resnet(x_, sigma)
                hs.append(x_)
                x_ = downsample(x_)

            else:
                # no downsampling in the last layer
                assert i == self.n_blocks - 1
                assert len(modules) == 1
                (resnet,) = modules
                x_ = resnet(x_, sigma)
                hs.append(x_)

        for modules in self.middle:
            (resnet,) = modules
            x_ = resnet(x_, sigma)

        for i, modules in enumerate(self.ups):

            if i < self.n_blocks - 1:
                assert len(modules) == 3
                resnet, upsample, combiner = modules

                skip = hs.pop()
                x_ = torch.cat((x_, skip), dim=1)
                x_ = resnet(x_, sigma)
                x_ = upsample(x_)

            else:
                # no upsampling in the last layer
                assert i == self.n_blocks - 1
                assert len(modules) == 1
                (resnet,) = modules
                x_ = torch.cat((x_, hs.pop()), dim=1)
                x_ = resnet(x_, sigma)

        x_ = self.out_conv(x_)

        if x.ndim == 2:
            # remove the singleton batch dimension
            assert x_.shape[0] == 1
            x_ = x_.squeeze(0)

        assert x.shape == x_.shape, "bad shapes"

        return x_


class RFF_MLP_Block(nn.Module):

    def __init__(self, emb_dim, n_fourier_frequencies):
        super().__init__()

        self.RFF_freq = nn.Parameter(n_fourier_frequencies * torch.randn([1, n_fourier_frequencies]),
                                     requires_grad=False)

        self.MLP = nn.ModuleList([
            nn.Linear(2 * n_fourier_frequencies, emb_dim),
            nn.Linear(emb_dim, emb_dim),
        ])

    def forward(self, sigma):
        """sigma: (B, 1) -> embedding (B, emb_dim)"""
        assert sigma.ndim == 2
        assert sigma.shape[1] == 1

        x = self._build_RFF_embedding(sigma)
        for layer in self.MLP:
            x = F.relu(layer(x))

        return x

    def _build_RFF_embedding(self, sigma):
        """sigma: (B, 1) -> random Fourier features (B, 2 * n_fourier_frequencies)"""
        freqs = self.RFF_freq
        table = 2 * np.pi * sigma * freqs
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)

        return table


class ResnetBlock(nn.Module):

    def __init__(
        self,
        dim,
        dim_out,
        use_norm=False,
        groups=8,
        emb_dim=32,
        num_dils=8,
        bias=True,
        kernel_size=5
    ):
        super().__init__()
        self.bias = bias
        self.use_norm = use_norm
        self.num_layers = num_dils

        self.film = Film(dim, emb_dim, bias=bias)
        self.res_conv = nn.Conv1d(
            dim, dim_out, 1, padding_mode="zeros",
            bias=bias) if dim != dim_out else nn.Identity()
        if self.use_norm:
            self.gnorm = nn.GroupNorm(groups, dim)
        self.first_conv = nn.Sequential(nn.GELU(),
                                        nn.Conv1d(dim, dim_out, 1, bias=bias))

        self.H = nn.ModuleList()
        for i in range(self.num_layers):
            self.H.append(
                GatedResidualLayer(dim_out, kernel_size, 2**i, bias=bias)
            )

    def forward(self, x, sigma):
        gamma, beta = self.film(sigma)

        if self.use_norm:
            x = self.gnorm(x)

        if self.bias:
            x = x * gamma + beta
        else:
            x = x * gamma

        y = self.first_conv(x)

        for h in self.H:
            y = h(y)

        return (y + self.res_conv(x)) / (2 ** 0.5)


class Film(nn.Module):

    def __init__(self, output_dim, emb_dim, bias=True):
        super().__init__()
        self.bias = bias
        if bias:
            self.output_layer = nn.Linear(emb_dim, 2 * output_dim)
        else:
            self.output_layer = nn.Linear(emb_dim, 1 * output_dim)

    def forward(self, sigma_encoding):
        sigma_encoding = self.output_layer(sigma_encoding)
        sigma_encoding = sigma_encoding.unsqueeze(-1)
        if self.bias:
            gamma, beta = torch.chunk(sigma_encoding, 2, dim=1)
        else:
            gamma = sigma_encoding
            beta = None

        return gamma, beta


class GatedResidualLayer(nn.Module):

    def __init__(self, dim, kernel_size, dilation, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(dim,
                              dim,
                              kernel_size=kernel_size,
                              dilation=dilation,
                              stride=1,
                              padding='same',
                              padding_mode='zeros',
                              bias=bias)
        self.act = nn.GELU()

    def forward(self, x):
        x = (x + self.conv(self.act(x))) / (2 ** 0.5)
        return x


class Upsample(nn.Module):

    def __init__(self, upsampling_factor):
        super().__init__()
        self.resample = torchaudio.transforms.Resample(1, upsampling_factor)

    def forward(self, x):
        return self.resample(x)


class Downsample(nn.Module):

    def __init__(self, downsampling_factor):
        super().__init__()
        self.resample = torchaudio.transforms.Resample(downsampling_factor, 1)

    def forward(self, x):
        return self.resample(x)


class CombinerUp(nn.Module):

    def __init__(self, mode, Npyr, Nx, bias=True):
        super().__init__()
        self.mode = mode
        self.conv1x1 = nn.Conv1d(Nx, Npyr, 1, bias=bias)
        torch.nn.init.constant_(self.conv1x1.weight, 0)

    def forward(self, pyr, x):
        if self.mode == "sum":
            x = self.conv1x1(x)
            if pyr is None:
                return x
            else:
                return (pyr[..., 0:x.shape[-1]] + x) / (2 ** 0.5)
        else:
            raise NotImplementedError


class CombinerDown(nn.Module):

    def __init__(self, mode, Nin, Nout, bias=True):
        super().__init__()
        self.mode = mode
        self.conv1x1 = nn.Conv1d(Nin, Nout, 1, bias=bias)

    def forward(self, pyr, x):
        if self.mode == "sum":
            pyr = self.conv1x1(pyr)
            return (pyr + x) / (2 ** 0.5)
        else:
            raise NotImplementedError

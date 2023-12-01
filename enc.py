from abc import ABC

import torch
from torch import Tensor

from model import MLP

class Encoder(ABC):
    type: str

    def encode(self, mu: Tensor, cov: Tensor) -> Tensor:
        pass

    def config(self) -> dict:
        pass

def build_model(enc_settings: dict, model_settings: dict):
    enc_type = enc_settings['type']
    enc_settings.pop('type')

    match enc_type:
        case PE.type:
            enc = PE(**enc_settings)
        case IPE.type:
            enc = IPE(**enc_settings)
        case _:
            raise ValueError(f'invalid encoding type: {enc_type}')

    model = MLP(**model_settings)
    return enc, model

class PE(Encoder):
    type = 'PE'

    def __init__(self, d_in: int, n_freqs: int):
        self.d_in = d_in
        self.n_freqs = n_freqs
        self.d_output = d_in * (2 * self.n_freqs)

        # compute frequencies
        self.freq_powers = torch.arange(0, n_freqs)
        self.frequencies = 2 ** self.freq_powers

        # compute fourrier feature matrix
        self.P = torch.hstack([torch.eye(d_in) * f for f in self.frequencies])

    def config(self):
        return dict(
            type=self.type,
            d_in=self.d_in,
            n_freqs=self.n_freqs
        )

    def encode(self, mu_batch: Tensor, cov_batch: Tensor) -> Tensor:
        px = mu_batch @ self.P
        gamma = torch.cat([torch.sin(px),
                           torch.cos(px)], dim=1)

        return gamma

class IPE(Encoder):
    type = 'IPE'

    def __init__(self, d_in: int, n_freqs: int):
        self.d_in = d_in
        self.n_freqs = n_freqs
        self.d_output = d_in * (2 * self.n_freqs)

        # compute frequencies
        self.freq_powers = torch.arange(0, n_freqs)
        self.frequencies = 2 ** self.freq_powers

        # compute fourrier feature matrix
        self.P = torch.hstack([torch.eye(d_in) * f for f in self.frequencies])

    def config(self):
        return dict(
            type=self.type,
            d_in=self.d_in,
            n_freqs=self.n_freqs
        )

    def encode(self, mu_batch: Tensor, cov_batch: Tensor) -> Tensor:
        mu_gamma = mu_batch @ self.P

        diags = torch.diagonal(cov_batch, dim1=2, dim2=1)
        cov_gamma_diags = torch.cat([(4 ** L) * diags for L in self.freq_powers], dim=1)

        gamma = torch.cat(
            [torch.sin(mu_gamma) * torch.exp(-0.5 * cov_gamma_diags),
             torch.cos(mu_gamma) * torch.exp(-0.5 * cov_gamma_diags)], dim=1)

        return gamma

class AltEnc(Encoder):
    type = 'IPE'

    def __init__(self, d_in: int, n_freqs: int):
        self.d_in = d_in
        self.n_freqs = n_freqs

        # compute frequencies
        self.pe_L = PE(d_in, n_freqs)
        self.pe_2 = PE(d_in ** 2, 2)

        self.d_output = self.pe_2.d_output + self.pe_L.d_output

    def config(self):
        return dict(
            type=self.type,
            d_in=self.d_in,
            n_freqs=self.n_freqs
        )

    def encode(self, mu_batch: Tensor, cov_batch: Tensor) -> Tensor:
        mu_encs = self.pe_L.encode(mu_batch, None)

        cov_features = torch.flatten(torch.triu(torch.sign(cov_batch) * torch.sqrt(torch.abs(cov_batch))), start_dim=1)
        cov_encs = self.pe_2.encode(cov_features, None)

        gamma = torch.concatenate((mu_encs, cov_encs), dim=1)
        return gamma

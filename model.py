import torch
from torch import nn

class PositionalEncoder(nn.Module):

    def __init__(self, d_in: int, n_freqs: int):
        """
        :param d_in: amount of input dimensions (e.g 1, 2, 3)
        :param n_freqs: number of frequencies to map inputs to
        """

        super().__init__()
        self.d_input = d_in
        self.n_freqs = n_freqs
        self.d_output = d_in * (2 * self.n_freqs)

        # compute frequencies
        self.freq_powers = torch.arange(0, n_freqs)
        self.frequencies = 2 ** self.freq_powers

        # create embedding functions alternating sin and cosine
        self.embed_fns = []
        for freq in self.frequencies:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))

    def forward(self, x) -> torch.Tensor:
        """
        apply positional encoding to input.
        :param x tensor of size (batch, N_in)
        """
        return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)

class IntegratedPositionalEncoderIsotropic(PositionalEncoder):

    def __init__(self, d_in: int, n_freqs: int):
        """
        :param d_in: amount of input dimensions (e.g 1, 2, 3)
        :param n_freqs: number of frequencies to map inputs to
        """
        super().__init__(d_in, n_freqs)

        self.embed_fns = []
        for f_n in self.freq_powers:
            self.embed_fns.append(lambda mu, sigma, f_n=f_n: torch.sin(2 ** f_n * mu) * torch.exp(
                -0.5 * 4 ** f_n * torch.hstack([sigma] * d_in)))
            self.embed_fns.append(lambda mu, sigma, f_n=f_n: torch.cos(2 ** f_n * mu) * torch.exp(
                -0.5 * 4 ** f_n * torch.hstack([sigma] * d_in)))

    def forward(self, x) -> torch.Tensor:
        """
        :param x: tensor of size (batch, d_in + d_in^2) representing the mean vector and covariance matrix
        """

        B, _ = x.shape

        # extract means and covariances from input tensor
        means = x[:, 0:self.d_input]
        sigmas = x[:, self.d_input:]

        return torch.concat([fn(means, sigmas) for fn in self.embed_fns], dim=-1)

class MLP(nn.Module):
    """
    Simple Coordinate based MLP
    """

    def __init__(self, d_input=2, n_hidden=3, d_output=3, hidden_size=128):
        super().__init__()

        self.d_input = d_input
        self.d_output = d_output
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        self.activation = nn.ReLU()

        # construct layers
        layers = ([nn.Linear(d_input, hidden_size), self.activation] +
                  [nn.Linear(hidden_size, hidden_size), self.activation] * n_hidden +
                  [nn.Linear(hidden_size, d_output), nn.Sigmoid()])

        self.ff = nn.Sequential(*layers)

    def forward(self, x):
        return self.ff(x)

    def config(self):
        return dict(
            d_input=self.d_input,
            n_hidden=self.n_hidden,
            d_output=self.d_output,
            hidden_size=self.hidden_size
        )

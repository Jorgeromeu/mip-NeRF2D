from unittest import TestCase

import torch

from enc import PE, IPE

class TestPE(TestCase):

    def setUp(self) -> None:
        self.B = 10

        self.d_in = 2
        self.n_freqs = 10
        self.mus = torch.randn(self.B, self.d_in)
        self.covs = torch.randn(self.B, self.d_in, self.d_in)

        self.encoder = PE(self.d_in, n_freqs=self.n_freqs)

    def test_pe_shape(self):
        pe = PE(self.d_in, n_freqs=self.n_freqs)
        y = pe.encode(self.mus, self.covs)

        expected_shape = torch.Size([self.B, self.encoder.d_output])
        self.assertEquals(y.shape, expected_shape)

    def test_ipe_shape(self):
        ipe = IPE(self.d_in, self.n_freqs)
        y = ipe.encode(self.mus, self.covs)

        print(y.shape)

    def test_ipe(self):
        ipe = IPE(self.d_in, self.n_freqs)

        y = ipe.encode(self.mus, self.covs)

        print(y.shape)

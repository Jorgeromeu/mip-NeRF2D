from unittest import TestCase

import matplotlib.pyplot as plt

from enc import IPE
from model import MLP
from model_evaluation import nerf_forward

class Test(TestCase):
    def test_nerf_forward(self):
        ipe = IPE(d_in=2, n_freqs=10)
        model = MLP(ipe.d_output)

        res = 10
        render = nerf_forward(model, ipe, res, 0.1)

        plt.imshow(render.permute(1, 2, 0).detach().numpy())
        plt.show()

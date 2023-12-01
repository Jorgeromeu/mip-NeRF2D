from unittest import TestCase

import numpy as np
import torch
from einops import rearrange
from matplotlib import pyplot as plt

from gaussian import Gaussian2D

def image_points(res):
    xs = (torch.arange(0, res) + 0.5) / res
    ys = (torch.arange(0, res) + 0.5) / res

    x, y = torch.meshgrid(xs, ys)
    points = torch.dstack((y, x))
    return points

def visualize_gaussian(im: torch.Tensor, gaussians: list[Gaussian2D], N_points=100):
    gaussian_samples = [g.scipy_gaussian().rvs(size=N_points) for g in gaussians]
    gaussian_pdfs = [g.scipy_gaussian().pdf(samples) for g, samples in zip(gaussians, gaussian_samples)]

    samples = np.concatenate(gaussian_samples)
    pdfs = np.concatenate(gaussian_pdfs)

    plt.scatter(x=samples[:, 0], y=samples[:, 1], c=pdfs)
    plt.imshow(im.permute(1, 2, 0), extent=[0, 1, 1, 0])
    plt.show()

class Test(TestCase):

    def test_visualize_gaussians(self):
        res = 10
        im = np.random.uniform(size=(res, res))
        im = torch.Tensor(im)

        points = image_points(res)
        mus = rearrange(points, 'h w dim -> (h w) dim')

        pixel_width = 1 / res
        scale = pixel_width * 0.3

        gs = [Gaussian2D(0, np.array([scale, scale]), mu) for mu in mus]
        visualize_gaussian(im, gs)

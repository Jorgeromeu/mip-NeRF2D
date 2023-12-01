from unittest import TestCase

import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F
from torchvision.io import read_image

from utils import image_points
from utils import sample_bilinear

class Test(TestCase):

    def test_sample_bilinear(self):
        im = read_image('../data/lena.png')
        shrunk = F.resize(im, 8, antialias=True)

        res_hd = 100
        hd = torch.zeros((3, res_hd, res_hd))

        points = image_points(res_hd)

        for i in range(res_hd):
            for j in range(res_hd):
                p = points[i, j]
                color = sample_bilinear(shrunk, p)
                hd[:, i, j] = color

        hd /= 255

        fig, axs = plt.subplots(1, 2)

        axs[0].imshow(shrunk.permute(1, 2, 0))
        axs[1].imshow(hd.permute(1, 2, 0))
        plt.show()

import numpy as np
import torch
from einops import rearrange
from torch import nn

import utils
from enc import Encoder
from gaussian import Gaussian2D


def res_to_scale(res, slope=0.3, intercept=0):
    return slope / res + intercept

def nerf_forward(
        model: nn.Module,
        enc: Encoder,
        res: int,
        scale: float,
        device='cpu') -> torch.Tensor:

    means = utils.image_points(res)
    means = rearrange(means, 'h w dim -> (h w) dim')

    cov = torch.Tensor(Gaussian2D(0, np.array([scale, scale]), means[0]).covariance_matrix())
    covs = torch.stack([cov for _ in range(len(means))])

    inputs = means.to(device)
    inputs_enc = enc.encode(inputs, covs)

    out = model.to(device)(inputs_enc)
    # convert output to image
    image = rearrange(out, '(h w) c -> c h w', h=res, w=res)

    return image

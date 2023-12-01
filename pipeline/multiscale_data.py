from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import wandb
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset
from torchvision.io import read_image
from tqdm import trange
from tqdm.contrib import tenumerate

import wandb_util
from gaussian import Gaussian2D
from utils import sample_bilinear

@dataclass
class MultiscaleImageDataset:
    means: Tensor
    covs: Tensor
    colors: Tensor

    def torch_dataset(self) -> Dataset:
        return TensorDataset(self.means, self.covs, self.colors)

    def to_artifact(self, artifact_name):
        # save dataset
        artifact = wandb.Artifact(name=artifact_name, type='multiscale-dataset')

        arrays = [
            ('means.npy', self.means),
            ('covs.npy', self.covs),
            ('colors.npy', self.colors)
        ]

        for fname, arr in arrays:
            with artifact.new_file(fname, 'wb') as f:
                np.save(f, arr)

        return artifact

def signal_under_gaussian(gaussian: Gaussian2D, signal, N_points=100):
    """
    Compute the value of the signal under the gaussian
    :param gaussian:
    :param signal:
    :param N_points:
    :return:
    """

    gaussian_scipy = gaussian.scipy_gaussian()

    # sample points
    samples = gaussian_scipy.rvs(size=N_points)

    # filter samples from out of domain
    samples = torch.Tensor(np.array([s for s in samples if (s >= 0).all() and (s <= 1).all()]))

    if len(samples) == 0:
        return torch.zeros(3)

    # get pdf of each sample
    samples_pdf = gaussian_scipy.pdf(samples)
    samples_weights = samples_pdf / samples_pdf.sum()

    colors = torch.zeros(len(samples), 3)
    for i in range(len(samples)):
        colors[i] = signal(samples[i])

    wsum = torch.sum(colors * samples_weights.reshape(-1, 1), dim=0)

    return wsum

def sample_gaussians_isotropic(min_scale, max_scale, N_gaussians=100) -> list[Gaussian2D]:
    means = np.random.uniform(0, 1, size=(N_gaussians, 2))
    scales = np.random.uniform(min_scale, max_scale, size=N_gaussians)

    gaussians = [Gaussian2D(0, np.array([scale, scale]), mean) for mean, scale in zip(means, scales)]
    return gaussians

def generate_multiscale_dataset(im: torch.Tensor, min_scale, max_scale, n_gaussians=100, n_points_per_gaussian=10):
    def signal(p):
        return sample_bilinear(im, p)

    gaussians = sample_gaussians_isotropic(N_gaussians=n_gaussians, min_scale=min_scale, max_scale=max_scale)

    colors = np.zeros((n_gaussians, 3))
    for i in trange(n_gaussians):
        colors[i] = signal_under_gaussian(gaussians[i], signal, N_points=n_points_per_gaussian)

    means = torch.Tensor([g.position for g in gaussians])
    covs = torch.Tensor([g.covariance_matrix() for g in gaussians])

    return means, covs, colors

def generate_multiscale_image_dataset(
        image_artifact: str,
        dataset_artifact_name: str,
        gaussians: list[Gaussian2D],
        n_points_per_gaussian=30,
):
    wandb.init(project=wandb_util.PROJECT_NAME, job_type='create-dataset')

    # read image
    image = wandb.use_artifact(image_artifact)
    image_dir = Path(image.download())
    im_path = image_dir / 'image.png'
    im = read_image(str(im_path)) / 255

    def signal(p):
        return sample_bilinear(im, p)

    # estimate image under each gaussian
    colors = np.zeros((len(gaussians), 3))
    for i, g in tenumerate(gaussians):
        colors[i] = signal_under_gaussian(g, signal, N_points=n_points_per_gaussian)

    means = np.array([g.position for g in gaussians])
    covs = np.array([g.covariance_matrix() for g in gaussians])

    # save dataset
    dataset = MultiscaleImageDataset(means, covs, colors)
    dataset_artifact = dataset.to_artifact(dataset_artifact_name)

    # save artifact metadata
    dataset_artifact.metadata['n_gaussians'] = len(gaussians)
    dataset_artifact.metadata['n_points_per_gaussian'] = n_points_per_gaussian

    wandb.log_artifact(dataset_artifact)
    wandb.finish()

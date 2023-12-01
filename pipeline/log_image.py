from dataclasses import dataclass
from pathlib import Path

import torch
import wandb
from torchvision.io import read_image

import wandb_util

@dataclass
class TrainImage:
    im: torch.Tensor
    displayname: str

    @classmethod
    def from_artifact(cls, artifact):
        art_dir = Path(artifact.download())
        image_path = art_dir / 'image.png'
        ground_truth = read_image(str(image_path)).div(255)
        displayname = artifact.get('displayname')
        return TrainImage(ground_truth, displayname)

def log_image(
        image_path: Path,
        artifact_name: str,
        image_displayname='Image'
):
    """
    Create a wandb artifact for a particular image

    :param artifact_name: name of the artifact
    :param image_path: path to image file
    :param image_displayname: display name of image for plotting
    :return:
    """

    wandb.init(project=wandb_util.PROJECT_NAME, job_type='create-image-artifact')

    artifact = wandb.Artifact(name=artifact_name, type='image')
    artifact.add_file(image_path, 'image.png')
    artifact.metadata['display-name'] = image_displayname
    wandb.log_artifact(artifact)
    wandb.finish()

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as F
import wandb
from torch import nn, Tensor
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from torchmetrics.functional.image import structural_similarity_index_measure
from torchvision.io import read_image
from tqdm import trange

import wandb_util
from enc import Encoder
from model_evaluation import nerf_forward, res_to_scale
from pipeline.multiscale_data import MultiscaleImageDataset

@dataclass
class TrainingSettings:
    n_epochs = 20
    batch_size = 200
    lr = 0.005
    device = 'cpu'

def load_dataset(artifact: wandb.Artifact):
    artifact_dir = Path(artifact.download())
    means = np.load(artifact_dir / 'means.npy')
    covs = np.load(artifact_dir / 'covs.npy')
    colors = np.load(artifact_dir / 'colors.npy')

    dataset = MultiscaleImageDataset(Tensor(means), Tensor(covs), Tensor(colors))
    return dataset

def train_model(model: nn.Module,
                encoding: Encoder,
                dataset_artifact: str,
                training_settings: TrainingSettings,
                output_artifact=None,
                run_name=None,
                group_name=None,
                no_init=False):
    if not no_init:
        wandb.init(project=wandb_util.PROJECT_NAME,
                   job_type='training',
                   name=run_name,
                   group=group_name)

    # hyperparameters
    wandb.config['training_settings'] = asdict(training_settings)
    wandb.config['model_settings'] = model.config()
    wandb.config['encoding_settings'] = encoding.config()

    # load dataset
    artifact = wandb.use_artifact(dataset_artifact)
    dataset = load_dataset(artifact)

    # find ground truth image
    dataset_creation_run = artifact.logged_by()
    input_image_artifact = wandb_util.first_used_artifact_of_type(dataset_creation_run, 'image')

    input_image_artifact_dir = Path(input_image_artifact.download())
    image_path = input_image_artifact_dir / 'image.png'
    ground_truth_full = read_image(str(image_path)).div(255)

    resolutions = {
        'full': 512,
        'mid': 258,
        'low': 64,
        'ultralow': 16
    }

    # get downsampled resolutions
    ground_truths = {label: F.resize(ground_truth_full, res, antialias=True) for label, res in resolutions.items()}

    # track gradients
    wandb.watch(model)

    # enter training mode and move to device
    model.to(training_settings.device)
    model.train()

    torch_dataset = dataset.torch_dataset()

    n_train = int(0.7 * len(torch_dataset))
    n_val = int(0.3 * len(torch_dataset))

    train_set, val_set = torch.utils.data.random_split(torch_dataset, [n_train, n_val])

    # create dataloader
    train_dataloader = DataLoader(
        train_set,
        batch_size=training_settings.batch_size
    )

    # create dataloader
    val_dataloader = DataLoader(
        val_set,
        batch_size=64
    )

    # setup optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=training_settings.lr)
    criterion = torch.nn.MSELoss()

    wandb.define_metric('loss', summary='min')
    wandb.define_metric('val_loss', summary='min')
    wandb.define_metric('epoch')
    wandb.define_metric('output', step_metric='epochs')

    wandb.watch(model, criterion, log='all', log_freq=10)

    for epoch in trange(training_settings.n_epochs, desc='epochs'):

        for i, (means, covs, colors) in enumerate(train_dataloader):
            # move tensors to device

            encodings = encoding.encode(means, covs)
            encodings = encodings.to(training_settings.device)
            colors = colors.to(training_settings.device)

            preds = model(encodings)
            loss = criterion(preds, colors)

            wandb.log({'loss': loss})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()

        # evaluate render at multiple scales
        with torch.no_grad():

            renders = {label: nerf_forward(model, encoding, res, res_to_scale(res), device=training_settings.device) for
                       label, res in resolutions.items()}

            render_mses = {label: mse_loss(renders[label], ground_truths[label]) for label in resolutions.keys()}
            render_ssims = {label: structural_similarity_index_measure(renders[label].unsqueeze(0),
                                                                       ground_truths[label].unsqueeze(0))
                            for label in resolutions.keys()}

        # compute validation loss on dataset
        with torch.no_grad():

            val_loss = 0.0

            for i, (means, covs, colors) in enumerate(val_dataloader):
                encodings = encoding.encode(means, covs)
                encodings = encodings.to(training_settings.device)
                colors = colors.to(training_settings.device)

                preds = model(encodings)
                loss = criterion(preds, colors)
                val_loss += loss.item()

        log_dict = {
            'epoch': epoch,
            'val_loss': val_loss
        }

        log_dict.update({f'render_{label}_mse': mse for label, mse in render_mses.items()})
        log_dict.update({f'render_{label}_ssim': mse for label, mse in render_ssims.items()})
        log_dict.update({f'render_{label}': wandb.Image(render.detach().permute(1, 2, 0).cpu().numpy())
                         for label, render in renders.items()})

        wandb.log(log_dict)

        # go back to training mode
        model.train()

    model.eval()

    # save trained model
    if output_artifact is not None:
        model_artifact = wandb.Artifact(output_artifact, type='model')

        with model_artifact.new_file('state_dict.pt') as f:
            torch.save(model.state_dict(), f.name)

        with model_artifact.new_file('enc_config.json') as f:
            json.dump(encoding.config(), f)

        with model_artifact.new_file('model_config.json') as f:
            json.dump(model.config(), f)

        wandb.log_artifact(model_artifact)

    wandb.finish()

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as F
import wandb
from torchmetrics.functional.image import structural_similarity_index_measure, peak_signal_noise_ratio
from tqdm import tqdm

import wandb_util
from enc import Encoder
from enc import build_model
from model import MLP
from model_evaluation import nerf_forward
from model_evaluation import res_to_scale
from pipeline.log_image import TrainImage

@dataclass
class TrainedModel:
    enc: Encoder
    model: MLP
    ground_truth: TrainImage
    dataset_artifact: wandb.Artifact

def load_model(model_artifact: wandb.Artifact):
    model_artifact_dir = Path(model_artifact.download())

    with open(model_artifact_dir / 'enc_config.json', 'r') as f:
        enc_config = json.load(f)

    with open(model_artifact_dir / 'model_config.json', 'r') as f:
        model_config = json.load(f)

    state_dict = torch.load(model_artifact_dir / 'state_dict.pt')

    enc, model = build_model(enc_config, model_config)
    model.load_state_dict(state_dict)

    training_run = model_artifact.logged_by()
    dataset_artifact = [a for a in training_run.used_artifacts() if a.type == 'multiscale-dataset'][0]
    dataset_creation_run = dataset_artifact.logged_by()
    input_image_artifact = [a for a in dataset_creation_run.used_artifacts() if a.type == 'image'][0]

    ground_truth = TrainImage.from_artifact(input_image_artifact)

    return TrainedModel(enc, model, ground_truth, dataset_artifact)

def save_renders(artifact_name: str, renders: list[torch.Tensor]):
    renders_artifact = wandb.Artifact(name=artifact_name, type='renders')

    renders_np = [r.detach().numpy() for r in renders]

    with renders_artifact.new_file('renders.pkl', mode='wb') as f:
        pickle.dump(renders_np, f)

    wandb.log_artifact(renders_artifact)

def evaluate_model_multires(
        model_artifact: str,
        resolutions,
        run_name: str = None,
        group_name: str = None,
        output_artifact_name: str = None

):
    wandb.init(project=wandb_util.PROJECT_NAME, job_type='render-multires',
               name=run_name, group=group_name)

    model_artifact = wandb.use_artifact(model_artifact)
    trained_model = load_model(model_artifact)

    renders = []
    for res in resolutions:
        render = nerf_forward(trained_model.model, trained_model.enc, res, res_to_scale(res))
        truth = F.resize(trained_model.ground_truth.im, int(res), antialias=True)

        # compare to truths
        mse = torch.nn.functional.mse_loss(render, truth)
        ssim = structural_similarity_index_measure(render.unsqueeze(0), truth.unsqueeze(0))
        pnsr = peak_signal_noise_ratio(render.unsqueeze(0), truth.unsqueeze(0))

        wandb.log({
            'render': wandb.Image(render),
            'res': res,
            'mse': mse,
            'ssim': ssim,
            'pnsr': pnsr,
        })

        renders.append(render)

    if output_artifact_name is not None:
        save_renders(output_artifact_name, renders)

    run = wandb.run
    wandb.finish()
    return run

def evaluate_model_across_scales(
        model_artifact: str,
        lo: float, hi: float, count=10,
        res=100,
        run_name: str = None,
        group_name: str = None,
        assume_constant=False
):
    wandb.init(project=wandb_util.PROJECT_NAME,
               job_type='evaluation-render-multiscale',
               group=group_name,
               name=run_name)

    wandb.config['res'] = res

    # load model
    model_artifact = wandb.use_artifact(model_artifact)
    trained_model = load_model(wandb.use_artifact(model_artifact))

    wandb.config['model_settings'] = trained_model.model.config()
    wandb.config['encoding_settings'] = trained_model.enc.config()

    # ground truth resized image
    ground_truth = F.resize(trained_model.ground_truth.im, res, antialias=True)

    wandb.log({
        'truth': wandb.Image(ground_truth)
    })

    # get scales
    scales = np.linspace(lo, hi, count)

    if assume_constant:
        scale = scales[0]
        render = nerf_forward(trained_model.model, trained_model.enc, res, scale)

        mse = torch.nn.functional.mse_loss(render, ground_truth)
        ssim = structural_similarity_index_measure(render.unsqueeze(0), ground_truth.unsqueeze(0))
        pnsr = peak_signal_noise_ratio(render.unsqueeze(0), ground_truth.unsqueeze(0))

        for scale in scales:
            wandb.log({
                'render': wandb.Image(render),
                'scale': scale,
                'pnsr': pnsr,
                'ssim': ssim,
                'mse': mse
            })

        wandb.finish()
        return

    for scale in tqdm(scales, desc='scales'):
        render = nerf_forward(trained_model.model, trained_model.enc, res, scale)

        mse = torch.nn.functional.mse_loss(render, ground_truth)
        ssim = structural_similarity_index_measure(render.unsqueeze(0), ground_truth.unsqueeze(0))
        pnsr = peak_signal_noise_ratio(render.unsqueeze(0), ground_truth.unsqueeze(0))

        wandb.log({
            'render': wandb.Image(render),
            'scale': scale,
            'pnsr': pnsr,
            'ssim': ssim,
            'mse': mse
        })

    wandb.finish()

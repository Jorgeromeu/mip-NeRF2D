from dataclasses import dataclass

from torch import nn

from enc import PE, IPE, Encoder
from model import MLP
from pipeline.train import TrainingSettings, train_model

@dataclass
class ExperimentModel:
    # info about saved model
    display_name: str
    artifact_name: str

    model: nn.Module
    encoding: Encoder
    training_set_artifact: str

def default_models():
    pe = PE(d_in=2, n_freqs=10)
    ipe = IPE(d_in=2, n_freqs=10)
    model = MLP(pe.d_output, n_hidden=4, d_output=3)

    models = [
        ExperimentModel('PE-single', 'pe-single', model, pe, 'collins-single-macro:latest'),
        ExperimentModel('IPE-single', 'ipe-single', model, ipe, 'collins-single-macro:latest'),
        ExperimentModel('PE-multi', 'pe-multi', model, pe, 'collins-multi-macro:latest'),
        ExperimentModel('IPE-multi', 'ipe-multi', model, ipe, 'collins-multi-macro:latest')
    ]

    return models

def train_models(models, train_settings, re_train=True):
    if re_train:
        for model in models:
            train_model(model.model, model.encoding,
                        f'{model.training_set_artifact}:latest',
                        train_settings,
                        output_artifact=model.artifact_name,
                        run_name=f'train-{model.display_name}')

def default_training_settings():
    train_settings = TrainingSettings()
    train_settings.batch_size = 4000
    train_settings.n_epochs = 20
    train_settings.lr = 0.007
    train_settings.batch_size = 2200
    train_settings.n_epochs = 25
    train_settings.n_samples = 100000
    return train_settings

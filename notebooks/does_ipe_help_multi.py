# define models
from enc import IPE, PE, AltEnc
from experiment_util import default_training_settings, ExperimentModel
from model import MLP
from model_evaluation import res_to_scale
from pipeline.evaluation import evaluate_model_across_scales

EXPERIMENT_NAME = 'ablate-IPE-multiscale'

dataset_multi = 'collins-truemulti'
dataset_single = 'collins-single'

ipe = IPE(d_in=2, n_freqs=20)
pe = PE(d_in=2, n_freqs=10)
alt = AltEnc(d_in=2, n_freqs=10)

# define train settings
train_settings = default_training_settings()
train_settings.lr = 0.007
train_settings.batch_size = 2200
train_settings.n_epochs = 25

models_multi = [
    # ExperimentModel('PE-multi', f'pe-{EXPERIMENT_NAME}', MLP(pe.d_output), pe, dataset_multi),
    # ExperimentModel('IPE-multi', f'ipe-{EXPERIMENT_NAME}', MLP(ipe.d_output), ipe, dataset_multi),
    ExperimentModel('ALT-multi', f'ipe-{EXPERIMENT_NAME}', MLP(alt.d_output), alt, dataset_multi),
]

# train_models(models_multi, train_settings)
evaluate_model_across_scales(models_multi[0], 0, res_to_scale(8), 30)

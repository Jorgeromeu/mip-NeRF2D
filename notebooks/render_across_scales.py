from enc import IPE, PE
from experiment_util import ExperimentModel, train_models, default_training_settings
from model import MLP
from model_evaluation import res_to_scale
from pipeline.evaluation import evaluate_model_across_scales

EXPERIMENT_NAME = 'compare_across_scales'
re_train = False
n_scales = 50

def run_experiment(models: list[ExperimentModel],
                   train_settings,
                   render_res,
                   re_train=False):
    # train all models
    train_models(models, train_settings, re_train=re_train)

    scale_lo = 0
    scale_hi = res_to_scale(9)

    for model in models:
        evaluate_model_across_scales(f'{model.artifact_name}:latest',
                                     scale_lo, scale_hi,
                                     count=n_scales,
                                     run_name=f'{model.display_name}-{render_res}',
                                     res=render_res,
                                     assume_constant=model.encoding.type == 'PE')

if __name__ == '__main__':

    # define models
    dataset_multi = 'collins-truemulti'
    dataset_multi_even_res = 'collins-multi'
    dataset_single = 'collins-single'

    ipe = IPE(d_in=2, n_freqs=20)
    pe = PE(d_in=2, n_freqs=10)

    # define train settings
    train_settings = default_training_settings()
    train_settings.lr = 0.007
    train_settings.batch_size = 2200
    train_settings.n_epochs = 25

    models_multi = [
        ExperimentModel('PE-multi', f'pe-{EXPERIMENT_NAME}', MLP(pe.d_output), pe, dataset_multi),
        ExperimentModel('IPE-multi', f'ipe-{EXPERIMENT_NAME}', MLP(ipe.d_output), ipe, dataset_multi),
    ]

    models_multi_even = [
        ExperimentModel('PE-multi-even', f'pe-{EXPERIMENT_NAME}', MLP(pe.d_output), pe, dataset_multi_even_res),
        ExperimentModel('IPE-multi-even', f'ipe-{EXPERIMENT_NAME}', MLP(ipe.d_output), ipe, dataset_multi_even_res),
    ]

    models_single = [
        ExperimentModel('PE-single', f'pe-single-{EXPERIMENT_NAME}', MLP(pe.d_output), pe, dataset_single),
        ExperimentModel('IPE-single', f'ipe-single-{EXPERIMENT_NAME}', MLP(ipe.d_output), ipe, dataset_single),
    ]

    for res in [32, 64, 128, 256, 512]:
        run_experiment(models_multi_even,
                       train_settings,
                       render_res=res,
                       re_train=False)

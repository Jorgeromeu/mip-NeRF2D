# mip-NeRF2D

Evaluating Integrated Positional Encoding from mip-NeRF on 2D image reconstruction

## Repository structure

Code is intended to run with `wandb`. The wandb project name can be configured in `wandb_util.py`

- `data/` contains some standard 512x512 test images
- `model.py` and `enc.py` define the actual models and encodings
- `model_evaluation.py` contains code for reconstructiong a 2D image given a trained model
- `pipeline.train.py` contains model traning procedure
- `pipeline.evaluation.py` contains scripts for evaluating trained models on different tasks
- `pipeline.{log_image.py, multiscale_data.py}` logic for creating datasets from 2D images
- `notebooks` consists of jupyter notebooks, using the functions defined in `pipeline` for each experiment

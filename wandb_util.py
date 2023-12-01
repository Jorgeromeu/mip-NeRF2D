from wandb import Artifact
from wandb.apis.public import Run

PROJECT_NAME = 'mip-NeRF2D'

def first_used_artifact_of_type(run: Run, artifact_type: str) -> Artifact:
    return [a for a in run.used_artifacts() if a.type == artifact_type][0]

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nuplan-devkit'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Diffusion-Planner'))

_CKPT = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'model.pth')
_ARGS = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'args.json')


@pytest.fixture(scope="session")
def model():
    import diff_adapter as da
    da.load_model(_CKPT, _ARGS)
    return da._model


@pytest.fixture(scope="session")
def config():
    import diff_adapter as da
    da.load_model(_CKPT, _ARGS)
    return da._config


@pytest.fixture(scope="session")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

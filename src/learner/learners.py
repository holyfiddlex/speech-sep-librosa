"""Learner definitions"""

from pathlib import Path

from fastai.vision.models import resnet34
from fastai.vision.learner import unet_learner
from fastai.losses import CrossEntropyLossFlat

from src.pipeline.dataloader import create_poi_dataloader

def create_poi_learner():
    """Function to create learner for person of interest learner"""
    dls = create_poi_dataloader(Path("../../../Music/LibriSpeech/test-clean/"), 61)
    learner = unet_learner(
        dls,
        resnet34,
        loss_func=CrossEntropyLossFlat(axis=1),
        self_attention=True,
        n_out=1,
    )
    return learner

import os

import torch

from mohawk.models import BaseModel
from typing import Optional
import skbio
import numpy as np
from mohawk._format import encode_sequence


def classify(model: BaseModel,  # assumes a trained model
             length: int,
             data_path: str,  # path to a FASTA file
             level: Optional[str] = 'genus',  # TODO might not need
             batch_size: Optional[int] = 1,
             format_kwargs: Optional[dict] = None,
             format_: Optional[str] = 'fasta',  # input to skbio.io.read
             output_directory: Optional[str] = None):

    # TODO may need some map from output to label -> could keep a dictionary
    #  in the model attributes ?
    if output_directory is None:
        output_directory = os.path.curdir

    sequences = list(skbio.io.read(data_path,
                                   format=format_,
                                   **format_kwargs))

    encoded = np.asarray([encode_sequence(seq) for seq in sequences])

    # TODO support to split each read in encoded into `length`-mers and keep
    #  some form of identifier for each of the sequence so they can be
    #  mapped back

    # TODO turn `encoded` into a dataset/dataloader

    # TODO predict on `encoded` dataset

    # TODO write out results to output_directory


def load_trained_model(filepath):
    # filepath has to be a path to a model saved with
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    return model

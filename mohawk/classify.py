import os

import torch

from mohawk.models import BaseModel
from typing import Union, Optional
import skbio
import numpy as np
from mohawk._format import encode_sequence


def classify(model: Union[BaseModel, str],  # assumes a trained model,
             # can pass path
             length: int,
             data_path: str,  # path to a FASTA file
             pad_value: Optional[int] = 0,
             level: Optional[str] = 'genus',  # TODO might not need
             batch_size: Optional[int] = 1,
             format_kwargs: Optional[dict] = None,
             format_: Optional[str] = 'fasta',  # input to skbio.io.read
             output_directory: Optional[str] = None):

    # TODO may need some map from output to label -> could keep a dictionary
    #  in the model attributes ?
    if output_directory is None:
        output_directory = os.path.curdir

    if isinstance(model, str):
        model = load_trained_model(model)

    sequences = list(skbio.io.read(data_path,
                                   format=format_,
                                   **format_kwargs))

    encoded = [(encode_sequence(seq), seq.metadata['id']) for seq in sequences]

    # TODO support to split each read in encoded into `length`-mers and keep
    #  some form of identifier for each of the sequence so they can be
    #  mapped back
    # TODO make function and move to _format.py
    encoded_sequences, ids = ensure_lengths(encoded,
                                            length,
                                            pad_value=pad_value)

    encoded_sequences = np.array(encoded_sequences)

    # TODO turn `encoded` into a dataset/dataloader

    # TODO predict on `encoded` dataset

    # TODO write out results to output_directory


# TODO maybe rename `encoded`
def ensure_lengths(encoded, length, pad_value=0):
    all_sequences = []
    all_ids = []
    for sequence, id_ in encoded:
        if len(sequence) < length:
            # if sequence is not long enough, add padding
            this_sequence = sequence.copy()
            while len(this_sequence) < length:
                this_sequence.append(pad_value)
            all_sequences.append(this_sequence)
            all_ids.append(id_)
        else:
            for start_index in range(len(sequence) - length + 1):
                this_sequence = sequence[start_index: start_index + length]
                all_sequences.append(this_sequence)
                all_ids.append(id_)

    return all_sequences, all_ids


def load_trained_model(filepath):
    # filepath has to be a path to a model saved with
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    return model

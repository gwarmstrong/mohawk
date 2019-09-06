import torch

from mohawk.models import BaseModel
from typing import Union, Optional
import skbio
import numpy as np
import pandas as pd
from mohawk._format import encode_sequence
from mohawk.trainer import prepare_dataloader


def classify(model: Union[BaseModel, str], length: int, data_path: str,
             pad_value: Optional[int] = 0, batch_size: Optional[int] = 1,
             format_kwargs: Optional[dict] = None,
             format_: Optional[str] = 'fasta'):

    # TODO may need some map from output to label -> could keep a dictionary
    #  in the model attributes ?
    if format_kwargs is None:
        format_kwargs = dict()

    if isinstance(model, str):
        model = load_trained_model(model)

    model.double()

    print("Read sequences...")
    sequences = list(skbio.io.read(data_path,
                                   format=format_,
                                   **format_kwargs))
    print("Done.")

    print("Encode sequences...")
    encoded = [(encode_sequence(seq), seq.metadata['id']) for seq in sequences]
    print("Done.")

    # TODO support to split each read in encoded into `length`-mers and keep
    #  some form of identifier for each of the sequence so they can be
    #  mapped back
    # TODO make function and move to _format.py
    print("Ensure lengths...")
    encoded_sequences, ids = ensure_lengths(encoded,
                                            length,
                                            pad_value=pad_value)
    print("Done.")
    print("Data size: {}".format(encoded_sequences.shape))
    print("Prepare dataloader...")
    # TODO turn `encoded` into a dataset/dataloader
    dataloader = prepare_dataloader(encoded_sequences,
                                    ids=ids,
                                    batch_size=batch_size)
    print("Done.")

    print("Predicting...")
    # TODO predict on `encoded` dataset
    all_ids = []
    all_class_predictions = []
    for index_epoch, data in enumerate(dataloader):
        print("it: {} out of {}".format(index_epoch, len(dataloader) - 1))
        epoch_ids = data['id']
        reads = data['read']
        # TODO here would be the place to put to device
        softmax_predictions = model(reads)
        predictions = softmax_predictions.argmax(1).numpy()
        all_ids.append(epoch_ids)
        all_class_predictions.append(predictions)

    all_ids = np.concatenate(all_ids)
    all_class_predictions = np.concatenate(all_class_predictions)
    print("Done.")

    transformed_predictions = model.class_encoder.inverse_transform(
        all_class_predictions)

    results = pd.DataFrame({'ids': all_ids,
                            'predictions': transformed_predictions})

    return results


# TODO maybe rename `encoded`
def ensure_lengths(encoded, length, pad_value=0):
    all_sequences = []
    all_ids = []
    for sequence, id_ in encoded:
        if len(sequence) < length:
            raise ValueError('All reads should have length {}.'.format(length))
            # if sequence is not long enough, add padding
            # this_sequence = sequence.copy()
            # pad_length = length - len(sequence)
            # this_sequence = np.append(this_sequence, [pad_value] * pad_length)
            # all_sequences.append(this_sequence)
            # all_ids.append(id_ + '__idx_0')
        elif len(sequence) > length:
            raise ValueError('All reads should have length {}.'.format(length))
            # for start_index in range(len(sequence) - length + 1):
            #     this_sequence = sequence[start_index: start_index + length]
            #     all_sequences.append(this_sequence)
            #     all_ids.append(id_ + '__idx_{}'.format(start_index))
        else:
            all_sequences.append(sequence)
            all_ids.append(id_)

    all_sequences = np.vstack(all_sequences)
    return all_sequences, all_ids


def load_trained_model(filepath):
    # filepath has to be a path to a model saved with
    checkpoint = torch.load(filepath, map_location='cpu')
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])

    model.class_encoder = checkpoint.get('class_encoder', None)

    model.eval()

    return model

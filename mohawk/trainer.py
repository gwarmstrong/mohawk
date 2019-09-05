import os
import numpy as np
import pandas as pd
from mohawk.data_downloader import data_downloader
from mohawk.simulation import simulate_from_genomes, id_to_lineage
from mohawk.dataset import SequenceDataset
from typing import Optional, List
from mohawk.models import BaseModel
from torch.utils.data import DataLoader


def trainer(model: BaseModel, distribution: List[float], total_reads: int,
            length: int, train_ratio: float,
            id_list: Optional[List[str]],
            level: Optional[str] = 'genus',
            channel: Optional[str] = 'representative',
            batch_size: Optional[int] = 1,
            data_directory: Optional[str] = None,
            random_seed: Optional[int] = None,
            external_validation_ids: Optional[List[str]] = None,
            n_external_validation_reads: Optional[int] = None,
            external_validation_distribution: Optional[List[float]] = None,
            weight: Optional[bool] = False,
            distribution_noise: Optional[bool] = True,
            model_kwargs: Optional[dict] = None,
            train_kwargs: Optional[dict] = None,
            summary_kwargs: Optional[dict] = None):

    if model_kwargs is None:
        model_kwargs = dict()
    if train_kwargs is None:
        train_kwargs = dict()
    if summary_kwargs is None:
        summary_kwargs = dict()

    # If id_list is not None, use the specified id's
    if id_list is not None:
        file_list = data_downloader(id_list,
                                    output_directory=data_directory,
                                    channel=channel)
    # if id_list _is_ None, just use whatever is in the directory (handled
    # by simulate_from_genomes)
    # TODO may need some error catching for if data_directory is empty
    else:
        file_list = [os.path.join(data_directory, file_) for file_ in
                     os.listdir(data_directory)]

                 # TODO flat directory structure for data_directory
    reads, ids = simulate_from_genomes(distribution, total_reads, length,
                                       file_list, channel, data_directory,
                                       random_seed,
                                       distribution_noise=distribution_noise)

    classes = id_to_lineage(ids, level, channel)

    # TODO maybe prep external validation function ?
    external_validation = False
    external_classes = None
    if external_validation_ids is not None and \
            n_external_validation_reads is not None and \
            external_validation_distribution is not None:
        data_downloader(external_validation_ids,
                        output_directory=data_directory,
                        channel=channel)
        external_reads, external_ids = simulate_from_genomes(
            external_validation_distribution, n_external_validation_reads,
            length, external_validation_ids, channel, data_directory,
            random_seed + 5, distribution_noise=distribution_noise)

        external_validation = True
        external_classes = id_to_lineage(external_ids, level, channel)

    elif external_validation_ids is not None or \
            n_external_validation_reads is not None or \
            external_validation_distribution is not None:
        raise ValueError('If any external validation parameters are '
                         'specified, all must be specified.')

    # split into train and validation
    num_samples = len(classes)

    train_indices, val_indices = train_val_split(num_samples, train_ratio,
                                                 random_seed=random_seed)

    # create Dataset and DataLoader for train and validation
    train_dataloader = prepare_dataloader(reads,
                                          classes,
                                          ids,
                                          batch_size=batch_size,
                                          indices=train_indices)
    val_dataloader = prepare_dataloader(reads,
                                        classes,
                                        ids,
                                        batch_size=batch_size,
                                        indices=val_indices)
    if external_validation:
        external_dataloader = prepare_dataloader(external_reads,
                                                 external_classes,
                                                 external_ids,
                                                 batch_size=batch_size)
    else:
        external_dataloader = None

    if weight:
        weights = prepare_weights(train_dataloader)
    else:
        weights = None

    training_model = model(n_classes=len(set(classes)),
                           seed=random_seed,
                           **model_kwargs)

    # conv1d complains if there are float's instead of doubles <- could slow
    # down training
    training_model.double()

    if train_kwargs['gpu']:
        training_model.cuda()

    # TODO have a separate directory for trained models
    # TODO as well as prefix/suffix option for naming
    training_model.fit(train_dataloader,
                       val_dataset=val_dataloader,
                       external_dataset=external_dataloader,
                       seed=random_seed,
                       summary_kwargs=summary_kwargs,
                       **train_kwargs)

    # TODO maybe some functionality for holding onto the best model
    #  parameters ? -> I think would entail making a deepcopy of best

    return training_model


def prepare_weights(dataloader):
    all_labels = dataloader.dataset.labels
    all_labels = pd.Series(all_labels).value_counts().sort_index()
    # will error if no labels present in training data for given class
    inv = (1 / all_labels)
    weights = inv / inv.sum()
    return weights


def prepare_dataloader(reads, classes=None, ids=None, indices=None,
                       batch_size=1,
                       shuffle=False):
    # TODO right now, cannot use NONE for classes and ids if indices is not
    #  None -> should fix this so subsetting can be done of classificaiton data
    if indices is not None:
        components = prep_reads_classes_ids(reads, classes, ids,
                                            indices)
    else:
        components = reads, classes, ids

    dataset = SequenceDataset(*components)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


def prep_reads_classes_ids(reads, classes, ids, indices):
    # TODO could error if indices is full/empty
    # if indices are not specified, keep everything
    sub_reads = reads[indices]
    classes_array = np.array(classes)
    sub_classes = classes_array[indices]
    ids_array = np.array(ids)
    sub_ids = ids_array[indices]

    return sub_reads, sub_classes, sub_ids


def train_val_split(num_samples, train_ratio, random_seed=None):
    indices = np.arange(num_samples)
    if random_seed is not None:
        np.random.seed(random_seed + 1)
    permutation = np.random.permutation(indices)
    train_max_index = int(num_samples * train_ratio)
    train_indices = permutation[:train_max_index]
    test_indices = permutation[train_max_index:]
    return train_indices, test_indices
